"""ipSAE from a predicted-aligned-error matrix, computed in process.

The pipeline evaluates ipSAE for every holo prediction and every cross-validation
seed, inside workers that already hold the PAE array in memory. Shelling out to
the `ipsae` CLI would pay a process spawn and two temp-file writes per call, so
this module reimplements the score over the array directly.

Definition, per ordered chain pair (aligned -> measured):

  1. For each residue i of the aligned chain, find the residues j of the measured
     chain with PAE(i, j) < `pae_cutoff`; let n0(i) be how many there are.
  2. d0(i) = 1.24 * (n0(i) - 15)^(1/3) - 1.8, floored at 1.0. The floor is not
     arbitrary: the expression crosses 1.0 at n0 = 27, so the piecewise function
     is continuous there.
  3. Score residue i as the mean of 1 / (1 + (PAE(i, j) / d0(i))^2) over the
     passing j only.
  4. The directional ipSAE is the maximum over i.

The score is asymmetric, so binder->target and target->binder differ; the
pipeline reports the minimum of the two as `ipsae_min_<target>`.

This implementation is PAE-only. The Dunbrack reference CLI additionally accepts
a distance cutoff, so the two need not agree numerically -- the validation script
runs both on known complexes and records the agreement before this is relied on.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["ipsae_from_pae", "directional_ipsae", "d0_from_n0"]

# Below this many passing residues the d0 expression falls under 1.0; the score
# uses 1.0 there so d0 never shrinks toward zero and inflates the ratio.
_D0_CONTINUITY_POINT = 27


def d0_from_n0(n0: int) -> float:
    """TM-score style d0 for `n0` residues, floored at 1.0.

    The scalar reference the tests pin. `_d0_array` is the vectorised form used
    on the hot path; the two must agree.
    """
    if n0 < _D0_CONTINUITY_POINT:
        return 1.0
    return 1.24 * (n0 - 15) ** (1 / 3) - 1.8


def _d0_array(n0: np.ndarray) -> np.ndarray:
    """`d0_from_n0` over an array, as a column vector.

    np.cbrt rather than ``** (1/3)`` so the discarded branch (n0 < 15) stays
    real instead of producing a nan that np.where would still have to carry.
    """
    return np.where(
        n0 < _D0_CONTINUITY_POINT, 1.0, 1.24 * np.cbrt(n0 - 15.0) - 1.8
    ).reshape(-1, 1)


def directional_ipsae(
    pae: np.ndarray,
    aligned: slice | Sequence[int],
    measured: slice | Sequence[int],
    pae_cutoff: float,
) -> float:
    """ipSAE aligned on one chain's residues, measured against another's.

    `aligned` and `measured` are slices when the caller knows the chains are
    contiguous blocks -- which `ipsae_from_pae` guarantees, since it lays them
    out that way -- and slicing takes a view where fancy indexing would copy.

    Returns 0.0 when no residue pair passes `pae_cutoff`, i.e. when the model
    predicts no interaction between the two chains.
    """
    if isinstance(aligned, slice) and isinstance(measured, slice):
        sub = pae[aligned, measured]
    else:
        sub = pae[np.ix_(np.asarray(aligned), np.asarray(measured))]

    passing = sub < pae_cutoff
    n0 = passing.sum(axis=1)
    if not n0.any():
        return 0.0

    # Plain np.where rather than a masked array: np.ma copies the block and
    # dispatches every element through the masked machinery for the same result.
    scores = np.where(passing, 1.0 / (1.0 + (sub / _d0_array(n0)) ** 2), 0.0)
    per_residue = np.where(n0 > 0, scores.sum(axis=1) / np.maximum(n0, 1), -np.inf)
    return float(per_residue.max())


def ipsae_from_pae(
    pae: np.ndarray,
    chain_ids: Sequence[str],
    chain_lengths: Sequence[int],
    *,
    pae_cutoff: float = 10.0,
    binder_chain: str | None = None,
) -> dict[str, float]:
    """ipSAE for every chain pair, plus the binder-centric summaries.

    Args:
        pae: square (L, L) PAE matrix, residues in the same order as
            `chain_ids` / `chain_lengths`.
        chain_ids: chain identifiers, in PAE order. Passed explicitly rather
            than derived positionally, so this function does not encode the
            pipeline's "binder is A, targets are B onward" convention.
        chain_lengths: residue count per chain, in the same order.
        pae_cutoff: PAE below which a residue pair counts as confident. The
            pipeline sets this from the campaign YAML; existing implementations
            default to 15, which is why it is never left implicit here.
        binder_chain: when given, the returned dict adds `ipsae_min_<target>`
            for each other chain, plus `ipsae_min` and `max_ipsae_min`.

    Returns:
        `ipsae_<aligned>_<measured>` for every ordered pair, plus the
        binder-centric keys when `binder_chain` is set.
    """
    pae = np.asarray(pae)

    if pae.ndim != 2:
        raise ValueError(f"PAE must be 2-D, got shape {pae.shape}")
    if pae.shape[0] != pae.shape[1]:
        raise ValueError(f"PAE must be square, got shape {pae.shape}")
    if len(chain_ids) != len(chain_lengths):
        raise ValueError(
            f"{len(chain_ids)} chain ids but {len(chain_lengths)} chain lengths"
        )
    if len(set(chain_ids)) != len(chain_ids):
        raise ValueError(f"chain ids must be unique, got {list(chain_ids)}")

    # The load-bearing check. A model that returns per-token PAE with tokens the
    # caller did not account for would otherwise mis-slice every chain and
    # produce entirely plausible wrong numbers, with no error anywhere.
    total = int(sum(chain_lengths))
    if pae.shape[0] != total:
        raise ValueError(
            f"PAE is {pae.shape[0]}x{pae.shape[0]} but the chains total {total} "
            f"residues ({dict(zip(chain_ids, chain_lengths))}). Refusing to "
            "slice: the chain offsets would be wrong."
        )

    if binder_chain is not None and binder_chain not in chain_ids:
        raise ValueError(
            f"binder_chain {binder_chain!r} not among chains {list(chain_ids)}"
        )

    blocks: dict[str, slice] = {}
    start = 0
    for chain_id, length in zip(chain_ids, chain_lengths):
        blocks[chain_id] = slice(start, start + length)
        start += length

    results: dict[str, float] = {}
    for i, first in enumerate(chain_ids):
        for second in list(chain_ids)[i + 1 :]:
            results[f"ipsae_{first}_{second}"] = directional_ipsae(
                pae, blocks[first], blocks[second], pae_cutoff
            )
            results[f"ipsae_{second}_{first}"] = directional_ipsae(
                pae, blocks[second], blocks[first], pae_cutoff
            )

    if binder_chain is None:
        return results

    # Key by the target chain explicitly, never by pair position: the positional
    # form is correct only while the binder happens to sort first.
    per_target = {
        target: min(
            results[f"ipsae_{binder_chain}_{target}"],
            results[f"ipsae_{target}_{binder_chain}"],
        )
        for target in chain_ids
        if target != binder_chain
    }
    for target, value in per_target.items():
        results[f"ipsae_min_{target}"] = value

    if per_target:
        values = list(per_target.values())
        results["ipsae_min"] = min(values)
        if len(values) > 1:
            results["max_ipsae_min"] = max(values)

    return results
