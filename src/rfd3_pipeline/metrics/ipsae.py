"""ipSAE from a predicted-aligned-error matrix, computed in process.

The pipeline evaluates ipSAE for every holo prediction and every cross-validation
seed, inside workers that already hold the PAE array in memory. Shelling out to
the `ipsae` CLI would pay a process spawn and two temp-file writes per call, so
this module reimplements the score over the array directly.

Definition, per ordered chain pair (aligned -> measured):

  1. For each residue i of the aligned chain, find the residues j of the measured
     chain with PAE(i, j) < `pae_cutoff`; let n0(i) be how many there are.
  2. d0(i) = 1.24 * (n0(i) - 15)^(1/3) - 1.8 for n0 >= 27, and 1.0 below that.
     The expression crosses 1.0 at n0 ~ 26.5, so this is **not** continuous:
     n0 = 26 gives 1.0 and n0 = 27 gives 1.0389. The reference has the same
     step, by a different route -- `calc_d0_array` clamps L to >= 26 and then
     floors the result at 1.0 -- and `_d0_array` was verified equal to it for
     every n0 from 0 to 200.
  3. Score residue i as the mean of 1 / (1 + (PAE(i, j) / d0(i))^2) over the
     passing j only.
  4. The directional ipSAE is the maximum over i.

The score is asymmetric, so binder->target and target->binder differ; the
pipeline reports the minimum of the two as `ipsae_min_<target>`.

**Verified against DunbrackLab/IPSAE on 2026-10-13** (`ipsae.py` at `main`,
the `ipSAE_d0res` path). Every ordered chain pair of a real 531-token holo PAE
agreed to 2.2e-16, and `_d0_array` matches `calc_d0_array` exactly over
n0 = 0..200. Both use a strict `<` against the cutoff and average only the
passing pairs.

Two differences remain, both deliberate:

* **This reports the minimum of the two directions, the reference's headline
  `ipsae_d0res_max` the maximum.** Per-direction values are identical, and both
  are recorded as `ipsae_<aligned>_<measured>`, so either summary is
  recoverable -- but `ipsae_min` is the stricter number and is not comparable
  to a literature threshold derived from the reference's maximum.
* This is PAE-only; the reference CLI also takes a distance cutoff, and its
  `ipSAE_d0chn` / `ipSAE_d0dom` variants take d0 from chain or domain length
  rather than per-residue n0. Those are different scores, not disagreements.

Note that the reference's *scalar* `calc_d0` disagrees with its own
`calc_d0_array` at exactly L = 27 (1.0 against 1.0389). The scalar is used only
for `d0chn` and `d0dom`, never for the `d0res` path this module implements, so
matching the array version is matching the right one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "ipsae_from_pae",
    "ipsae_from_tokens",
    "cross_chain_pae_floor",
    "directional_ipsae",
    "d0_from_n0",
]

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
    pairs: str = "all",
) -> dict[str, float]:
    """ipSAE per chain pair, plus the binder-centric summaries.

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
        pairs: `"all"` for every ordered chain pair, or `"binder"` for only
            those involving `binder_chain`. The pipeline passes `"binder"`:
            the ordered pairs grow as (N+1)N in the number of targets, and the
            target-to-target ones never involve the designed chain. At two
            targets that is 10 columns against 7; at five it is 37 against 17,
            doubled again because holo and cross-validation both write them.
            Target-to-target scores are still worth having for a one-off sanity
            check -- on this campaign the native pair reads 0.88/0.86 where the
            designed interface reads 0.61 -- which is why `"all"` remains
            available rather than being removed.

    Returns:
        `ipsae_<aligned>_<measured>` per ordered pair, plus the binder-centric
        keys when `binder_chain` is set:

        * `ipsae_min_<target>` -- the lower of the two directions for that
          target, since the score is asymmetric and the weaker direction is the
          honest one. **This is what the pipeline gates and ranks on.**
        * `ipsae_max_<target>` -- the higher of the two, which is the reference
          implementation's headline `ipsae_d0res_max`. Recorded only so a score
          can be held against a published threshold, since those are derived
          from the maximum.
        * `ipsae_min` -- the lowest `ipsae_min_<target>`, i.e. the binder's
          worst interface.
        * `max_ipsae_min` -- the highest `ipsae_min_<target>`, reported only
          with more than one target and so something to choose between.

        **`ipsae_max_<target>` and `max_ipsae_min` are different things** and
        sit one letter apart. The first maximises over the two *directions* of
        one target; the second maximises over *targets* of each one's minimum.
        For a binder on targets B and C, `ipsae_max_B` can exceed
        `max_ipsae_min` while telling you nothing about C.
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
    if pairs not in ("all", "binder"):
        raise ValueError(f"pairs must be 'all' or 'binder', got {pairs!r}")
    if pairs == "binder" and binder_chain is None:
        raise ValueError("pairs='binder' needs binder_chain to say which it is")

    blocks: dict[str, slice] = {}
    start = 0
    for chain_id, length in zip(chain_ids, chain_lengths):
        blocks[chain_id] = slice(start, start + length)
        start += length

    results: dict[str, float] = {}
    for i, first in enumerate(chain_ids):
        for second in list(chain_ids)[i + 1 :]:
            if pairs == "binder" and binder_chain not in (first, second):
                continue
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
    directions = {
        target: (
            results[f"ipsae_{binder_chain}_{target}"],
            results[f"ipsae_{target}_{binder_chain}"],
        )
        for target in chain_ids
        if target != binder_chain
    }
    per_target = {target: min(pair) for target, pair in directions.items()}
    for target, value in per_target.items():
        results[f"ipsae_min_{target}"] = value
        # The reference's headline `ipsae_d0res_max`, recorded so a score can be
        # compared with a published threshold -- those are derived from the
        # maximum, and `ipsae_min_<target>` is systematically stricter. The
        # pipeline still gates and ranks on the minimum: the weaker direction is
        # the honest one for a design we are deciding whether to believe.
        results[f"ipsae_max_{target}"] = max(directions[target])

    if per_target:
        values = list(per_target.values())
        results["ipsae_min"] = min(values)
        if len(values) > 1:
            results["max_ipsae_min"] = max(values)

    return results


def ipsae_from_tokens(
    pae: np.ndarray,
    token_chains: Sequence[str],
    *,
    binder_chain: str | None = None,
    pae_cutoff: float = 10.0,
    keep: Sequence[str] | None = None,
    pairs: str = "all",
) -> dict[str, float]:
    """ipSAE from a PAE plus one chain label per token.

    The shared entry point for both folding stages, because both models report
    per-token chain membership rather than chain lengths: ESMFold2 writes
    `asym_id` beside the PAE in its `*_pae.npz`, and the AF3 fork writes
    `token_chain_ids` in its per-sample confidences JSON. Deriving the blocks
    here means neither caller has to count residues and risk disagreeing with
    the matrix it is slicing.

    `keep` restricts the calculation to those chains, which is how ligands are
    excluded. A metal ion is a single token, and a one-residue "chain" produces
    a defined but meaningless ipSAE that would then be folded into `ipsae_min`
    as though it were a target. Passing the campaign's protein chains is both
    correct and self-documenting.

    Raises:
        ValueError: if the labels do not match the matrix, or if any chain's
            tokens are not contiguous. Non-contiguous tokens would make the
            slices in `ipsae_from_pae` describe the wrong residues while
            returning entirely plausible numbers -- the same class of silent
            error its own total check guards against.
    """
    pae = np.asarray(pae)
    labels = [str(label) for label in token_chains]

    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError(f"PAE must be square and 2-D, got shape {pae.shape}")
    if len(labels) != pae.shape[0]:
        raise ValueError(
            f"{len(labels)} token labels but PAE is "
            f"{pae.shape[0]}x{pae.shape[0]}; refusing to guess the alignment"
        )

    # Contiguous runs, in the order they appear.
    runs: list[tuple[str, int, int]] = []
    for index, label in enumerate(labels):
        if runs and runs[-1][0] == label:
            runs[-1] = (label, runs[-1][1], index + 1)
        else:
            runs.append((label, index, index + 1))

    seen: set[str] = set()
    for label, _, _ in runs:
        if label in seen:
            raise ValueError(
                f"chain {label!r} appears in more than one block of tokens "
                f"({[r[0] for r in runs]}); every chain must be contiguous for "
                "its PAE slice to describe it"
            )
        seen.add(label)

    if keep is not None:
        wanted = list(keep)
        missing = [chain for chain in wanted if chain not in seen]
        if missing:
            raise ValueError(
                f"chain(s) {missing} are not among the tokens "
                f"({sorted(seen)}); the prediction does not hold them"
            )
        runs = [run for run in runs if run[0] in wanted]
        index = np.concatenate([np.arange(start, stop) for _, start, stop in runs])
        pae = pae[np.ix_(index, index)]

    return ipsae_from_pae(
        pae,
        [label for label, _, _ in runs],
        [stop - start for _, start, stop in runs],
        pae_cutoff=pae_cutoff,
        binder_chain=binder_chain,
        pairs=pairs,
    )


def cross_chain_pae_floor(
    pae: np.ndarray,
    token_chains: Sequence[str],
    binder_chain: str,
    targets: Sequence[str],
) -> dict[str, float]:
    """The lowest PAE between the binder and each target.

    Diagnostic, and it exists because ipSAE can be legitimately **0.0** and
    that is indistinguishable from a broken extraction. The score counts only
    residue pairs under `pae_cutoff`; if none qualify, every residue scores
    zero and so does the maximum over them. Recording the floor says which
    happened: a floor above the cutoff means the model placed no confident
    contact, while a floor well under it alongside a zero score would mean the
    blocks were mis-sliced.

    Returns `pae_floor_<target>` per target, in the PAE's own units.
    """
    pae = np.asarray(pae, dtype=np.float32)
    labels = np.asarray([str(label) for label in token_chains])
    if len(labels) != pae.shape[0]:
        raise ValueError(
            f"{len(labels)} token labels but PAE is {pae.shape[0]}x{pae.shape[0]}"
        )

    binder = labels == binder_chain
    if not binder.any():
        raise ValueError(f"binder chain {binder_chain!r} is not among the tokens")

    floors: dict[str, float] = {}
    for target in targets:
        mask = labels == target
        if not mask.any():
            continue
        block = pae[np.ix_(binder, mask)]
        other = pae[np.ix_(mask, binder)]
        floors[f"pae_floor_{target}"] = float(min(block.min(), other.min()))
    return floors
