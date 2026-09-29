"""Binder-target interface contacts, epitope coverage, and hotspot fraction.

One implementation serves two callers:

* the backbone filter, run on the RFD3 design, where `recall` is the hotspot
  contact fraction the campaign gates on;
* the holo metrics, run on the predicted complex, where the full column set is
  recorded.

Generalized from Optimus's `determine_binding_interface`: chains are passed in
rather than assumed to be "A" and `chr(ord('B') + i)`, hotspots are plain
integers parsed upstream, and the per-residue contact threshold is a parameter.

Like every metric here, this returns numbers and index sets. It does not know
the campaign's thresholds and emits no pass/fail verdict -- gating is one layer's
job, applied uniformly, so the list of gates exists as data in one place.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..structure import chain_subset, heavy_atoms

__all__ = ["InterfaceResult", "binding_interface", "epitope_coverage"]


@dataclass(frozen=True)
class InterfaceResult:
    """Contacts between one binder chain and one target chain."""

    binder_chain: str
    target_chain: str
    paratope: tuple[int, ...]
    epitope: tuple[int, ...]
    recall: float
    precision: float
    f1: float

    def to_columns(self) -> dict[str, object]:
        """Metric columns, suffixed by target chain.

        Unprefixed: the stage a metric was computed at is the reporting layer's
        concern, not the metric's.
        """
        tail = self.target_chain
        return {
            f"paratope_indices_{tail}": ",".join(map(str, self.paratope)),
            f"paratope_length_{tail}": len(self.paratope),
            f"epitope_indices_{tail}": ",".join(map(str, self.epitope)),
            f"epitope_length_{tail}": len(self.epitope),
            f"epitope_recall_{tail}": self.recall,
            f"epitope_precision_{tail}": self.precision,
            f"epitope_f1_{tail}": self.f1,
        }


def epitope_coverage(
    actual: Sequence[int], desired: Sequence[int]
) -> tuple[float, float, float]:
    """Recall, precision and F1 of an actual epitope against the hotspot list.

    * recall -- fraction of the requested hotspots that were contacted. This is
      the pipeline's hotspot contact fraction.
    * precision -- fraction of the contacted epitope that was requested, i.e.
      the inverse of off-target spillover.
    * f1 -- Jaccard-style balance of the two.

    All three are 0.0 when either set is empty, since no meaningful ratio
    exists; callers that allow an empty hotspot list must not gate on recall.
    """
    actual_set, desired_set = set(actual), set(desired)
    if not actual_set or not desired_set:
        return 0.0, 0.0, 0.0

    hit = actual_set & desired_set
    return (
        len(hit) / len(desired_set),
        len(hit) / len(actual_set),
        len(hit) / len(actual_set | desired_set),
    )


def _contacted_residues(atoms, contacted: np.ndarray, min_contacts: int) -> tuple[int, ...]:
    """Residue ids having at least `min_contacts` contacting heavy atoms.

    Grouped with biotite's residue-wise reduction rather than by `res_id` alone,
    so insertion codes and chain boundaries delimit residues correctly.
    """
    import biotite.structure as struc

    counts = struc.apply_residue_wise(atoms, contacted.astype(np.int32), np.sum)
    residue_ids = struc.get_residues(atoms)[0]
    return tuple(sorted(int(r) for r in residue_ids[counts >= min_contacts]))


def binding_interface(
    atoms,
    binder_chain: str,
    target_chain: str,
    hotspots: Sequence[int] = (),
    *,
    cutoff: float = 4.5,
    min_contacts: int = 1,
) -> InterfaceResult:
    """Contacts between `binder_chain` and `target_chain` in one structure.

    Args:
        atoms: a biotite `AtomArray` holding both chains.
        binder_chain, target_chain: chain ids, from the resolved campaign config.
        hotspots: requested epitope residue ids **on `target_chain`**, already
            renumbered into this structure's numbering by `rfd3_spec`.
        cutoff: heavy-atom contact distance in angstrom.
        min_contacts: heavy atoms of a residue that must be within `cutoff`
            for that residue to count as contacted. The pipeline uses 1; raising
            it to 2 tightens the hotspot definition without touching callers.

    Returns:
        An `InterfaceResult`. Hydrogens are excluded throughout.
    """
    import biotite.structure as struc

    binder = heavy_atoms(chain_subset(atoms, binder_chain))
    target = heavy_atoms(chain_subset(atoms, target_chain))

    if binder.array_length() == 0:
        raise ValueError(f"no heavy atoms found for binder chain {binder_chain!r}")
    if target.array_length() == 0:
        raise ValueError(f"no heavy atoms found for target chain {target_chain!r}")

    # Neighbour indices, not a dense (n_binder, n_target) mask: only the two
    # per-atom reductions below are ever used, and the mask costs ~90x the
    # memory -- 92 MB at 3000 residues against ~1 MB here.
    neighbours = struc.CellList(target, cell_size=cutoff).get_atoms(
        binder.coord, radius=cutoff
    )
    valid = neighbours >= 0  # -1 pads the ragged rows

    target_contacted = np.zeros(target.array_length(), dtype=bool)
    target_contacted[neighbours[valid]] = True

    epitope = _contacted_residues(target, target_contacted, min_contacts)
    paratope = _contacted_residues(binder, valid.any(axis=1), min_contacts)
    recall, precision, f1 = epitope_coverage(epitope, hotspots)

    return InterfaceResult(
        binder_chain=binder_chain,
        target_chain=target_chain,
        paratope=paratope,
        epitope=epitope,
        recall=recall,
        precision=precision,
        f1=f1,
    )
