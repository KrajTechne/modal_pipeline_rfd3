"""Superposition-based RMSDs, all on alpha carbons.

Three metrics, differing only in what is aligned and what is measured:

* `rmsd_motif_on_target`  -- align on the target chains, measure the motif
  residues. Does the scaffolded motif sit where RFD3 placed it?
* `rmsd_binder_on_target` -- align on the target chains, measure the whole
  binder. Recorded, not gated.
* `rmsd_binder_apo_holo`  -- align binder-on-binder, measure the binder. Does
  the binder fold the same free as bound?

The last one is deliberately *not* target-aligned. The original plan specified
aligning it on the target chains, which cannot work: the apo prediction is
binder-only and has no target chains. Binder-on-binder superposition is both the
only computable form and the meaningful one.

The two target-aligned metrics are normally wanted together for the same pair of
structures, so `target_superposition` does the match and the fit once and both
read their own residues off it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..structure import chain_subset, match_ca

__all__ = [
    "TargetSuperposition",
    "superimpose_rmsd",
    "target_superposition",
    "rmsd_motif_on_target",
    "rmsd_binder_on_target",
    "rmsd_binder_apo_holo",
]


@dataclass(frozen=True)
class TargetSuperposition:
    """A designed and a predicted structure fitted on their target chains.

    Holds the fit so several measurements can be read off one superposition:
    matching and superimposing are the expensive parts, and the motif and the
    whole binder differ only in which residues they then measure.
    """

    designed: object
    fitted: object
    keys: list[tuple[str, int]]
    binder_chain: str

    def rmsd_over(self, residues: Sequence[int] | None) -> float:
        """RMSD over binder residues, or the whole binder when None."""
        import biotite.structure as struc

        chain_of = np.array([chain for chain, _ in self.keys])
        if residues is None:
            mask = chain_of == self.binder_chain
        else:
            wanted = {(self.binder_chain, int(r)) for r in residues}
            if missing := wanted - set(self.keys):
                raise ValueError(
                    f"{len(missing)} measured residue(s) absent from the shared "
                    f"set, e.g. {sorted(missing)[:5]}"
                )
            mask = np.array([key in wanted for key in self.keys])

        if not mask.any():
            raise ValueError("no residues selected to measure")
        return float(struc.rmsd(self.designed[mask], self.fitted[mask]))


def superimpose_rmsd(
    fixed,
    mobile,
    align_mask: np.ndarray | None = None,
    measure_mask: np.ndarray | None = None,
) -> float:
    """Superimpose `mobile` onto `fixed`, then RMSD over the measured subset.

    Both arrays must already be alpha carbons in matching order -- use
    `structure.match_ca` to guarantee that. `align_mask` selects the atoms the
    superposition is fitted on; `measure_mask` selects those the RMSD is
    computed over. Either defaults to all atoms.
    """
    import biotite.structure as struc

    if fixed.array_length() != mobile.array_length():
        raise ValueError(
            f"fixed has {fixed.array_length()} atoms, mobile has "
            f"{mobile.array_length()}; pair them with match_ca() first"
        )

    if align_mask is None:
        align_mask = np.ones(fixed.array_length(), dtype=bool)
    if measure_mask is None:
        measure_mask = np.ones(fixed.array_length(), dtype=bool)
    if not align_mask.any():
        raise ValueError("align_mask selects no atoms")
    if not measure_mask.any():
        raise ValueError("measure_mask selects no atoms")

    fitted, _ = struc.superimpose(fixed=fixed, mobile=mobile, atom_mask=align_mask)
    return float(struc.rmsd(fixed[measure_mask], fitted[measure_mask]))


def target_superposition(
    designed, predicted, binder_chain: str, target_chains: Sequence[str]
) -> TargetSuperposition:
    """Fit `predicted` onto `designed` using the target chains only.

    Aligning on the target rather than the whole complex is what makes binder
    displacement visible instead of being absorbed into the fit.
    """
    import biotite.structure as struc

    chains = [binder_chain, *target_chains]
    designed_ca, predicted_ca, keys = match_ca(designed, predicted, chains=chains)

    align_mask = np.isin(np.array([chain for chain, _ in keys]), list(target_chains))
    if not align_mask.any():
        raise ValueError(
            f"target chains {list(target_chains)} are absent from the shared "
            "residue set; cannot align on them"
        )

    fitted, _ = struc.superimpose(
        fixed=designed_ca, mobile=predicted_ca, atom_mask=align_mask
    )
    return TargetSuperposition(
        designed=designed_ca, fitted=fitted, keys=keys, binder_chain=binder_chain
    )


def rmsd_motif_on_target(
    designed,
    predicted,
    binder_chain: str,
    target_chains: Sequence[str],
    motif_residues: Sequence[int],
) -> float:
    """Target-aligned RMSD over the scaffolded motif residues.

    `motif_residues` are binder-chain residue ids in **output** numbering, i.e.
    the values of `diffused_index_map`, as produced by `rfd3_spec`.
    """
    if len(motif_residues) == 0:
        raise ValueError("no motif residues given")
    fit = target_superposition(designed, predicted, binder_chain, target_chains)
    return fit.rmsd_over(motif_residues)


def rmsd_binder_on_target(
    designed, predicted, binder_chain: str, target_chains: Sequence[str]
) -> float:
    """Target-aligned RMSD over the whole binder chain."""
    fit = target_superposition(designed, predicted, binder_chain, target_chains)
    return fit.rmsd_over(None)


def rmsd_binder_apo_holo(holo, apo, binder_chain: str) -> float:
    """Binder-on-binder RMSD between the apo and holo predictions.

    The binder chain is pulled out of the holo complex and the apo structure is
    superimposed onto it. Residues are paired on (chain, res_id), so a model
    that drops or renumbers residues raises rather than silently mis-pairing.
    """
    holo_binder = chain_subset(holo, binder_chain)
    if holo_binder.array_length() == 0:
        raise ValueError(f"binder chain {binder_chain!r} absent from the holo structure")

    holo_ca, apo_ca, _ = match_ca(holo_binder, apo, chains=[binder_chain])
    return superimpose_rmsd(holo_ca, apo_ca)
