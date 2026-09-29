"""Radius of gyration for the backbone compactness filter.

The number RFD3 reports in its metadata JSON is for the whole complex, so the
binder's own Rg has to be recomputed on the binder chain alone -- which is the
entire reason this module exists.

The filter compares against the empirical folded-protein scaling
`Rg_expected = 2.38 * N^0.365`, and the campaign's `multiplier` decides what
ratio passes. That comparison is the gate layer's job: this module returns
`rg`, `rg_expected` and their ratio, like every other metric returns numbers.
"""

from __future__ import annotations

from ..structure import ca_only, chain_subset

__all__ = ["expected_radius_of_gyration", "rg_report"]

# Empirical scaling for compact folded proteins. The plan fixes both; only the
# multiplier applied to the ratio is configurable, and that lives in the YAML.
PREFACTOR = 2.38
EXPONENT = 0.365


def expected_radius_of_gyration(n_residues: int) -> float:
    """Empirical Rg for a compact folded protein of `n_residues`."""
    if n_residues <= 0:
        raise ValueError(f"n_residues must be positive, got {n_residues}")
    return PREFACTOR * n_residues**EXPONENT


def rg_report(atoms, binder_chain: str) -> dict[str, float | int]:
    """Binder radius of gyration, its expectation, and their ratio.

    Computed on the binder chain's alpha carbons only. Unweighted Rg and
    mass-weighted Rg coincide there, since every atom is a carbon.
    """
    import biotite.structure as struc

    binder_ca = ca_only(chain_subset(atoms, binder_chain))
    n_residues = binder_ca.array_length()
    if n_residues == 0:
        raise ValueError(f"no alpha carbons found for binder chain {binder_chain!r}")

    rg = float(struc.gyration_radius(binder_ca))
    expected = expected_radius_of_gyration(n_residues)

    return {
        "binder_length": n_residues,
        "rg": rg,
        "rg_expected": expected,
        "rg_ratio": rg / expected,
    }
