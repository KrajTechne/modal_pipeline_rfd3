"""Structure and confidence metrics for the RFD3 pipeline.

Every function here is GPU-free and model-free: it takes structures and PAE
arrays and returns numbers. Two conventions hold throughout:

* Chain identity is always passed in from the resolved campaign config, never
  assumed, so nothing here encodes "binder is A, targets are B onward".
* Metrics return values, never verdicts. Thresholds and pass/fail live in one
  gate layer so the set of gates exists as data in a single place.
"""

from .geometry import expected_radius_of_gyration, rg_report
from .interface import InterfaceResult, binding_interface, epitope_coverage
from .ipsae import d0_from_n0, directional_ipsae, ipsae_from_pae
from .rmsd import (
    TargetSuperposition,
    rmsd_binder_apo_holo,
    rmsd_binder_on_target,
    rmsd_motif_on_target,
    superimpose_rmsd,
    target_superposition,
)

__all__ = [
    "InterfaceResult",
    "TargetSuperposition",
    "binding_interface",
    "d0_from_n0",
    "directional_ipsae",
    "epitope_coverage",
    "expected_radius_of_gyration",
    "ipsae_from_pae",
    "rg_report",
    "rmsd_binder_apo_holo",
    "rmsd_binder_on_target",
    "rmsd_motif_on_target",
    "superimpose_rmsd",
    "target_superposition",
]
