"""Shared fixtures and builders.

Synthetic structures are hand-built AtomArrays rather than fixture files so the
geometry under test is explicit; the alphaV/beta3 input is the one real fixture.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUTS = REPO_ROOT / "inputs_motif_scaffold_alphav_beta3"
ALPHAV_PDB = INPUTS / "alphav_beta3_fib10_clean.pdb"
CAMPAIGN_YAML = INPUTS / "alphav_rgd_unindex_claude.yaml"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# The campaign under development. Kept here so the derived length (519-539)
# asserted in several files cannot drift apart from the contig it comes from.
ALPHAV_CONTIG = "100-120,/0,B1-177,/0,C1-242"
ALPHAV_HOTSPOTS = (
    "B59,B86,B87,B89,B121,B124,B127,"
    "C10,C11,C12,C15,C102,C103,C104,C105,C106,C107,C109,C225"
)


def make_atoms(chain_id: str, res_ids, coords, atom_name: str = "CA"):
    """A single-chain AtomArray of one atom per residue."""
    import biotite.structure as struc

    res_ids = list(res_ids)
    coords = np.asarray(coords, dtype=float)
    assert len(res_ids) == len(coords)

    atoms = struc.AtomArray(len(res_ids))
    atoms.coord = coords
    atoms.chain_id = np.array([chain_id] * len(res_ids))
    atoms.res_id = np.array(res_ids)
    atoms.res_name = np.array(["GLY"] * len(res_ids))
    atoms.atom_name = np.array([atom_name] * len(res_ids))
    atoms.element = np.array(["C"] * len(res_ids))
    return atoms


def rotation_matrix(axis, angle):
    """Rodrigues rotation, for checking that superposition removes rigid motion."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def build(raw: dict, **sections) -> dict:
    """The campaign YAML with sections shallow-merged over a deep copy."""
    merged = copy.deepcopy(raw)
    for section, changes in sections.items():
        merged[section] = {**merged.get(section, {}), **changes}
    return merged


def with_contig(raw: dict, contig: str) -> dict:
    """The campaign with every spec's contig replaced."""
    specs = copy.deepcopy(raw["rfdiffusion3"]["specs"])
    for spec in specs:
        spec["contig"] = contig
    return build(raw, rfdiffusion3={"specs": specs})


def with_specs(raw: dict, mutate) -> dict:
    """The campaign with `mutate` applied to a deep copy of its spec list."""
    specs = copy.deepcopy(raw["rfdiffusion3"]["specs"])
    mutate(specs)
    return build(raw, rfdiffusion3={"specs": specs})


@pytest.fixture
def raw():
    """The campaign YAML, freshly parsed so tests may mutate it."""
    return yaml.safe_load(CAMPAIGN_YAML.read_text())


@pytest.fixture
def helix_coords():
    """A short, non-degenerate 3-D path; anything but collinear points."""
    t = np.linspace(0.0, 4.0 * np.pi, 12)
    return np.stack([np.cos(t) * 5.0, np.sin(t) * 5.0, t * 1.5], axis=1)


@pytest.fixture
def helix_holo(helix_coords):
    """`(holo, binder)` -- a two-chain complex and its isolated binder.

    The target is offset well clear of the binder so it never contributes
    contacts or superposition weight.
    """
    binder = make_atoms("A", range(1, 13), helix_coords)
    target = make_atoms("B", range(1, 13), helix_coords + 30.0)
    return binder + target, binder


@pytest.fixture(scope="session")
def alphav_structure():
    """The checked-in alphaV/beta3 input: FN10 (A), the integrin pair (B, C),
    and four Mn(2+) ions (D, E)."""
    from rfd3_pipeline.structure import load_structure

    if not ALPHAV_PDB.exists():
        pytest.skip(f"input structure not present: {ALPHAV_PDB}")
    return load_structure(ALPHAV_PDB)


@pytest.fixture
def alphav_hotspots():
    """The campaign's hotspot list, parsed."""
    from rfd3_pipeline.contig import parse_residue_selection

    return parse_residue_selection(ALPHAV_HOTSPOTS)


def rgd_carboxylate(structure):
    """The RGD aspartate's carboxylate oxygens -- the motif's binding anchor."""
    return structure[
        (structure.chain_id == "A")
        & (structure.res_id == 79)
        & np.isin(structure.atom_name, ["OD1", "OD2"])
    ]


def min_distance(first, second) -> float:
    import biotite.structure as struc

    return float(
        min(struc.distance(first, atom).min() for atom in second)
        if second.array_length()
        else np.inf
    )
