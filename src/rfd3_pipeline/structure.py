"""Structure loading, chain selection, and CA matching.

RFD3 writes gzipped mmCIF; the folding kits write CIF or PDB. `load_structure`
takes any of them so callers never branch on extension.

`match_ca` exists because every RMSD in this pipeline compares two structures
that *should* describe the same residues but come from different programs.
Pairing their atoms by file order -- which is what a bare length assertion
amounts to -- silently mis-pairs residues whenever one writer orders chains
differently. Matching on (chain, residue id) makes that a loud failure instead.
"""

from __future__ import annotations

import gzip
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

__all__ = [
    "load_structure",
    "chain_subset",
    "chains_present",
    "ca_only",
    "heavy_atoms",
    "chain_sequence",
    "chain_lengths",
    "residues_by_chain",
    "match_ca",
]

# Substituted for any residue with no single-letter code. Fixed here rather than
# per call site because `chain_sequence` feeds the content-addressed MSA cache
# key: two implementations disagreeing on this character would silently produce
# two cache entries for one chain.
UNKNOWN_RESIDUE = "X"


def load_structure(path: str | Path, model: int = 1):
    """Read a structure file, optionally gzipped, as an AtomArray.

    Format dispatch is biotite's, so every format it supports works here and no
    table of extensions has to be kept in step with what RFD3 and the folding
    kits emit. Gzip is the one thing biotite's loader does not handle: its entry
    point takes a path rather than a handle, so a compressed file is expanded to
    a temporary copy that keeps the inner extension and is removed afterwards.
    """
    import biotite.structure.io as structure_io

    path = Path(path)
    if path.suffix.lower() != ".gz":
        return structure_io.load_structure(path, model=model)

    inner_suffix = Path(path.stem).suffix
    if not inner_suffix:
        raise ValueError(
            f"cannot infer a structure format from {path.name!r}; a gzipped "
            "structure needs its format extension, e.g. design.cif.gz"
        )

    with tempfile.NamedTemporaryFile(suffix=inner_suffix, delete=False) as temporary:
        expanded = Path(temporary.name)
        with gzip.open(path, "rb") as compressed:
            shutil.copyfileobj(compressed, temporary)
    try:
        return structure_io.load_structure(expanded, model=model)
    finally:
        expanded.unlink(missing_ok=True)


def chain_subset(atoms, chains: str | Sequence[str]):
    """Atoms belonging to one chain or any of several."""
    if isinstance(chains, str):
        chains = [chains]
    return atoms[np.isin(atoms.chain_id, list(chains))]


def chains_present(atoms) -> list[str]:
    """Chain ids present, sorted."""
    import biotite.structure as struc

    return sorted(str(chain) for chain in struc.get_chains(atoms))


def ca_only(atoms):
    """Alpha carbons, in the order they appear."""
    return atoms[atoms.atom_name == "CA"]


def heavy_atoms(atoms):
    """Everything but hydrogens."""
    return atoms[atoms.element != "H"]


def chain_sequence(atoms, chain: str) -> str:
    """One-letter sequence of a chain, from its alpha carbons.

    The single implementation in the pipeline: the MSA cache hashes this string
    to key its entries, `ipsae_from_pae` derives chain lengths from it, and
    campaign resolution compares it against `seq_targets`. Residues with no
    one-letter code become `UNKNOWN_RESIDUE`.
    """
    import biotite.structure as struc

    residues = ca_only(chain_subset(atoms, chain)).res_name
    return "".join(
        struc.info.one_letter_code(str(name)) or UNKNOWN_RESIDUE for name in residues
    )


def chain_lengths(atoms, chains: Sequence[str]) -> list[int]:
    """Residue count per chain, in the order given.

    Matches `ipsae_from_pae`'s `chain_lengths` argument, which must agree with
    the PAE matrix's own dimensions.
    """
    alpha_carbons = ca_only(atoms)
    return [int((alpha_carbons.chain_id == chain).sum()) for chain in chains]


def residues_by_chain(atoms) -> dict[str, set[int]]:
    """Residue ids present, grouped by chain, in one pass.

    Callers that ask about several chains -- contig mapping and length
    derivation both do -- build this once rather than rescanning the array per
    chain.
    """
    chain_ids = atoms.chain_id
    return {
        str(chain): {int(resi) for resi in np.unique(atoms.res_id[chain_ids == chain])}
        for chain in np.unique(chain_ids)
    }


def match_ca(first, second, chains: Sequence[str] | None = None):
    """Align two structures' alpha carbons onto a shared, ordered residue set.

    Returns `(first_ca, second_ca, keys)` where the two arrays hold exactly the
    residues present in both, in the same (chain, res_id) order, and `keys` is
    that ordering as a list of (chain, res_id) tuples.

    Raises when the shared set is empty, which in this pipeline means the two
    structures disagree about chain naming -- a real error rather than a
    degenerate RMSD of 0.
    """
    first_ca, second_ca = ca_only(first), ca_only(second)
    if chains is not None:
        first_ca = chain_subset(first_ca, chains)
        second_ca = chain_subset(second_ca, chains)

    # .tolist() on both annotations: leaving chain_id as numpy scalars makes
    # every key tuple hash through numpy rather than str, for no benefit.
    first_keys = list(zip(first_ca.chain_id.tolist(), first_ca.res_id.tolist()))
    second_keys = list(zip(second_ca.chain_id.tolist(), second_ca.res_id.tolist()))

    shared = sorted(set(first_keys) & set(second_keys))
    if not shared:
        raise ValueError(
            "no shared (chain, residue) alpha carbons between the two "
            f"structures; first has chains {chains_present(first_ca)}, "
            f"second has {chains_present(second_ca)}"
        )

    first_index = {key: i for i, key in enumerate(first_keys)}
    second_index = {key: i for i, key in enumerate(second_keys)}

    return (
        first_ca[[first_index[key] for key in shared]],
        second_ca[[second_index[key] for key in shared]],
        shared,
    )
