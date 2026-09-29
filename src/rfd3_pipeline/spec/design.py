"""Reading an RFD3 design back: motif map, sampled contig, cross-checks.

RFD3 writes one metadata JSON beside each structure (`dump_prediction_metadata_json`
defaults to True, `input.md:105`). Two things in it matter here:

* `diffused_index_map` -- input residue to output residue, for every motif
  residue. This is **ground truth** for the binder's scaffolded motif, because
  unindexed motif residues are placed wherever the model chose and no arithmetic
  can predict where.
* `extra.sampled_contig` -- the length actually drawn for each variable-length
  designed region (`input.md:130,152`), which makes otherwise-indeterminate
  output positions exact.

The map also carries target-chain entries. The original plan said to ignore
them; this module uses them instead as an **exact cross-check** on the contig
arithmetic in `rfd3_pipeline.contig`. Where RFD3 and our arithmetic both describe
a residue and disagree, that is a bug in the arithmetic, and it is far cheaper to
find here than in a ranked list of designs.

Note on provenance: written without a real RFD3 output file to hand. Both key
paths are the ones upstream documents, and a mismatch raises naming the keys that
were actually present -- check against the first real design.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..contig import (
    Component,
    ResidueKey,
    as_components,
    format_residue,
    map_input_to_output,
    parse_residue_key,
    renumber_hotspots,
)
from ..structure import chains_present

__all__ = [
    "DesignMetadata",
    "MotifMap",
    "assert_output_chains",
    "cross_check_target_mapping",
    "load_design_metadata",
    "renumber_hotspots_for_design",
]

_INDEX_MAP_KEY = "diffused_index_map"
_EXTRA_KEY = "extra"
_SAMPLED_CONTIG_KEY = "sampled_contig"


@dataclass(frozen=True)
class MotifMap:
    """Input-to-output positions for the binder's scaffolded motif residues."""

    binder_chain: str
    entries: dict[ResidueKey, ResidueKey]

    @property
    def output_residues(self) -> tuple[int, ...]:
        """Motif residue ids in output numbering, ascending."""
        return tuple(sorted(resi for _, resi in self.entries.values()))

    def compact(self) -> str:
        """`A77>A34;A78>A35` -- a one-cell record of the numbering translation."""
        pairs = sorted(self.entries.items(), key=lambda item: item[1][1])
        return ";".join(
            f"{format_residue(source)}>{format_residue(destination)}"
            for source, destination in pairs
        )


@dataclass(frozen=True)
class DesignMetadata:
    """One design's metadata, parsed once.

    Consumers take this rather than a raw dict, so the JSON layout is known in
    exactly one place and residue keys are parsed exactly once.
    """

    binder_entries: dict[ResidueKey, ResidueKey]
    target_entries: dict[ResidueKey, ResidueKey]
    sampled_contig: str | None

    def motif_map(self, binder_chain: str) -> MotifMap:
        if not self.binder_entries:
            raise ValueError(
                f"diffused_index_map has no entries landing on binder chain "
                f"{binder_chain!r}; it maps onto "
                f"{sorted({chain for chain, _ in self.target_entries.values()})}"
            )
        return MotifMap(binder_chain=binder_chain, entries=self.binder_entries)


def _read_index_map(payload: dict) -> dict[str, str]:
    if _INDEX_MAP_KEY not in payload:
        raise KeyError(
            f"no {_INDEX_MAP_KEY!r} in the design metadata; top-level keys are "
            f"{sorted(payload)}. Confirm the layout against a real RFD3 output."
        )
    found = payload[_INDEX_MAP_KEY]
    if not isinstance(found, dict):
        raise ValueError(
            f"{_INDEX_MAP_KEY!r} should be an object, got {type(found).__name__}"
        )
    return found


def parse_design_metadata(payload: dict, binder_chain: str) -> DesignMetadata:
    """Split a metadata dict into binder and target entries, parsed once."""
    binder_entries: dict[ResidueKey, ResidueKey] = {}
    target_entries: dict[ResidueKey, ResidueKey] = {}

    for raw_source, raw_destination in _read_index_map(payload).items():
        source = parse_residue_key(raw_source)
        destination = parse_residue_key(raw_destination)
        if destination[0] == binder_chain:
            binder_entries[source] = destination
        else:
            target_entries[source] = destination

    extra = payload.get(_EXTRA_KEY)
    sampled = extra.get(_SAMPLED_CONTIG_KEY) if isinstance(extra, dict) else None

    return DesignMetadata(
        binder_entries=binder_entries,
        target_entries=target_entries,
        sampled_contig=sampled if isinstance(sampled, str) else None,
    )


def load_design_metadata(path: str | Path, binder_chain: str) -> DesignMetadata:
    """Read and parse a design's metadata JSON."""
    with open(path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(payload).__name__}")
    return parse_design_metadata(payload, binder_chain)


def cross_check_target_mapping(
    derived: dict[ResidueKey, ResidueKey],
    reference: dict[ResidueKey, ResidueKey],
) -> int:
    """Assert contig arithmetic agrees with RFD3's own map where both apply.

    Returns the number of residues compared. Raises on disagreement, reporting
    up to five, because a systematic off-by-one shows up in every shared residue
    at once and the count is more useful than the list.
    """
    shared = set(derived) & set(reference)
    if not shared:
        return 0

    mismatches = [
        (key, derived[key], reference[key])
        for key in sorted(shared)
        if derived[key] != reference[key]
    ]
    if mismatches:
        shown = "; ".join(
            f"{format_residue(key)}: contig says {format_residue(ours)}, "
            f"RFD3 says {format_residue(theirs)}"
            for key, ours, theirs in mismatches[:5]
        )
        raise ValueError(
            f"contig arithmetic disagrees with {_INDEX_MAP_KEY} for "
            f"{len(mismatches)} of {len(shared)} shared residues: {shown}"
        )
    return len(shared)


def renumber_hotspots_for_design(
    hotspots: str | Sequence[ResidueKey],
    contig: str | Sequence[Component],
    metadata: DesignMetadata,
    structure=None,
) -> dict[ResidueKey, ResidueKey]:
    """Renumber hotspots against one realised design, then verify the result.

    Composes three things that are separate concerns: the sampled contig
    replaces the declared one when the design recorded it, the pure arithmetic
    lives in `rfd3_pipeline.contig`, and the result is checked against RFD3's
    own target-chain entries.
    """
    components = as_components(metadata.sampled_contig or contig)
    mapping = renumber_hotspots(hotspots, components, structure=structure)
    cross_check_target_mapping(
        map_input_to_output(components, structure=structure), metadata.target_entries
    )
    return mapping


def assert_output_chains(structure, expected: Sequence[str]) -> None:
    """Check the design's chains are exactly the configured binder and targets.

    The convention -- binder first, then targets in contig order, lettered from A
    -- is derived from how RFD3 renumbers, not guaranteed by it. Asserting it on
    the first design of each spec turns a contig whose ordering differs from the
    declared chains into an immediate failure, rather than a campaign of ipSAE
    columns attributed to the wrong chain.
    """
    present = chains_present(structure)
    if present != sorted(expected):
        raise ValueError(
            f"design has chains {present} but the campaign declares "
            f"{sorted(expected)}. Check the contig's chain order against "
            "chain_binder / chain_targets."
        )
