"""Contig strings: parsing, output numbering, and design length.

A contig is RFD3's description of what to build. Its grammar, from
`models/rfd3/docs/input.md:189-207`:

* components are comma separated;
* a leading chain label means "take these residues from the input structure"
  (`B1-177`, or `A203` for a single residue);
* no chain label means "design this many residues" -- a range like `100-120` is
  sampled uniformly per design;
* `/0` is a chain break.

The output structure renumbers everything: chains are lettered in contig order
starting at A, and residues within each chain are numbered from 1. So for
`100-120,/0,B1-177,/0,C1-242` the designed binder becomes chain A, input chain B
becomes output chain B, and input chain C becomes output chain C -- an identity
mapping for the targets, but only because those contigs happen to start at 1 and
appear in alphabetical order. Neither is guaranteed: `100-120,/0,C1-242,/0,B1-177`
would make input chain C into output chain B.

That non-identity case is exactly what makes this module worth testing hard. A
mapping error here does not raise; it silently shifts every hotspot and motif
index, and every downstream contact count and RMSD with them.

This module is a leaf: it imports only `structure`, so both the campaign schema
and the design readers can depend on it without a cycle. One grammar, parsed in
one place.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from string import ascii_uppercase

from .structure import residues_by_chain

__all__ = [
    "ChainBreak",
    "Component",
    "Designed",
    "FromInput",
    "IndeterminateMapping",
    "as_components",
    "derive_length",
    "expected_output_chains",
    "format_residue",
    "format_residues",
    "map_input_to_output",
    "output_chain_ids",
    "parse_contig",
    "parse_residue_key",
    "parse_residue_selection",
    "renumber_hotspots",
    "segments",
]

# One grammar for "chain label followed by a residue number", shared by the
# contig parser and by the reader for RFD3's diffused_index_map. Two regexes in
# two modules would let one side accept a chain id the other rejects, and the
# cross-check between them is precisely where that would surface as a confusing
# mismatch.
_CHAIN_RESIDUE = r"([A-Za-z][A-Za-z0-9]*?)(\d+)"

_FROM_INPUT = re.compile(rf"^{_CHAIN_RESIDUE}(?:-(\d+))?$")
_RESIDUE_KEY = re.compile(rf"^{_CHAIN_RESIDUE}$")
_DESIGNED = re.compile(r"^(\d+)(?:-(\d+))?$")
_CHAIN_BREAK = re.compile(r"^/(\d+)$")

ResidueKey = tuple[str, int]


class IndeterminateMapping(ValueError):
    """Raised when output positions depend on a length RFD3 has not yet sampled.

    A variable-length designed region shifts everything after it within the same
    output chain, so input residues following one cannot be placed until the
    design exists. Use the sampled contig from the design's metadata JSON, which
    records the length actually drawn.
    """


@dataclass(frozen=True)
class FromInput:
    """Residues copied from the input structure."""

    chain: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"{self.chain}{self.start}-{self.end} runs backwards")

    @property
    def residues(self) -> list[ResidueKey]:
        return [(self.chain, i) for i in range(self.start, self.end + 1)]

    @property
    def nominal_length(self) -> int:
        """Residue count implied by the range, ignoring gaps in the structure."""
        return self.end - self.start + 1

    def __str__(self) -> str:
        return f"{self.chain}{self.start}-{self.end}"


@dataclass(frozen=True)
class Designed:
    """A de novo region; `minimum == maximum` means a fixed length."""

    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if self.minimum <= 0:
            raise ValueError(f"designed length must be positive, got {self.minimum}")
        if self.maximum < self.minimum:
            raise ValueError(f"designed range {self.minimum}-{self.maximum} runs backwards")

    @property
    def is_fixed(self) -> bool:
        return self.minimum == self.maximum


@dataclass(frozen=True)
class ChainBreak:
    """A chain break; starts a new output chain."""


Component = FromInput | Designed | ChainBreak


def parse_residue_key(key: str) -> ResidueKey:
    """`"A125"` -> `("A", 125)`, on the shared chain-residue grammar."""
    match = _RESIDUE_KEY.match(key.strip())
    if not match:
        raise ValueError(f"cannot parse residue key {key!r}; expected e.g. 'A125'")
    return match.group(1), int(match.group(2))


def format_residue(key: ResidueKey) -> str:
    """`("A", 125)` -> `"A125"`. The inverse of `parse_residue_key`."""
    return f"{key[0]}{key[1]}"


def format_residues(keys: Sequence[ResidueKey], limit: int = 5) -> str:
    """A short, comma-joined sample of residue keys, for error messages."""
    return ", ".join(format_residue(key) for key in list(keys)[:limit])


def parse_contig(text: str) -> tuple[Component, ...]:
    """Parse a contig string into components, left to right."""
    if not text or not text.strip():
        raise ValueError("empty contig")

    parsed: list[Component] = []
    for raw in text.split(","):
        token = raw.strip()
        if not token:
            continue

        if match := _CHAIN_BREAK.match(token):
            if match.group(1) != "0":
                raise ValueError(
                    f"unsupported chain break {token!r}; RFD3 documents only '/0'"
                )
            parsed.append(ChainBreak())
        elif match := _FROM_INPUT.match(token):
            chain, start, end = match.group(1), int(match.group(2)), match.group(3)
            parsed.append(FromInput(chain, start, int(end) if end else start))
        elif match := _DESIGNED.match(token):
            low, high = int(match.group(1)), match.group(2)
            parsed.append(Designed(low, int(high) if high else low))
        else:
            raise ValueError(f"cannot parse contig component {token!r}")

    if not parsed:
        raise ValueError("contig contains no components")
    return tuple(parsed)


def as_components(contig: str | Sequence[Component]) -> tuple[Component, ...]:
    """Accept either a contig string or an already-parsed one."""
    return parse_contig(contig) if isinstance(contig, str) else tuple(contig)


def parse_residue_selection(text: str) -> tuple[ResidueKey, ...]:
    """Input residues referenced by an `unindex` or `select_*` string.

    These share the contig punctuation but mean something different: bare
    integers are sequence offsets between unindexed components, not designed
    regions, and are ignored here. Only the chain-labelled residues matter for
    validating that a selection exists in the input and does not collide with
    the contig.
    """
    if not text or not text.strip():
        return ()

    residues: list[ResidueKey] = []
    for raw in text.split(","):
        token = raw.strip()
        if not token or _CHAIN_BREAK.match(token) or _DESIGNED.match(token):
            continue
        if match := _FROM_INPUT.match(token):
            chain, start, end = match.group(1), int(match.group(2)), match.group(3)
            residues.extend(FromInput(chain, start, int(end) if end else start).residues)
        else:
            raise ValueError(f"cannot parse residue selection component {token!r}")
    return tuple(residues)


def segments(components: Sequence[Component]) -> list[list[Component]]:
    """Split on chain breaks; each group becomes one output chain."""
    grouped: list[list[Component]] = [[]]
    for component in components:
        if isinstance(component, ChainBreak):
            grouped.append([])
        else:
            grouped[-1].append(component)
    return [group for group in grouped if group]


def output_chain_ids(count: int) -> list[str]:
    """Output chain labels: A, B, C, ... in contig order."""
    if count > len(ascii_uppercase):
        raise ValueError(f"{count} output chains exceeds the A-Z labelling scheme")
    return list(ascii_uppercase[:count])


def expected_output_chains(contig: str | Sequence[Component]) -> list[str]:
    """Output chain ids implied by a contig, in order."""
    return output_chain_ids(len(segments(as_components(contig))))


def _kept_residues(
    component: FromInput, present: dict[str, set[int]] | None
) -> list[ResidueKey]:
    """The component's residues that survive the presence filter."""
    if present is None:
        return component.residues
    in_chain = present.get(component.chain, set())
    return [key for key in component.residues if key[1] in in_chain]


def map_input_to_output(
    components: Sequence[Component],
    structure=None,
) -> dict[ResidueKey, ResidueKey]:
    """Map every input residue named by the contig to its output position.

    Args:
        components: parsed contig.
        structure: the input structure. When given, residues absent from it are
            skipped rather than counted, so gaps in a chain do not shift
            everything after them. Strongly recommended: without it this assumes
            every residue in every range is present.

    Returns:
        `{(input_chain, input_resi): (output_chain, output_resi)}`.

    Raises:
        IndeterminateMapping: an input residue follows a variable-length
            designed region within the same output chain.
    """
    present = residues_by_chain(structure) if structure is not None else None
    mapping: dict[ResidueKey, ResidueKey] = {}

    groups = segments(components)
    for chain_id, group in zip(output_chain_ids(len(groups)), groups):
        position = 0
        determinate = True

        for component in group:
            if isinstance(component, Designed):
                position += component.minimum
                determinate = determinate and component.is_fixed
                continue

            for key in _kept_residues(component, present):
                position += 1
                if not determinate:
                    raise IndeterminateMapping(
                        f"{format_residue(key)} follows a variable-length designed "
                        f"region in output chain {chain_id}; its position depends "
                        "on the sampled length. Use the sampled contig from the "
                        "design metadata instead."
                    )
                mapping[key] = (chain_id, position)

    return mapping


def derive_length(components: Sequence[Component], structure=None) -> tuple[int, int]:
    """Total design length as (minimum, maximum), across all chains.

    This is what RFD3's `length` field constrains, and the reason the pipeline
    never accepts a hand-typed value: `length` and `contig` encode overlapping
    information, so a typo in one produces designs of the wrong size with no
    error.

    With `structure`, input ranges are counted by residues actually present, so
    a gapped chain gives the true length rather than the nominal span.
    """
    present = residues_by_chain(structure) if structure is not None else None

    low = high = 0
    for component in components:
        if isinstance(component, ChainBreak):
            continue
        if isinstance(component, Designed):
            low += component.minimum
            high += component.maximum
        else:
            count = len(_kept_residues(component, present))
            low += count
            high += count
    return low, high


def renumber_hotspots(
    hotspots: str | Sequence[ResidueKey],
    contig: str | Sequence[Component],
    structure=None,
) -> dict[ResidueKey, ResidueKey]:
    """Hotspots in input numbering to their output chain and residue.

    Pure arithmetic over the contig. Choosing a sampled contig over the declared
    one, and cross-checking the result against RFD3's own map, are separate
    concerns handled where a design's metadata actually exists -- see
    `spec.design`.

    Raises:
        ValueError: a hotspot is not covered by the contig.
    """
    requested = (
        parse_residue_selection(hotspots) if isinstance(hotspots, str) else tuple(hotspots)
    )
    if not requested:
        return {}

    mapping = map_input_to_output(as_components(contig), structure=structure)

    missing = [key for key in requested if key not in mapping]
    if missing:
        raise ValueError(
            f"{len(missing)} hotspot(s) are not taken from the input by this "
            f"contig, e.g. {format_residues(missing)}. A hotspot must lie inside "
            "a contig range."
        )

    return {key: mapping[key] for key in requested}
