"""Resolve a campaign against its input structure, and emit the RFD3 spec file.

`config.py` checks everything a YAML can be checked for on its own, including
contig syntax. This module adds only what needs coordinates -- residues
existing, selections colliding with real residues, gap-aware lengths, declared
sequences matching the file -- and derives the values that must never be typed
by hand.

Resolution is all-or-nothing and runs before any GPU is allocated: a campaign
either produces a `ResolvedCampaign` or raises. The result holds plain data only,
so it can be written to `campaign.resolved.yaml` and handed between Modal
functions; steps that need coordinates load the structure themselves from
`structure_path`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from ..config import CampaignConfig, SpecConfig
from ..contig import (
    FromInput,
    ResidueKey,
    derive_length,
    expected_output_chains,
    format_residue,
    format_residues,
    map_input_to_output,
    renumber_hotspots,
)
from ..structure import chain_sequence, load_structure, residues_by_chain

__all__ = [
    "ResolvedCampaign",
    "ResolvedSpec",
    "resolve_campaign",
    "build_rfd3_specs",
    "dump_rfd3_specs",
]


@dataclass(frozen=True)
class ResolvedSpec:
    """One spec with everything the contig and structure imply filled in."""

    name: str
    contig: str
    unindex: str | None
    select_fixed_atoms: dict[str, str]
    length: tuple[int, int]
    output_chains: list[str]
    hotspots: dict[ResidueKey, ResidueKey]
    target_chain_map: dict[str, str]
    """Input chain id -> output chain id, for the chains the contig copies.

    Usually the identity, but only because contigs normally list targets in
    alphabetical order. A contig that lists them otherwise renames them, and
    anything keyed by chain -- `seq_targets`, and through it which MSA is handed
    to which chain -- has to be translated through this rather than assumed.
    """

    @property
    def length_string(self) -> str:
        """RFD3's `min-max` form, or a bare integer when the length is fixed."""
        low, high = self.length
        return str(low) if low == high else f"{low}-{high}"

    def hotspots_by_output_chain(self) -> dict[str, list[int]]:
        """Output-numbered hotspot residues, grouped for `binding_interface`.

        Grouped here rather than at each call site, since every consumer of
        hotspots wants exactly this shape.
        """
        grouped: dict[str, list[int]] = {}
        for chain, resi in self.hotspots.values():
            grouped.setdefault(chain, []).append(resi)
        return {chain: sorted(resi) for chain, resi in grouped.items()}


@dataclass(frozen=True)
class ResolvedCampaign:
    """A validated campaign plus everything derived from its structure.

    Deliberately holds no `AtomArray`: this object crosses process boundaries,
    and a step that needs coordinates calls `load_structure(structure_path)`.
    """

    config: CampaignConfig
    structure_path: Path
    specs: list[ResolvedSpec]
    target_sequences: dict[str, str]
    cross_validation_seeds: list[int]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Plain-data form, for `campaign.resolved.yaml`."""
        return {
            "config": self.config.model_dump(mode="json"),
            "structure_path": str(self.structure_path),
            "specs": [asdict(spec) for spec in self.specs],
            "target_sequences": dict(self.target_sequences),
            "cross_validation_seeds": list(self.cross_validation_seeds),
            "warnings": list(self.warnings),
        }


def _check_contig_coverage(
    spec: SpecConfig, present: dict[str, set[int]]
) -> list[str]:
    """Every contig range must match real residues; partial matches warn."""
    warnings: list[str] = []
    for component in spec.components:
        if not isinstance(component, FromInput):
            continue
        in_chain = present.get(component.chain, set())
        found = [key for key in component.residues if key[1] in in_chain]
        if not found:
            raise ValueError(
                f"spec {spec.name!r}: contig component {component} matches no "
                "residues in the input structure"
            )
        absent = [key for key in component.residues if key[1] not in in_chain]
        if absent:
            warnings.append(
                f"spec {spec.name!r}: contig component {component} names "
                f"{len(absent)} residue(s) not in the input structure "
                f"(e.g. {format_residues(absent)}); they are skipped, so the "
                f"design is {len(found)} residues here rather than "
                f"{component.nominal_length}"
            )
    return warnings


def _build_chain_map(
    spec: SpecConfig, mapping: dict[ResidueKey, ResidueKey]
) -> tuple[dict[str, str], list[str]]:
    """Input chain -> output chain, with a warning when the contig renames one."""
    chain_map: dict[str, str] = {}
    for (input_chain, _), (output_chain, _) in mapping.items():
        previous = chain_map.setdefault(input_chain, output_chain)
        if previous != output_chain:
            raise ValueError(
                f"spec {spec.name!r}: input chain {input_chain} is split across "
                f"output chains {previous} and {output_chain}; the pipeline keys "
                "sequences and MSAs by chain, so a split chain is ambiguous"
            )

    renamed = {
        source: destination
        for source, destination in chain_map.items()
        if source != destination
    }
    warnings = (
        [
            f"spec {spec.name!r}: the contig renames chains "
            + ", ".join(f"{s}->{d}" for s, d in sorted(renamed.items()))
            + ". seq_targets stays keyed by input chain; use "
            "ResolvedSpec.target_chain_map to reach the output chain."
        ]
        if renamed
        else []
    )
    return chain_map, warnings


def _check_selections(
    spec: SpecConfig,
    unindexed: tuple[ResidueKey, ...],
    contig_residues: set[ResidueKey],
    present: dict[str, set[int]],
) -> None:
    """Unindexed residues must exist and must not also be taken by the contig."""
    absent = sorted(
        key for key in unindexed if key[1] not in present.get(key[0], set())
    )
    if absent:
        raise ValueError(
            f"spec {spec.name!r}: unindex names {len(absent)} residue(s) absent "
            f"from the input structure, e.g. {format_residues(absent)}"
        )

    # RFD3 raises on this too, but failing here costs seconds rather than a
    # container start and a checkpoint load.
    overlap = sorted(set(unindexed) & contig_residues)
    if overlap:
        raise ValueError(
            f"spec {spec.name!r}: {len(overlap)} residue(s) appear in both the "
            f"contig and unindex, e.g. {format_residues(overlap)}. A motif "
            "residue is either indexed by the contig or unindexed, never both."
        )


def _resolve_spec(
    spec: SpecConfig,
    campaign: CampaignConfig,
    structure,
    present: dict[str, set[int]],
) -> tuple[ResolvedSpec, list[str]]:
    components = spec.components
    warnings = _check_contig_coverage(spec, present)

    mapping = map_input_to_output(components, structure=structure)
    chain_map, rename_warnings = _build_chain_map(spec, mapping)
    warnings.extend(rename_warnings)

    _check_selections(spec, spec.unindexed_residues, set(mapping), present)

    resolved = ResolvedSpec(
        name=spec.name,
        contig=spec.contig,
        unindex=spec.unindex,
        select_fixed_atoms=dict(spec.select_fixed_atoms),
        length=derive_length(components, structure=structure),
        output_chains=expected_output_chains(components),
        hotspots=renumber_hotspots(
            campaign.rfdiffusion3.hotspots, components, structure=structure
        ),
        target_chain_map=chain_map,
    )
    return resolved, warnings


def _resolve_target_sequences(
    campaign: CampaignConfig, structure
) -> tuple[dict[str, str], list[str]]:
    """Target sequences read from the structure, checked against the config.

    Read here once so the MSA cache and the ipSAE chain lengths share one
    definition -- the cache hashes these strings, so a second implementation
    with a different unknown-residue character would key the same chain twice.
    """
    observed = {
        chain: chain_sequence(structure, chain)
        for chain in campaign.general.chain_targets
    }
    declared = campaign.general.seq_targets

    if not declared:
        return observed, [
            "seq_targets is empty; target sequences were read from the input "
            "structure for the MSA cache"
        ]

    for chain, sequence in declared.items():
        actual = observed[chain]
        if len(actual) != len(sequence):
            raise ValueError(
                f"seq_targets[{chain!r}] has {len(sequence)} residues but chain "
                f"{chain} has {len(actual)} alpha carbons in the input structure"
            )
        if actual != sequence:
            at = next(i for i, (a, b) in enumerate(zip(actual, sequence)) if a != b)
            raise ValueError(
                f"seq_targets[{chain!r}] disagrees with the structure at position "
                f"{at}: structure has {actual[at]!r}, config has {sequence[at]!r}"
            )
    return observed, []


def resolve_campaign(
    campaign: CampaignConfig,
    structure=None,
    structure_root: str | Path = ".",
) -> ResolvedCampaign:
    """Validate a campaign against its input structure and fill in derived values.

    Args:
        campaign: a validated `CampaignConfig`.
        structure: an already-loaded structure, mainly for tests. Loaded from
            `path_input_structure` when omitted.
        structure_root: base for a relative `path_input_structure`.

    Raises:
        ValueError: any inconsistency between the config and the structure.
    """
    path = Path(structure_root) / campaign.rfdiffusion3.path_input_structure
    if structure is None:
        if not path.exists():
            raise FileNotFoundError(f"input structure not found: {path}")
        structure = load_structure(path)

    present = residues_by_chain(structure)
    warnings: list[str] = []
    specs: list[ResolvedSpec] = []

    for spec in campaign.rfdiffusion3.specs:
        resolved, spec_warnings = _resolve_spec(spec, campaign, structure, present)
        specs.append(resolved)
        warnings.extend(spec_warnings)

    target_sequences, sequence_warnings = _resolve_target_sequences(campaign, structure)
    warnings.extend(sequence_warnings)

    if not campaign.refolding.consumes_msa:
        warnings.append(
            f"refolding model {campaign.refolding.model.value} does not read MSAs; "
            "the cached target MSA will be built but unused for holo"
        )
    if campaign.msa.needs_secret:
        warnings.append(
            "msa credentials are set, which is the one case requiring a Modal secret"
        )

    return ResolvedCampaign(
        config=campaign,
        structure_path=path,
        specs=specs,
        target_sequences=target_sequences,
        cross_validation_seeds=campaign.cross_validation_seeds,
        warnings=warnings,
    )


def build_rfd3_specs(resolved: ResolvedCampaign) -> dict[str, dict]:
    """The RFD3 input mapping: one key per spec, shared fields copied onto each.

    `length` is deliberately omitted. It is optional when a contig is present,
    it constrains the contig rather than the whole system, and emitting a derived
    copy of information the contig already carries just creates a second value to
    keep in sync. `ResolvedSpec.length` keeps it for assertions and reporting.
    """
    rfd3 = resolved.config.rfdiffusion3

    shared: dict[str, object] = {
        "input": rfd3.path_input_structure,
        "dialect": rfd3.dialect,
        "infer_ori_strategy": rfd3.infer_ori_strategy,
    }
    if rfd3.hotspots:
        shared["select_hotspots"] = rfd3.hotspots
    if rfd3.is_non_loopy is not None:
        shared["is_non_loopy"] = rfd3.is_non_loopy
    if rfd3.ligand:
        shared["ligand"] = rfd3.ligand

    specs: dict[str, dict] = {}
    for spec in resolved.specs:
        entry = dict(shared)
        entry["contig"] = spec.contig
        if spec.unindex:
            entry["unindex"] = spec.unindex
        if spec.select_fixed_atoms:
            entry["select_fixed_atoms"] = dict(spec.select_fixed_atoms)
        specs[spec.name] = entry
    return specs


def dump_rfd3_specs(resolved: ResolvedCampaign) -> str:
    """`build_rfd3_specs` as the YAML handed to `rfd3 design inputs=...`."""
    return yaml.safe_dump(
        build_rfd3_specs(resolved), sort_keys=False, default_flow_style=False
    )
