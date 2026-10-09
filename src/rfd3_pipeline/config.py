"""Campaign YAML schema.

Validated before any GPU is allocated -- and before the input structure is even
opened -- so a bad campaign file fails in seconds rather than forty minutes into
RFD3.

Everything checkable from the YAML alone is checked here, including contig and
selection syntax: `CampaignConfig.model_validate` succeeding means the campaign
is well-formed, not merely well-shaped. Checks that genuinely need coordinates
-- residues existing, selections colliding, gap-aware lengths -- live in
`spec.resolve`.

Per-model differences live in one `MODEL_CAPABILITIES` table rather than in
scattered conditionals. Adding a model is a row; omitting a field is a KeyError
at import rather than a silent fall-through.
"""

from __future__ import annotations

from string import ascii_letters, digits

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contig import (
    Component,
    expected_output_chains,
    parse_contig,
    parse_residue_selection,
)

__all__ = [
    "DRY_RUN_OVERRIDES",
    "DRY_RUN_SUFFIX",
    "CampaignConfig",
    "CrossValidationConfig",
    "apply_dry_run",
    "dry_run_banner",
    "FoldingModel",
    "GeneralConfig",
    "IpsaeConfig",
    "KitCard",
    "MODEL_CAPABILITIES",
    "Mode",
    "ModelCapabilities",
    "MpnnConfig",
    "MpnnType",
    "MsaConfig",
    "RefoldingConfig",
    "Rfd3Config",
    "RunConfig",
    "SeedStyle",
    "SpecConfig",
    "Weights",
    "folding_variant",
    "seed_sequence",
]


class Mode(StrEnum):
    """Anthropic kit acceleration mode. A mode that cannot engage exits 3."""

    OFF = "off"
    EXACT = "exact"
    FAST = "fast"
    BIG = "big"


class KitCard(StrEnum):
    """`--config` card. The kits ship these three only."""

    H100 = "h100"
    H200 = "h200"
    A100 = "a100"


class FoldingModel(StrEnum):
    ESMFOLD2 = "esmfold2"
    ESMFOLD2_FAST = "esmfold2-fast"
    BOLTZ2 = "boltz2"
    AF3_JAX = "af3_jax"
    # The sokrypton/alphafold3 fork driven directly, without the Anthropic kit.
    # A separate member rather than a flag on AF3_JAX because almost nothing is
    # shared: no `run.sh`, no `--config` card, no `--mode`, no `--variant`, a
    # different weights layout, and native AF3 weights that actually load.
    AF3_FORK = "af3_fork"


class Weights(StrEnum):
    """Which checkpoint af3_jax loads. The kit ships OpenFold3 by default."""

    OPENFOLD3 = "openfold3"
    AF3_NATIVE = "af3_native"


class MpnnType(StrEnum):
    SOLUBLE = "soluble"
    PROTEIN = "protein"
    LIGAND = "ligand"


class SeedStyle(StrEnum):
    """How a model is asked for several seeds.

    LIST -- seeds chosen by us and reproducible, whether passed as
    `--seeds a,b,c` (esmfold2) or as the input JSON's own list (af3_jax's
    `modelSeeds`).
    COUNT -- `--num_seeds N`, the model deriving them from the one seed in its
    input.

    COUNT is avoidable on every model here, and avoiding it is preferred.
    Verified 2026-10-09 against the AF3 fork: `--num_seeds` is *optional*
    (`run_alphafold.py:414-424` -- "If not set, AlphaFold 3 will use the seeds
    as provided in the input JSON"), and even when set the seeds are
    `range(base, base + N)` (`folding_input.py:1514`), which is exactly what
    `seed_sequence` already computes rather than anything random. The plan's
    note that the fork's seeds are "readable only from its output JSON" was
    wrong on both counts.
    """

    LIST = "list"
    COUNT = "count"


@dataclass(frozen=True)
class ModelCapabilities:
    """What one folding model supports. One row per model, no conditionals."""

    # Whether this model is driven through an Anthropic optimization kit
    # (`run.sh`, `--config <card>`, `--mode`, `--variant`) or invoked directly.
    # Required rather than defaulted, so a new row has to say which it is: the
    # kit-only rules below -- the GPU needing a card, `--mode` meaning
    # anything, `n_gpu_shard` sharding a prediction -- are all false for a
    # model called directly, and silently applying them would restrict GPUs
    # for no reason and accept settings that reach nothing.
    uses_kit: bool
    supports_templates: bool
    # Which MSA fields a protein chain carries in this model's input, which is
    # what the caller has to write. Two flags rather than one enum because the
    # question at assembly time is per-field ("emit an unpaired one? a paired
    # one?"), and because a campaign can then ask whether *any* of its models
    # needs paired alignments without pattern-matching an enum.
    #
    # `msa_paired` is the one that says the caller must do the pairing itself.
    # ESMFold2 takes a single a3m per chain and pairs internally; the AF3 fork
    # takes `unpaired_msa` and `paired_msa` separately. That difference decides
    # whether the MSA cache has to build paired alignments at all -- and paired
    # alignments are not cacheable in general, being a property of the set of
    # chains queried together rather than of one sequence.
    msa_unpaired: bool
    msa_paired: bool
    seed_style: SeedStyle
    # Keyed by whether the step supplies an MSA. None when the model's variant
    # is not derived per step.
    derived_variants: dict[bool, str] | None = None
    fixed_variant: str | None = None
    default_weights: Weights | None = None

    def __post_init__(self) -> None:
        # Pairing is a refinement of per-chain alignments, so paired-without-
        # unpaired is not a shape any of these models offers. Checked at import,
        # in keeping with the table's habit of failing on a bad row rather than
        # letting it through.
        if self.msa_paired and not self.msa_unpaired:
            raise ValueError(
                "msa_paired requires msa_unpaired: a model that takes paired "
                "alignments also takes the per-chain ones they refine"
            )

    @property
    def consumes_msa(self) -> bool:
        """Derived, not stored, so it cannot contradict the two flags.

        Storing it alongside them would allow `consumes_msa=True` with both
        flags false, which nothing would catch.
        """
        return self.msa_unpaired or self.msa_paired

    @property
    def accepts_weights(self) -> bool:
        return self.default_weights is not None

    def variant_for(self, msa_present: bool) -> str | None:
        if self.derived_variants is not None:
            return self.derived_variants[msa_present]
        return self.fixed_variant


MODEL_CAPABILITIES: dict[FoldingModel, ModelCapabilities] = {
    # `--variant` conflates model choice with MSA presence, so the campaign
    # names the model and the step supplies the MSA (esmfold2/README.md:72-73).
    FoldingModel.ESMFOLD2: ModelCapabilities(
        uses_kit=True,
        supports_templates=False,
        # One `"msa"` a3m path per chain (`opt/esmfold2_opt/inputs.py:7-9`);
        # the model pairs internally, so the caller never supplies pairing.
        msa_unpaired=True,
        msa_paired=False,
        seed_style=SeedStyle.LIST,
        derived_variants={False: "full_nomsa", True: "full_msa"},
    ),
    FoldingModel.ESMFOLD2_FAST: ModelCapabilities(
        uses_kit=True,
        supports_templates=False,
        # Reads no alignments at all, so `consumes_msa` derives to False and
        # the whole MSA cache is inert for a campaign using it.
        msa_unpaired=False,
        msa_paired=False,
        seed_style=SeedStyle.LIST,
        derived_variants={False: "fast", True: "fast"},
    ),
    FoldingModel.BOLTZ2: ModelCapabilities(
        uses_kit=True,
        # Templates unverified against the kit; left False so a campaign cannot
        # silently request something that may be ignored.
        supports_templates=False,
        msa_unpaired=True,
        # UNVERIFIED, and note the asymmetry with `supports_templates` above:
        # there, False prevents a silent drop. Here, False risks silently
        # *under*-supplying -- if boltz2 does want paired alignments we would
        # never build them, and the predictions would be quietly weaker rather
        # than refused. Check this row against the kit before trusting it.
        msa_paired=False,
        seed_style=SeedStyle.LIST,
    ),
    FoldingModel.AF3_JAX: ModelCapabilities(
        uses_kit=True,
        supports_templates=True,
        # LIST, not COUNT: the fold-input JSON carries its own `modelSeeds`
        # list, and `--num_seeds` is optional. Passing the list explicitly
        # keeps cross-validation's five seeds ours and reproducible, which
        # matters more here than anywhere -- CV exists to be compared against.
        # See the SeedStyle docstring for the source lines.
        # `unpaired_msa` and `paired_msa` are separate fields on each protein
        # chain (`msa_server.py`), so the caller does the pairing. This is the
        # row that makes the MSA cache need a paired path at all.
        msa_unpaired=True,
        msa_paired=True,
        seed_style=SeedStyle.LIST,
        fixed_variant="p2",
        default_weights=Weights.OPENFOLD3,
    ),
    # Verified end to end 2026-10-09 on real campaign shards: five seeds at one
    # diffusion sample each, targets read from the MSA cache, exit 0 and a full
    # confidence set. See `experiments/af3_fork_image.py`.
    FoldingModel.AF3_FORK: ModelCapabilities(
        uses_kit=False,
        supports_templates=True,
        # Same fold-input schema as af3_jax -- it is the same upstream code --
        # so the same two MSA fields per protein chain, and the caller still
        # owns the pairing.
        msa_unpaired=True,
        msa_paired=True,
        # `modelSeeds` in the input JSON. Measured, not assumed: two runs in
        # separate containers with seeds 0-4 returned identical metrics.
        seed_style=SeedStyle.LIST,
        # No `--variant`: the weights directory is named outright by
        # `--model_dir`, so there is nothing for a variant to select.
        # Both sets load into this one codebase, which is the fork's whole
        # point -- `af3_native` here is a weights choice, not a port.
        default_weights=Weights.OPENFOLD3,
    ),
}

# Modal GPU string -> kit card. GPUs absent from this table have no kit config,
# so a folding step cannot run on them: the mode would print NOT ACTIVE and
# exit 3 rather than falling back.
GPU_TO_CARD: dict[str, KitCard] = {
    "A100": KitCard.A100,
    "A100-40GB": KitCard.A100,
    "A100-80GB": KitCard.A100,
    "H100": KitCard.H100,
    "H200": KitCard.H200,
}

# `--n_gpu P` shards ONE prediction across P GPUs of a single host, and the kits
# accept only these values. It is not concurrency; that is `max_concurrent`.
VALID_GPU_SHARDS = (1, 2, 4, 8)


def folding_variant(model: FoldingModel, msa_present: bool) -> str | None:
    """The kit's `--variant` for a step, derived rather than configured."""
    return MODEL_CAPABILITIES[model].variant_for(msa_present)


def seed_sequence(base_seed: int, count: int) -> list[int]:
    """Concrete seeds for a step, so a re-run is comparable to its original."""
    if count < 1:
        raise ValueError(f"need at least one seed, got {count}")
    return [base_seed + offset for offset in range(count)]


def _split_commas(value: str) -> list[str]:
    """One definition of the comma-separated shorthand several fields accept."""
    return [part.strip() for part in value.split(",") if part.strip()]


class _Base(BaseModel):
    """Reject unknown keys: a typo should fail, not be silently ignored."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class GeneralConfig(_Base):
    chain_binder: str = "A"
    chain_targets: list[str]
    name_design_campaign: str
    run_name: str
    base_seed: int = 0
    seq_targets: dict[str, str] = Field(default_factory=dict)

    @field_validator("chain_targets", mode="before")
    @classmethod
    def _accept_comma_string(cls, value: Any) -> Any:
        return _split_commas(value) if isinstance(value, str) else value

    @model_validator(mode="before")
    @classmethod
    def _accept_comma_sequences(cls, data: Any) -> Any:
        """`seq_targets` as one comma-joined string, paired with chain_targets."""
        if not isinstance(data, dict) or not isinstance(data.get("seq_targets"), str):
            return data

        chains = data.get("chain_targets")
        if isinstance(chains, str):
            chains = _split_commas(chains)
        parts = _split_commas(data["seq_targets"])
        if not chains or len(chains) != len(parts):
            raise ValueError(
                f"seq_targets has {len(parts)} comma-separated sequence(s) but "
                f"chain_targets names {len(chains or [])} chain(s); use a mapping "
                "of chain id to sequence instead"
            )
        return {**data, "seq_targets": dict(zip(chains, parts))}

    @model_validator(mode="after")
    def _check_chains(self) -> GeneralConfig:
        if not self.chain_targets:
            raise ValueError("chain_targets must name at least one chain")
        if len(set(self.chain_targets)) != len(self.chain_targets):
            raise ValueError(f"duplicate target chains: {self.chain_targets}")
        if self.chain_binder in self.chain_targets:
            raise ValueError(
                f"chain_binder {self.chain_binder!r} also appears in chain_targets"
            )
        return self

    @property
    def all_chains(self) -> list[str]:
        """Expected output chains: binder first, then targets in contig order."""
        return [self.chain_binder, *self.chain_targets]


class SpecConfig(_Base):
    """One RFD3 design specification.

    An explicit `name` rather than a list index, so output keys stay stable when
    specs are added, removed or reordered -- otherwise a resumed run matches
    `skip_existing` against the previous run's outputs under shifted names.
    """

    name: str
    contig: str
    unindex: str | None = None
    select_fixed_atoms: dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name_survives_a_filename(cls, value: str) -> str:
        """Every output path in the pipeline descends from this string.

        `name` reaches `design_id`, then `seq_id`, then shard filenames, RFD3
        output keys and the AF3 fork's job directories. The fork is the strict
        one: it writes to `sanitised_name()`, dropping anything outside
        `[A-Za-z0-9_-.]` (`folding_input.py:1031-1035`), so a spec called
        "rgd core" would have its results land somewhere the pipeline does not
        look -- and cross-validation would exit 0 while reporting "no output
        directory" for every sequence. Two names differing only in a stripped
        character would be worse still: their outputs would collide silently.

        Rejected here rather than sanitised, because quietly renaming a user's
        spec would make the output directory not match the campaign file.
        """
        allowed = set(ascii_letters + digits + "_-.")
        bad = sorted({character for character in value if character not in allowed})
        if not value:
            raise ValueError("spec name must not be empty")
        if bad:
            raise ValueError(
                f"spec name {value!r} contains {bad}, which cannot survive a "
                f"filename; every output path descends from it. Use letters, "
                f"digits, underscore, hyphen or dot."
            )
        return value

    @model_validator(mode="after")
    def _check_selections(self) -> SpecConfig:
        # Parsed here so a malformed contig fails config validation rather than
        # surviving until the input structure is loaded.
        self.components  # noqa: B018 - raises on a bad contig

        unindexed = set(parse_residue_selection(self.unindex or ""))
        if self.select_fixed_atoms and not unindexed:
            raise ValueError(
                f"spec {self.name!r} sets select_fixed_atoms without unindex; "
                "there is nothing for it to apply to"
            )
        for selector in self.select_fixed_atoms:
            outside = sorted(set(parse_residue_selection(selector)) - unindexed)
            if outside:
                from .contig import format_residues

                raise ValueError(
                    f"spec {self.name!r}: select_fixed_atoms key {selector!r} "
                    f"covers residue(s) that are not unindexed, e.g. "
                    f"{format_residues(outside)}"
                )
        return self

    @property
    def components(self) -> tuple[Component, ...]:
        return parse_contig(self.contig)

    @property
    def unindexed_residues(self) -> tuple[tuple[str, int], ...]:
        return parse_residue_selection(self.unindex or "")


class Rfd3Config(_Base):
    gpu: str = "A100-80GB"
    max_concurrent: int = Field(default=1, ge=1)
    path_input_structure: str
    specs: list[SpecConfig]

    hotspots: str = ""
    infer_ori_strategy: Literal["com", "hotspots"] = "hotspots"
    is_non_loopy: bool | None = None
    dialect: int = 2
    ligand: str | None = None

    # Always derived from the contig; see spec.resolve. Present only so a YAML
    # carrying one is rejected loudly rather than silently believed.
    length: str | None = None

    # Top-level `seed` in RFD3's composed config, whose own default is `null`
    # (`configs/inference_engine/base.yaml`, merged at the root because the
    # engine option is `@package _global_`) -- so unseeded sampling is
    # upstream's behaviour, not an omission here. Left None at full scale on
    # purpose: once a dry run has shown the workflow sound, diverse backbones
    # across a campaign are the point. Set it to make a run repeatable, which
    # is what a dry run needs and what a controlled sampler comparison
    # requires -- `step_scale=3` against stock cannot be attributed if the
    # backbones differ between runs too.
    seed: int | None = None
    n_batches: int = Field(default=1, ge=1)
    diffusion_batch_size: int = Field(default=8, ge=1)

    num_timesteps: int = Field(default=200, ge=1)
    step_scale: float = 1.5
    gamma_0: float = 0.6

    threshold_interresidue_clashes_backbone: int = 0
    threshold_interresidue_clashes_sidechain: int = 3
    threshold_loop_fraction: float = 0.5
    threshold_radius_gyration_multiplier: float = 1.1
    threshold_hotspot_contacts: float = 0.4

    @field_validator("path_input_structure")
    @classmethod
    def _normalise_path(cls, value: str) -> str:
        """Backslashes do not resolve in Modal's Linux containers."""
        return value.replace("\\", "/")

    @field_validator("ligand")
    @classmethod
    def _ligand_ranges_are_not_supported(cls, value: str | None) -> str | None:
        """RFD3 splits ligand strings on ',' or '-', so a hyphen reads as a range."""
        if value and "-" in value:
            raise ValueError(
                f"ligand {value!r} contains '-', which RFD3 parses as a range "
                "pattern; list instances separately, e.g. 'E703,E704'"
            )
        return value

    @model_validator(mode="after")
    def _check_specs(self) -> Rfd3Config:
        if not self.specs:
            raise ValueError("at least one spec is required")
        names = [spec.name for spec in self.specs]
        if len(set(names)) != len(names):
            raise ValueError(f"spec names must be unique, got {names}")
        if self.length is not None:
            raise ValueError(
                f"length is derived from the contig, not configured (got "
                f"{self.length!r}). Remove it, or set it to null."
            )
        if self.infer_ori_strategy == "hotspots" and not self.hotspots:
            raise ValueError(
                "infer_ori_strategy is 'hotspots' but no hotspots are given"
            )
        parse_residue_selection(self.hotspots)  # raises on malformed hotspots
        return self

    @property
    def designs_per_spec(self) -> int:
        return self.n_batches * self.diffusion_batch_size

    @property
    def total_designs(self) -> int:
        return self.designs_per_spec * len(self.specs)


class MpnnConfig(_Base):
    gpu: str = "L40S"
    max_concurrent: int = Field(default=1, ge=1)
    mpnn_type: MpnnType = MpnnType.SOLUBLE
    mpnn_sampling_temp: float = Field(default=0.1, gt=0)
    num_seqs: int = Field(default=8, ge=1)
    rm_aa: str = "C"


class _FoldingBase(_Base):
    model: FoldingModel
    mode: Mode = Mode.FAST
    gpu: str = "A100-80GB"
    n_gpu_shard: int = 1
    max_concurrent: int = Field(default=1, ge=1)
    chunk_size: int = Field(default=16, ge=1)
    target_template: bool = False
    variant: str | None = None
    weights: Weights | None = None

    @model_validator(mode="after")
    def _check_folding(self) -> _FoldingBase:
        capabilities = MODEL_CAPABILITIES[self.model]

        # Kit-driven models only. The card is a kit concept, and a model called
        # directly runs on whatever GPU Modal will give it -- restricting it to
        # this table would rule out cards that work.
        if capabilities.uses_kit and self.gpu not in GPU_TO_CARD:
            raise ValueError(
                f"gpu {self.gpu!r} has no kit card; {self.model.value} needs one "
                f"of {sorted(GPU_TO_CARD)}. A mode that cannot engage exits 3."
            )

        if not capabilities.uses_kit and self.n_gpu_shard != 1:
            raise ValueError(
                f"n_gpu_shard={self.n_gpu_shard} is a kit setting and "
                f"{self.model.value} is called directly; it has no equivalent. "
                "For concurrency use max_concurrent."
            )

        if self.n_gpu_shard not in VALID_GPU_SHARDS:
            raise ValueError(
                f"n_gpu_shard must be one of {VALID_GPU_SHARDS}, got {self.n_gpu_shard}"
            )
        if self.n_gpu_shard > 1 and self.mode is not Mode.BIG:
            raise ValueError(
                f"n_gpu_shard={self.n_gpu_shard} requires mode 'big'; it shards one "
                f"prediction across GPUs, and mode is {self.mode.value!r}. For "
                "concurrency use max_concurrent."
            )

        if self.target_template and not capabilities.supports_templates:
            raise ValueError(
                f"{self.model.value} does not support templates; "
                "set target_template: false"
            )

        if capabilities.derived_variants is not None:
            if self.variant is not None:
                raise ValueError(
                    f"variant is derived per step for {self.model.value}, not set "
                    "in config: apo has no MSA, holo has one"
                )
        elif capabilities.fixed_variant is not None:
            if self.variant is None:
                self.variant = capabilities.fixed_variant
        elif self.variant is not None:
            raise ValueError(f"{self.model.value} takes no variant")

        if self.weights is not None and not capabilities.accepts_weights:
            raise ValueError(
                f"weights is meaningful only for models that ship several "
                f"checkpoints, not {self.model.value}"
            )
        if self.weights is None:
            self.weights = capabilities.default_weights

        return self

    @property
    def kit_config(self) -> KitCard:
        """The `--config` card, derived from the GPU rather than restated.

        Raises for a model called directly, which has no card. `mode` is in the
        same position: it is defaulted on this base, so a campaign cannot be
        stopped from setting it, but nothing reads it for a non-kit model.
        """
        if not self.capabilities.uses_kit:
            raise ValueError(
                f"{self.model.value} is called directly and has no kit card"
            )
        return GPU_TO_CARD[self.gpu]

    @property
    def capabilities(self) -> ModelCapabilities:
        return MODEL_CAPABILITIES[self.model]

    @property
    def consumes_msa(self) -> bool:
        """Whether this model reads a target MSA at all.

        False for esmfold2-fast, which means a cached target MSA is built and
        then ignored -- a legitimate cheap-screening choice, but one worth
        logging rather than discovering from a flat ipTM distribution.
        """
        return self.capabilities.consumes_msa

    @property
    def seed_style(self) -> SeedStyle:
        return self.capabilities.seed_style


class RefoldingConfig(_FoldingBase):
    threshold_plddt_binder: float = 70.0
    """The apo gate, and the whole of it. 70 is the user's prior practice -- a
    binder should fold on its own, not only when the target holds it.

    Apo also records `apo_rmsd_to_design`, which is **watched and deliberately
    not gated** [decided 2026-10-09]. Two notes for whoever reads it next: it
    is not an oversight, so it does not want a threshold inventing; and it is
    not dead, so the column should not be removed as unused. Apo has no
    interface to score and no second structure to place the binder against, so
    confidence is what it can gate on and reproducing the backbone is what it
    can report."""

    threshold_rmsd_apo_holo: float = 4.0
    """**Diagnostic only since 2026-10-09; the holo gate no longer applies
    it.** Still computed as `holo_rmsd_apo_holo`, because a binder that changes
    conformation on binding is worth seeing.

    Dropped from the gate because it measures the wrong thing for this
    pipeline's purpose and `threshold_plddt_binder` already covers most of what
    it was there for. It superimposes the binder on the binder, so it reports
    fold change and is blind to placement -- and it passed a design whose
    binder sat 21 A from where it was designed while rejecting one that placed
    the motif to 0.76 A. "Does it fold without the target" is the apo gate's
    question, asked earlier and more cheaply."""

    threshold_rmsd_binder_on_target: float = 5.0
    """The holo gate's placement test: fit on the targets, measure the binder.

    This is the column the gate was missing. Unlike every RMSD that preceded
    it, the superposition is fitted on the target chains, so the binder's
    displacement stays in the measurement instead of being fitted away.

    5.0 A is a starting value, not a calibrated one. On the only four sequences
    measured it separates them exactly as ipSAE does -- 3.77 and 4.90 passing
    against 15.59 and 20.98 -- which is suggestive and not evidence. Revisit
    once a real campaign has produced a distribution."""

    threshold_rmsd_motif: float = 2.0
    """The holo gate's motif test, applied **only when the campaign scaffolds a
    motif** -- that is, when a spec sets `unindex`. For such a campaign this is
    the design objective rather than a diagnostic: the scaffold is allowed to
    move, the motif is not.

    Declared and unused until 2026-10-09, when `rmsd_motif_on_target` was
    wired. The value is the campaign's own and was not re-chosen: measured
    motif RMSDs are 0.48, 0.76 and 0.83 against one at 7.24, so 2.0 separates
    them with room to spare -- though four sequences do not calibrate a
    threshold."""
    threshold_ipTM: float = 0.7
    """Raised from 0.3 on 2026-10-09, because at 0.3 it was doing no work.
    Across the four sequences measured, holo ipTM ran 0.735 to 0.908 -- every
    one of them clearing 0.3 by a wide margin, so the entire discriminating
    power of the gate sat in a single other term. 0.7 at least bites.

    It is still the weaker of the gate's interface signals. ipTM disagreed
    sharply with ipSAE on the same structures: one design scored 0.797 ipTM
    with 0.012 ipSAE and its binder 21 A out of place."""


class CrossValidationConfig(_FoldingBase):
    num_seeds: int = Field(default=5, ge=1)


class MsaConfig(_Base):
    host_url: str = "https://api.colabfold.com"
    use_env: bool = True
    use_filter: bool = True
    pairing_strategy: Literal["greedy", "complete"] = "greedy"
    username: str | None = None
    password: str | None = None
    auth_header: str | None = None

    @model_validator(mode="after")
    def _one_auth_method(self) -> MsaConfig:
        basic = self.username is not None or self.password is not None
        if basic and self.auth_header is not None:
            raise ValueError("use either username/password or auth_header, not both")
        if (self.username is None) != (self.password is None):
            raise ValueError("username and password must be given together")
        return self

    @property
    def needs_secret(self) -> bool:
        """Credentials are the one thing requiring a Modal secret."""
        return self.username is not None or self.auth_header is not None


class IpsaeConfig(_Base):
    pae_cutoff: float = 10.0
    """Read by `metrics.ipsae`, and applied to holo and cross-validation alike
    -- scores computed at different cutoffs would not be comparable, which is
    the whole point of cross-validating.

    **10, stricter than the reference implementations' 15** (which is why the
    plan recorded it explicitly: an unstated 10 would have been silently
    ignored). Briefly set to 15 on 2026-10-09 and reverted the same day.

    It is not a free parameter, and the measured consequence is worth knowing
    before anyone changes it again. ipSAE counts only residue pairs below this
    PAE, so a complex whose closest cross-chain confidence sits above it scores
    exactly **0**. Cross-validation does: OpenFold3's closest binder-target
    PAE on this campaign is 10.6 to 12.8, so `cv_ipsae_*` reads 0.0 at a cutoff
    of 10 and barely above it at 15. That is a real finding -- the independent
    model reproduces the binder's fold and is not confident about the interface
    -- rather than a miscalibration. `pae_floor_<target>` is recorded beside
    every score so a zero is readable rather than mysterious."""

    dist_cutoff: float = 10.0
    """**Not consumed yet.** `metrics/ipsae.py` is PAE-only; the Dunbrack
    reference CLI additionally takes a distance cutoff, and this records the
    value to use if the pipeline ever calls it or grows a distance-aware
    implementation. Kept rather than deleted so the intended value is not lost,
    but tuning it today changes nothing -- which is worth knowing before
    treating it as a knob."""


class RunConfig(_Base):
    dry_run: bool = False
    """Smoke-test scale: see `apply_dry_run`.

    A campaign that quietly produced a handful of designs because this was left
    on would be a miserable thing to debug, so it is never only a flag. It
    renames the run, so dry-run output lands in its own directory and can never
    be mistaken for a real campaign, and it is recorded in `summary.json`.
    """


# Appended to `run_name` whenever dry-run is active. The point is that the
# output directory itself says so -- nobody should have to find the flag to
# explain why a campaign made two designs.
DRY_RUN_SUFFIX = "-dryrun"

# What dry-run reduces. One spec, the smallest batch that still exercises the
# per-batch code paths, and the cheapest folding configuration. Everything else
# -- contigs, thresholds, chains, the ligand -- is left exactly as configured,
# because the point is to exercise the real campaign's shape at a scale where a
# mistake costs minutes.
DRY_RUN_OVERRIDES: dict[str, dict[str, Any]] = {
    # Three designs rather than two: enough that the backbone filters can both
    # pass and fail within one run, which two cannot reliably do.
    # `seed` is not here: it is derived from `general.base_seed` in
    # `apply_dry_run`, so a campaign that sets a base seed gets that one rather
    # than a literal pinned in this table.
    "rfdiffusion3": {"n_batches": 1, "diffusion_batch_size": 3, "max_concurrent": 1},
    "mpnn": {"num_seqs": 2, "max_concurrent": 1},
    "refolding": {
        "model": "esmfold2",
        "mode": "fast",
        "max_concurrent": 1,
        # The apo gate, because at smoke scale it otherwise blocks the last
        # two stages entirely: two sequences per backbone at temperature 0.1
        # are weak designs, and every one failed 70 while one of them still
        # reproduced its backbone to 1.31 A. A dry run exists to exercise the
        # pipeline's shape, so a gate that stops it before holo defeats that --
        # and lowering the real threshold to make a smoke test pass would be
        # the wrong fix.
        "threshold_plddt_binder": 40.0,
    },
    "cross_validation": {"num_seeds": 1, "max_concurrent": 1},
}


def apply_dry_run(payload: dict[str, Any]) -> dict[str, Any]:
    """Reduce a raw campaign payload to smoke-test scale.

    Operates on the parsed YAML before validation, so the result is validated
    like any other campaign and a bad override fails the same way a bad file
    would. Keeps only the first spec, and renames the run.

    Returns a new dict; the input is not modified.
    """
    reduced = {
        section: {**payload.get(section, {}), **changes}
        for section, changes in DRY_RUN_OVERRIDES.items()
        if section in payload
    }
    merged = {**payload, **reduced}

    specs = merged.get("rfdiffusion3", {}).get("specs") or []
    if specs:
        merged["rfdiffusion3"] = {**merged["rfdiffusion3"], "specs": specs[:1]}

    # Seed RFD3 for a dry run, derived from the campaign's base seed rather
    # than pinned in DRY_RUN_OVERRIDES so a campaign that sets one gets its
    # own. A smoke test has to be repeatable to be worth comparing against; a
    # full campaign deliberately leaves this None, because once the workflow is
    # known sound, diverse backbones are the point. An explicit seed in the
    # campaign is respected rather than overwritten.
    if "rfdiffusion3" in merged and merged["rfdiffusion3"].get("seed") is None:
        base_seed = (payload.get("general") or {}).get("base_seed", 0)
        merged["rfdiffusion3"] = {**merged["rfdiffusion3"], "seed": base_seed}

    general = dict(merged.get("general", {}))
    run_name = str(general.get("run_name", "run"))
    if not run_name.endswith(DRY_RUN_SUFFIX):
        general["run_name"] = run_name + DRY_RUN_SUFFIX
    merged["general"] = general

    merged["run"] = {**merged.get("run", {}), "dry_run": True}
    return merged


def dry_run_banner(campaign: CampaignConfig) -> str:
    """A loud, specific description of what dry-run changed."""
    rfd3 = campaign.rfdiffusion3
    return (
        "DRY RUN -- smoke-test scale, not a real campaign.\n"
        f"  run_name : {campaign.general.run_name} "
        f"(suffixed, so output cannot be confused with a real run)\n"
        f"  specs    : {len(rfd3.specs)} ({rfd3.specs[0].name})\n"
        f"  designs  : {rfd3.total_designs} "
        f"({rfd3.n_batches} batch x {rfd3.diffusion_batch_size})\n"
        f"  seqs/des : {campaign.mpnn.num_seqs}\n"
        f"  cv seeds : {campaign.cross_validation.num_seeds}\n"
        "  Remove --dry-run for the configured campaign."
    )


class CampaignConfig(_Base):
    general: GeneralConfig
    rfdiffusion3: Rfd3Config
    mpnn: MpnnConfig
    refolding: RefoldingConfig
    cross_validation: CrossValidationConfig
    msa: MsaConfig = Field(default_factory=MsaConfig)
    ipsae: IpsaeConfig = Field(default_factory=IpsaeConfig)
    run: RunConfig = Field(default_factory=RunConfig)

    @model_validator(mode="after")
    def _refolding_model_reads_msas(self) -> "CampaignConfig":
        """The refolding model must consume MSAs.

        Holo folds the binder against its targets, and a target folded from
        sequence alone is a worse model of the thing being bound -- so the
        interface metrics would still be produced, and would describe a complex
        nobody asked for. Rejected here rather than warned about, because
        choosing the refolding model is a decision to make deliberately: the
        user confirms the model supports alignments.

        Config-time rather than at the stage, so it fails on the client in
        seconds instead of after RFD3 and MPNN have spent GPU time.

        The practical effect is that `esmfold2-fast` cannot be a *refolding*
        model. That is the point of keeping it in the enum and the capability
        table: the schema can name what it is and refuse it for this role,
        rather than the choice silently degrading holo.
        """
        if not self.refolding.consumes_msa:
            raise ValueError(
                f"refolding.model {self.refolding.model.value!r} does not read "
                f"MSAs (consumes_msa is False), so holo would fold its targets "
                f"from sequence alone and still report interface metrics. "
                f"Choose a model that supports alignments."
            )
        return self

    @model_validator(mode="after")
    def _cross_validation_uses_another_model(self) -> "CampaignConfig":
        """Cross-validation must not use the refolding model.

        The point of the stage is agreement between *independent* models, so the
        same model on both sides is not cross-anything: it would run a second
        full folding pass and report a model agreeing with itself. Nothing would
        fail, and the resulting ranks would look like corroboration.

        It also underwrites the one-image-per-model dispatch in `app.py`: a
        Modal function's image is fixed at decoration time, so the two stages
        are always two functions, and this guarantees the two models really do
        differ rather than happening to.
        """
        if self.cross_validation.model == self.refolding.model:
            raise ValueError(
                f"cross_validation.model and refolding.model are both "
                f"{self.refolding.model.value!r}; cross-validation must use a "
                f"different model or it validates a model against itself"
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def _accept_hyphenated_key(cls, data: Any) -> Any:
        """The original YAML used `cross-validation`, awkward to reach in Python."""
        if isinstance(data, dict) and "cross-validation" in data:
            if "cross_validation" in data:
                raise ValueError(
                    "both 'cross-validation' and 'cross_validation' are present; "
                    "keep only 'cross_validation'"
                )
            data = dict(data)
            data["cross_validation"] = data.pop("cross-validation")
        return data

    @model_validator(mode="after")
    def _check_campaign(self) -> CampaignConfig:
        declared = set(self.general.chain_targets)
        described = set(self.general.seq_targets)
        if described and described != declared:
            raise ValueError(
                f"seq_targets describes chains {sorted(described)} but "
                f"chain_targets names {sorted(declared)}"
            )

        # Text-only, so it belongs here rather than waiting for the structure.
        expected = self.general.all_chains
        for spec in self.rfdiffusion3.specs:
            chains = expected_output_chains(spec.components)
            if chains != expected:
                raise ValueError(
                    f"spec {spec.name!r}: the contig produces output chains "
                    f"{chains}, but the campaign declares {expected}. Output "
                    "chains are lettered from A in contig order, so the binder "
                    "segment comes first and one segment must follow per "
                    "declared target chain."
                )
        return self

    @property
    def cross_validation_seeds(self) -> list[int]:
        return seed_sequence(self.general.base_seed, self.cross_validation.num_seeds)
