"""Thresholds and pass/fail, in one place.

Metrics return numbers; nothing in `metrics/` knows a threshold. This module is
the single layer that compares them, so the set of gates exists as data rather
than being spread across the step that happens to compute each value. That is
what makes `fail_reasons` and the funnel in `summary.json` derivable instead of
assembled by hand.

A gate names a metric column, a comparison, and a threshold read from the
campaign config. Evaluating a set of them yields the `*_pass` columns, an
overall verdict, and the reasons -- always in the gates' declared order, so the
reason string for a given failure is stable across runs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "Gate",
    "GateResult",
    "Stage",
    "backbone_gates",
    "evaluate_gates",
    "metric_rules",
    "refold_gates",
    "refold_verdict",
]

Comparison = Literal["<=", ">="]


class Stage:
    """Pipeline stages, in order. `stage_reached` records how far a design got."""

    RFD3 = "rfd3"
    BACKBONE_FILTERS = "backbone_filters"
    MPNN = "mpnn"
    APO = "apo"
    HOLO = "holo"
    CV = "cv"

    ORDER = (RFD3, BACKBONE_FILTERS, MPNN, APO, HOLO, CV)


@dataclass(frozen=True)
class Gate:
    """One threshold comparison on one metric column."""

    metric: str
    comparison: Comparison
    threshold: float

    def passes(self, value: float) -> bool:
        if self.comparison == "<=":
            return value <= self.threshold
        return value >= self.threshold

    def describe(self, value: float) -> str:
        """`loop_fraction 0.72 > 0.5` -- the comparison that failed."""
        violated = ">" if self.comparison == "<=" else "<"
        return f"{self.metric} {value:g} {violated} {self.threshold:g}"

    @property
    def rule(self) -> str:
        """`<= 0.5` -- the form `summary.json` records per metric."""
        return f"{self.comparison} {self.threshold:g}"


@dataclass(frozen=True)
class GateResult:
    passed: bool
    columns: dict[str, bool]
    failures: list[str]
    """Human-readable failing comparisons, in gate order."""

    failed_metrics: list[str]
    """Just the metric names, for counting per-metric failures in summary.json."""

    @property
    def fail_reasons(self) -> str:
        return "; ".join(self.failures)


def evaluate_gates(
    gates: Sequence[Gate], values: Mapping[str, float], prefix: str = ""
) -> GateResult:
    """Apply every gate, reporting all failures rather than stopping at the first.

    Reporting all of them matters for the funnel: `summary.json` counts failures
    per metric, so a design that fails three gates has to appear in three counts.

    Raises:
        KeyError: a gated metric is absent. Within a stage every gated value
            must have been computed, so a missing one is a bug rather than a
            failure -- a design that never reached this stage is excluded by
            `stage_reached` instead.
    """
    missing = [gate.metric for gate in gates if gate.metric not in values]
    if missing:
        raise KeyError(
            f"gated metric(s) not computed: {missing}. Available: "
            f"{sorted(values)}"
        )

    head = f"{prefix}_" if prefix else ""
    columns: dict[str, bool] = {}
    failures: list[str] = []
    failed_metrics: list[str] = []

    for gate in gates:
        value = float(values[gate.metric])
        ok = gate.passes(value)
        columns[f"{head}{gate.metric}_pass"] = ok
        if not ok:
            failures.append(gate.describe(value))
            failed_metrics.append(gate.metric)

    return GateResult(
        passed=not failures,
        columns=columns,
        failures=failures,
        failed_metrics=failed_metrics,
    )


def backbone_gates(rfd3_config) -> tuple[Gate, ...]:
    """The five backbone filters, read from the campaign's `rfdiffusion3` block.

    Declared in the order the plan lists them, which is also the order
    `fail_reasons` reports.
    """
    return (
        Gate(
            "clashes_backbone",
            "<=",
            rfd3_config.threshold_interresidue_clashes_backbone,
        ),
        Gate(
            "clashes_sidechain",
            "<=",
            rfd3_config.threshold_interresidue_clashes_sidechain,
        ),
        Gate("loop_fraction", "<=", rfd3_config.threshold_loop_fraction),
        Gate("rg_ratio", "<=", rfd3_config.threshold_radius_gyration_multiplier),
        Gate("hotspot_fraction", ">=", rfd3_config.threshold_hotspot_contacts),
    )


def refold_gates(
    refolding_config, stage: str, *, has_motif: bool = False
) -> tuple[Gate, ...]:
    """The gates for one refolding stage, read from the `refolding` block.

    Declared here rather than applied by hand in `app.py` for the reason this
    module exists: the rules recorded in `summary.json` and the rules applied
    to the data come from one declaration and cannot drift apart. The first
    version of the holo gate restated its two thresholds in `app.py` and wrote
    `holo_pass_iptm` instead of the `{metric}_pass` form the funnel counts, so
    the per-metric failure counts silently stayed zero.

    **Apo asks whether the binder folds; holo asks where it went.** That split
    is a fact about what each stage can measure rather than a preference. Apo
    folds the binder alone, so there is no interface to score and no second
    structure to place it against -- `plddt_binder` is all it has. Holo has the
    complex, so it tests the interface (`iptm`) and the placement
    (`rmsd_binder_on_target`, fitted on the targets so displacement is not
    fitted away).

    `has_motif` adds the motif gate, and the caller decides it from the
    campaign rather than this module guessing: a spec that sets `unindex` is
    scaffolding a motif, and for such a campaign the motif's placement *is* the
    objective while the scaffold is free to move. Per-design the same question
    is `n_motif_residues > 0`.

    Deliberately **not** gated here any more: `rmsd_apo_holo`. See
    `RefoldingConfig.threshold_rmsd_apo_holo` for why -- it measures fold
    change, is blind to placement, and passed a design sitting 21 A from where
    it was designed.
    """
    if stage == Stage.APO:
        return (
            Gate("plddt_binder", ">=", refolding_config.threshold_plddt_binder),
        )
    if stage == Stage.HOLO:
        gates = [
            Gate("iptm", ">=", refolding_config.threshold_ipTM),
            Gate(
                "rmsd_binder_on_target",
                "<=",
                refolding_config.threshold_rmsd_binder_on_target,
            ),
        ]
        if has_motif:
            gates.append(
                Gate(
                    "rmsd_motif_on_target",
                    "<=",
                    refolding_config.threshold_rmsd_motif,
                )
            )
        return tuple(gates)
    # CV has no gates: nothing runs after it, so there is nothing to gate for.
    # Its numbers -- fold, placement, motif, ipTM and ipSAE -- inform ranking
    # instead, which is why all five are computed and none is thresholded.
    return ()


def refold_verdict(
    metrics: Mapping[str, object], stage: str, refolding_config
) -> dict:
    """This stage's gate columns, its verdict, and why it failed.

    The counterpart to `evaluate_gates` for a refolding stage: it reads the
    metrics the stage measured, applies `refold_gates`, and returns the columns
    to merge into the record. Kept here rather than in `app.py` so the
    thresholds, the column names and the verdict all come from one place --
    the first version compared thresholds in `app.py` and wrote
    `holo_pass_iptm` where the funnel counts `holo_iptm_pass`, which made every
    per-metric failure count read zero while sequences were in fact failing.

    Two deliberate departures from plain `evaluate_gates`:

    **An unmeasured gate is a failure, not an error.** `evaluate_gates` raises
    on a metric it cannot find, which is right when a missing value means a
    bug. Holo's `rmsd_apo_holo` is legitimately absent when apo left no
    structure behind, so that is settled here instead -- and settled as a
    failure, because the stage after holo is the most expensive in the pipeline
    and spending it on a sequence whose interface could not be measured buys
    nothing.

    **`fail_stage` and `fail_reasons` are unprefixed.** That is the pipeline's
    cross-stage answer to "why did this design die", written by `filters.py`
    since the backbone filters existed. A stage-specific
    `holo_fail_reasons` would leave those two columns null on a holo failure,
    so the one place a reader looks would say nothing.
    """
    # The motif gate is decided per design, not per campaign: a campaign can
    # scaffold a motif while an individual design records none, and gating on a
    # metric that was never computed would fail it for the wrong reason.
    prefix = f"{stage}_"
    gates = refold_gates(
        refolding_config,
        stage,
        has_motif=bool(metrics.get(f"{prefix}n_motif_residues")),
    )
    if not gates:
        return {}
    values = {gate.metric: metrics.get(f"{prefix}{gate.metric}") for gate in gates}
    unmeasured = sorted(metric for metric, value in values.items() if value is None)
    if unmeasured:
        return {
            f"{prefix}pass": False,
            "fail_stage": stage,
            "fail_reasons": [f"{metric} not measured" for metric in unmeasured],
        }

    result = evaluate_gates(gates, values, prefix=stage)
    columns: dict = {**result.columns, f"{prefix}pass": result.passed}
    if not result.passed:
        columns["fail_stage"] = stage
        columns["fail_reasons"] = result.failures
    elif metrics.get("fail_stage") == stage:
        # Retract this stage's own earlier verdict, and only its own. A re-run
        # that now passes must not leave the previous attempt's `fail_stage`
        # behind -- the same stale-column problem `REFOLD_SUPERSEDES` solves
        # for prefixed columns, which these two are not. Scoped to `== stage`
        # so passing holo cannot erase a failure recorded earlier.
        columns["fail_stage"] = None
        columns["fail_reasons"] = []
    return columns


def metric_rules(gates: Iterable[Gate]) -> dict[str, str]:
    """`{"loop_fraction": "<= 0.5", ...}` for the `summary.json` metrics block."""
    return {gate.metric: gate.rule for gate in gates}
