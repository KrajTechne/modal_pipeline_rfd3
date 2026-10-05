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


def metric_rules(gates: Iterable[Gate]) -> dict[str, str]:
    """`{"loop_fraction": "<= 0.5", ...}` for the `summary.json` metrics block."""
    return {gate.metric: gate.rule for gate in gates}
