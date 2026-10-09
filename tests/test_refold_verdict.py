"""Apo and holo verdicts, and the columns the funnel counts.

The first version of this logic lived in `app.py` and compared thresholds by
hand. It produced correct `holo_pass` values, so it looked finished, while
being wrong in two ways that only `summary.json` would have shown: it named its
columns `holo_pass_iptm` where the funnel counts `holo_iptm_pass`, and it wrote
a stage-specific `holo_fail_reasons` instead of the unprefixed `fail_reasons`
that every other stage uses.
"""

from __future__ import annotations

from rfd3_pipeline.config import RefoldingConfig
from rfd3_pipeline.gates import Stage, refold_gates, refold_verdict


def refolding(**overrides) -> RefoldingConfig:
    settings = {
        "model": "esmfold2",
        "threshold_plddt_binder": 70.0,
        "threshold_ipTM": 0.3,
        "threshold_rmsd_apo_holo": 4.0,
    }
    settings.update(overrides)
    return RefoldingConfig(**settings)


class TestGateDeclarations:
    def test_apo_gates_on_plddt_alone(self):
        """Apo folds the binder by itself: no interface to score, and no second
        structure to compare against."""
        gates = refold_gates(refolding(), Stage.APO)
        assert [g.metric for g in gates] == ["plddt_binder"]

    def test_holo_gates_on_interface_and_drift(self):
        gates = refold_gates(refolding(), Stage.HOLO)
        assert [g.metric for g in gates] == ["iptm", "rmsd_apo_holo"]

    def test_cv_has_no_gates(self):
        """Nothing runs after it, so there is nothing to gate for."""
        assert refold_gates(refolding(), Stage.CV) == ()

    def test_thresholds_come_from_the_config(self):
        gates = refold_gates(refolding(threshold_ipTM=0.7), Stage.HOLO)
        assert gates[0].threshold == 0.7


class TestApo:
    def test_a_passing_fold(self):
        out = refold_verdict({"apo_plddt_binder": 82.0}, Stage.APO, refolding())
        assert out["apo_pass"] is True
        assert out["apo_plddt_binder_pass"] is True

    def test_a_failing_fold_names_the_comparison(self):
        out = refold_verdict({"apo_plddt_binder": 58.1}, Stage.APO, refolding())
        assert out["apo_pass"] is False
        assert out["fail_stage"] == Stage.APO
        assert out["fail_reasons"] == ["plddt_binder 58.1 < 70"]


class TestHolo:
    def test_both_gates_must_hold(self):
        out = refold_verdict(
            {"holo_iptm": 0.9, "holo_rmsd_apo_holo": 0.7}, Stage.HOLO, refolding()
        )
        assert out["holo_pass"] is True

    def test_strong_interface_still_fails_on_drift(self):
        """The case the real dry run produced: ipTM 0.888 with 5.6 A of
        apo/holo drift. A binder that changes shape on binding has interface
        numbers describing a conformation apo never vetted."""
        out = refold_verdict(
            {"holo_iptm": 0.888, "holo_rmsd_apo_holo": 5.627}, Stage.HOLO, refolding()
        )
        assert out["holo_pass"] is False
        assert out["holo_iptm_pass"] is True
        assert out["holo_rmsd_apo_holo_pass"] is False
        assert out["fail_reasons"] == ["rmsd_apo_holo 5.627 > 4"]

    def test_every_failure_is_reported_not_just_the_first(self):
        """The funnel counts failures per metric, so a sequence failing two
        gates has to appear in two counts."""
        out = refold_verdict(
            {"holo_iptm": 0.1, "holo_rmsd_apo_holo": 9.0}, Stage.HOLO, refolding()
        )
        assert out["holo_iptm_pass"] is False
        assert out["holo_rmsd_apo_holo_pass"] is False
        assert len(out["fail_reasons"]) == 2

    def test_the_column_names_are_the_ones_the_funnel_counts(self):
        """`{stage}_{metric}_pass`, not `{stage}_pass_{metric}`. Getting this
        wrong cost nothing visible and zeroed every per-metric count."""
        out = refold_verdict(
            {"holo_iptm": 0.9, "holo_rmsd_apo_holo": 0.7}, Stage.HOLO, refolding()
        )
        assert "holo_iptm_pass" in out and "holo_pass_iptm" not in out


class TestUnmeasured:
    def test_an_unmeasured_gate_fails_closed(self):
        """Absent when apo left no structure behind. Unknown is not a pass: the
        next stage is the most expensive in the pipeline."""
        out = refold_verdict(
            {"holo_iptm": 0.9, "holo_rmsd_apo_holo": None}, Stage.HOLO, refolding()
        )
        assert out["holo_pass"] is False
        assert out["fail_reasons"] == ["rmsd_apo_holo not measured"]

    def test_it_says_so_rather_than_blaming_a_threshold(self):
        out = refold_verdict({}, Stage.HOLO, refolding())
        assert out["fail_reasons"] == [
            "iptm not measured",
            "rmsd_apo_holo not measured",
        ]


class TestRerunsAreSelfCorrecting:
    def test_passing_retracts_this_stages_earlier_failure(self):
        """`REFOLD_SUPERSEDES` clears prefixed columns on a re-run; these two
        are unprefixed, so they are cleared here or not at all."""
        stale = {"holo_iptm": 0.9, "holo_rmsd_apo_holo": 0.7, "fail_stage": Stage.HOLO}
        out = refold_verdict(stale, Stage.HOLO, refolding())
        assert out["fail_stage"] is None
        assert out["fail_reasons"] == []

    def test_passing_does_not_erase_another_stages_failure(self):
        """Scoped to this stage, so a holo re-run cannot hide an apo failure."""
        out = refold_verdict(
            {"holo_iptm": 0.9, "holo_rmsd_apo_holo": 0.7, "fail_stage": Stage.APO},
            Stage.HOLO,
            refolding(),
        )
        assert "fail_stage" not in out

    def test_a_gateless_stage_writes_nothing(self):
        assert refold_verdict({"cv_iptm_mean": 0.6}, Stage.CV, refolding()) == {}
