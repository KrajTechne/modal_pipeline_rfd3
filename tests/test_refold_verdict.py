"""Apo and holo verdicts, and the columns the funnel counts.

The first version of this logic lived in `app.py` and compared thresholds by
hand. It produced correct `holo_pass` values, so it looked finished, while
being wrong in two ways that only `summary.json` would have shown: it named its
columns `holo_pass_iptm` where the funnel counts `holo_iptm_pass`, and it wrote
a stage-specific `holo_fail_reasons` instead of the unprefixed `fail_reasons`
that every other stage uses.

The gate was revised on 2026-10-09. Apo keeps `plddt_binder` -- it folds the
binder alone, so confidence is all it can judge. Holo moved from "fold change
plus ipTM" to **placement plus ipTM plus, when the campaign scaffolds a motif,
the motif's placement**, because the old pair passed a design whose binder sat
21 A from where it was designed and rejected one that placed the motif to
0.76 A.
"""

from __future__ import annotations

import pytest

from rfd3_pipeline.config import RefoldingConfig
from rfd3_pipeline.gates import Stage, refold_gates, refold_verdict


def refolding(**overrides) -> RefoldingConfig:
    settings = {
        "model": "esmfold2",
        "threshold_plddt_binder": 70.0,
        "threshold_ipTM": 0.7,
        "threshold_rmsd_binder_on_target": 5.0,
        "threshold_rmsd_motif": 2.0,
        "threshold_rmsd_apo_holo": 4.0,
    }
    settings.update(overrides)
    return RefoldingConfig(**settings)


def holo(**metrics) -> dict:
    """A holo metrics dict that passes everything unless overridden."""
    base = {
        "holo_iptm": 0.9,
        "holo_rmsd_binder_on_target": 2.0,
        "holo_rmsd_motif_on_target": 0.8,
        "holo_n_motif_residues": 3,
    }
    base.update(metrics)
    return base


class TestGateDeclarations:
    def test_apo_gates_on_confidence_alone(self):
        """Apo folds the binder by itself: no interface to score, and no second
        structure to place it against."""
        assert [g.metric for g in refold_gates(refolding(), Stage.APO)] == [
            "plddt_binder"
        ]

    def test_holo_gates_on_interface_and_placement(self):
        gates = refold_gates(refolding(), Stage.HOLO)
        assert [g.metric for g in gates] == ["iptm", "rmsd_binder_on_target"]

    def test_the_motif_gate_is_added_only_when_scaffolding_one(self):
        gates = refold_gates(refolding(), Stage.HOLO, has_motif=True)
        assert [g.metric for g in gates] == [
            "iptm",
            "rmsd_binder_on_target",
            "rmsd_motif_on_target",
        ]

    def test_fold_change_is_no_longer_gated(self):
        """`rmsd_apo_holo` is still computed and still has a threshold, but it
        measures fold change and is blind to placement, so it informs rather
        than decides."""
        for kwargs in ({}, {"has_motif": True}):
            metrics = [g.metric for g in refold_gates(refolding(), Stage.HOLO, **kwargs)]
            assert "rmsd_apo_holo" not in metrics

    def test_cv_has_no_gates(self):
        """Nothing runs after it. Its five criteria -- fold, placement, motif,
        ipTM and ipSAE -- inform ranking instead."""
        assert refold_gates(refolding(), Stage.CV) == ()

    def test_thresholds_come_from_the_config(self):
        gates = refold_gates(refolding(threshold_ipTM=0.55), Stage.HOLO)
        assert gates[0].threshold == 0.55


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

    def test_fold_rmsd_is_watched_not_gated(self):
        """Recorded by the stage and deliberately absent from its gate, so a
        poor backbone reproduction does not by itself reject a sequence."""
        out = refold_verdict(
            {"apo_plddt_binder": 82.0, "apo_rmsd_to_design": 9.9}, Stage.APO, refolding()
        )
        assert out["apo_pass"] is True


class TestHolo:
    def test_all_three_gates_must_hold(self):
        assert refold_verdict(holo(), Stage.HOLO, refolding())["holo_pass"] is True

    def test_a_displaced_binder_fails_however_confident(self):
        """The case that motivated the revision: ipTM 0.797 with the binder
        21 A from where it was designed used to pass."""
        out = refold_verdict(
            holo(holo_iptm=0.797, holo_rmsd_binder_on_target=20.98,
                 holo_rmsd_motif_on_target=7.24),
            Stage.HOLO,
            refolding(),
        )
        assert out["holo_pass"] is False
        assert out["holo_iptm_pass"] is True
        assert out["holo_rmsd_binder_on_target_pass"] is False

    def test_a_well_placed_motif_does_not_excuse_a_displaced_scaffold(self):
        """Measured on a real design: motif at 0.476 A with the whole binder
        15.6 A out, because the motif anchored and the scaffold swung away."""
        out = refold_verdict(
            holo(holo_rmsd_binder_on_target=15.59, holo_rmsd_motif_on_target=0.476),
            Stage.HOLO,
            refolding(),
        )
        assert out["holo_pass"] is False

    def test_a_misplaced_motif_fails_a_scaffolding_campaign(self):
        out = refold_verdict(
            holo(holo_rmsd_motif_on_target=7.24), Stage.HOLO, refolding()
        )
        assert out["holo_pass"] is False
        assert out["fail_reasons"] == ["rmsd_motif_on_target 7.24 > 2"]

    def test_a_design_with_no_motif_is_not_held_to_the_motif_gate(self):
        """Per design, not per campaign: a campaign can scaffold a motif while
        one design records none, and gating on a metric never computed would
        fail it for the wrong reason."""
        out = refold_verdict(
            holo(holo_n_motif_residues=0, holo_rmsd_motif_on_target=None),
            Stage.HOLO,
            refolding(),
        )
        assert out["holo_pass"] is True

    def test_every_failure_is_reported_not_just_the_first(self):
        """The funnel counts failures per metric, so a sequence failing two
        gates has to appear in two counts."""
        out = refold_verdict(
            holo(holo_iptm=0.1, holo_rmsd_binder_on_target=30.0),
            Stage.HOLO,
            refolding(),
        )
        assert out["holo_iptm_pass"] is False
        assert out["holo_rmsd_binder_on_target_pass"] is False
        assert len(out["fail_reasons"]) == 2

    def test_the_column_names_are_the_ones_the_funnel_counts(self):
        """`{stage}_{metric}_pass`, not `{stage}_pass_{metric}`. Getting this
        wrong cost nothing visible and zeroed every per-metric count."""
        out = refold_verdict(holo(), Stage.HOLO, refolding())
        assert "holo_iptm_pass" in out and "holo_pass_iptm" not in out


class TestUnmeasured:
    def test_an_unmeasured_gate_fails_closed(self):
        """Placement is absent when the design structure could not be read.
        Unknown is not a pass: the next stage is the most expensive one."""
        out = refold_verdict(
            holo(holo_rmsd_binder_on_target=None), Stage.HOLO, refolding()
        )
        assert out["holo_pass"] is False
        assert out["fail_reasons"] == ["rmsd_binder_on_target not measured"]

    def test_it_says_so_rather_than_blaming_a_threshold(self):
        out = refold_verdict({"holo_n_motif_residues": 3}, Stage.HOLO, refolding())
        assert out["fail_reasons"] == [
            "iptm not measured",
            "rmsd_binder_on_target not measured",
            "rmsd_motif_on_target not measured",
        ]


class TestRerunsAreSelfCorrecting:
    def test_passing_retracts_this_stages_earlier_failure(self):
        """`REFOLD_SUPERSEDES` clears prefixed columns on a re-run; these two
        are unprefixed, so they are cleared here or not at all."""
        out = refold_verdict(
            holo(fail_stage=Stage.HOLO), Stage.HOLO, refolding()
        )
        assert out["fail_stage"] is None
        assert out["fail_reasons"] == []

    def test_passing_does_not_erase_another_stages_failure(self):
        """Scoped to this stage, so a holo re-run cannot hide an apo failure."""
        out = refold_verdict(holo(fail_stage=Stage.APO), Stage.HOLO, refolding())
        assert "fail_stage" not in out

    def test_a_gateless_stage_writes_nothing(self):
        assert refold_verdict({"cv_iptm_mean": 0.6}, Stage.CV, refolding()) == {}
