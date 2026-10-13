from __future__ import annotations

import pytest
import yaml
from conftest import CAMPAIGN_YAML, FIXTURES, REPO_ROOT, build

from rfd3_pipeline.config import CampaignConfig
from rfd3_pipeline.filters import backbone_record
from rfd3_pipeline.gates import (
    Gate,
    Stage,
    backbone_gates,
    evaluate_gates,
    metric_rules,
)
from rfd3_pipeline.spec import load_design_metadata, resolve_campaign
from rfd3_pipeline.structure import load_structure

DESIGN_CIF = FIXTURES / "specs_rgd_core_0_model_0.cif.gz"
DESIGN_JSON = FIXTURES / "specs_rgd_core_0_model_0.json"


class TestGate:
    def test_upper_bound(self):
        gate = Gate("loop_fraction", "<=", 0.5)
        assert gate.passes(0.25) and gate.passes(0.5)
        assert not gate.passes(0.51)

    def test_lower_bound(self):
        gate = Gate("hotspot_fraction", ">=", 0.4)
        assert gate.passes(0.4) and gate.passes(0.95)
        assert not gate.passes(0.39)

    def test_describe_names_the_violated_comparison(self):
        assert Gate("loop_fraction", "<=", 0.5).describe(0.72) == (
            "loop_fraction 0.72 > 0.5"
        )
        assert Gate("hotspot_fraction", ">=", 0.4).describe(0.1) == (
            "hotspot_fraction 0.1 < 0.4"
        )

    def test_rule_is_the_summary_json_form(self):
        assert Gate("rg_ratio", "<=", 1.1).rule == "<= 1.1"


class TestEvaluateGates:
    GATES = (Gate("a", "<=", 1.0), Gate("b", ">=", 0.5), Gate("c", "<=", 0.0))

    def test_all_passing(self):
        result = evaluate_gates(self.GATES, {"a": 0.5, "b": 0.9, "c": 0.0})
        assert result.passed
        assert result.failures == []
        assert result.columns == {"a_pass": True, "b_pass": True, "c_pass": True}

    def test_reports_every_failure_not_just_the_first(self):
        """summary.json counts failures per metric, so all of them must appear."""
        result = evaluate_gates(self.GATES, {"a": 9.0, "b": 0.1, "c": 3.0})
        assert not result.passed
        assert result.failed_metrics == ["a", "b", "c"]
        assert len(result.failures) == 3

    def test_failures_follow_gate_order(self):
        result = evaluate_gates(self.GATES, {"a": 9.0, "b": 0.1, "c": 0.0})
        assert result.fail_reasons == "a 9 > 1; b 0.1 < 0.5"

    def test_prefix_applies_to_columns(self):
        result = evaluate_gates((Gate("a", "<=", 1.0),), {"a": 0.0}, prefix="bb")
        assert result.columns == {"bb_a_pass": True}

    def test_missing_metric_is_a_bug_not_a_failure(self):
        """Within a stage every gated value must exist; stage_reached excludes
        designs that never got here."""
        with pytest.raises(KeyError, match="not computed"):
            evaluate_gates(self.GATES, {"a": 0.5})

    def test_metric_rules(self):
        assert metric_rules(self.GATES) == {"a": "<= 1", "b": ">= 0.5", "c": "<= 0"}


class TestBackboneGates:
    def test_read_from_the_campaign(self, raw):
        gates = backbone_gates(CampaignConfig.model_validate(raw).rfdiffusion3)
        assert [g.metric for g in gates] == [
            "clashes_backbone",
            "clashes_sidechain",
            "loop_fraction",
            "rg_ratio",
            "hotspot_fraction",
        ]
        assert metric_rules(gates)["rg_ratio"] == "<= 1.1"
        assert metric_rules(gates)["hotspot_fraction"] == ">= 0.4"


@pytest.fixture(scope="module")
def real_campaign():
    return CampaignConfig.model_validate(yaml.safe_load(CAMPAIGN_YAML.read_text()))


@pytest.fixture(scope="module")
def real_inputs(real_campaign):
    if not DESIGN_CIF.exists():
        pytest.skip(f"design fixture not present: {DESIGN_CIF}")
    resolved = resolve_campaign(real_campaign, structure_root=REPO_ROOT)
    spec = next(s for s in resolved.specs if s.name == "rgd_core")
    design = load_structure(DESIGN_CIF)
    metadata = load_design_metadata(DESIGN_JSON, binder_chain="A")
    return design, metadata, spec


class TestBackboneRecordOnTheRealDesign:
    def test_the_design_passes_every_gate(self, real_inputs, real_campaign):
        design, metadata, spec = real_inputs
        record = backbone_record(design, metadata, spec, real_campaign)
        assert record["bb_pass"] is True
        assert record["fail_reasons"] == ""
        assert record["fail_stage"] is None
        assert record["stage_reached"] == Stage.BACKBONE_FILTERS

    def test_clash_and_loop_values_come_from_rfd3(self, real_inputs, real_campaign):
        design, metadata, spec = real_inputs
        record = backbone_record(design, metadata, spec, real_campaign)
        assert record["bb_clashes_backbone"] == 0.0
        assert record["bb_clashes_sidechain"] == 0.0
        assert record["bb_loop_fraction"] == pytest.approx(0.2523, abs=1e-3)

    def test_rg_is_ours_with_rfd3s_recorded_alongside(self, real_inputs, real_campaign):
        """Ours is gated because it is unambiguously the binder chain."""
        design, metadata, spec = real_inputs
        record = backbone_record(design, metadata, spec, real_campaign)
        assert record["bb_binder_length"] == 110
        assert record["bb_rg"] == pytest.approx(12.465, abs=1e-2)
        assert record["bb_rfd3_rg"] == pytest.approx(12.577, abs=1e-2)
        assert record["bb_rg_ratio"] == pytest.approx(0.942, abs=1e-2)

    def test_hotspot_fraction_is_pooled_across_target_chains(
        self, real_inputs, real_campaign
    ):
        """The threshold is a fraction of all hotspots the campaign listed, so a
        split epitope must not be penalised for the split."""
        design, metadata, spec = real_inputs
        record = backbone_record(design, metadata, spec, real_campaign)
        assert record["bb_n_hotspots_total"] == 19  # 7 on B, 12 on C
        assert record["bb_n_hotspots_hit"] == 18
        assert record["bb_hotspot_fraction"] == pytest.approx(18 / 19)

    def test_off_target_spillover_is_recorded_not_gated(self, real_inputs, real_campaign):
        """Precision below recall means the epitope reaches beyond the hotspots."""
        design, metadata, spec = real_inputs
        record = backbone_record(design, metadata, spec, real_campaign)
        assert record["bb_epitope_recall_B"] == pytest.approx(1.0)
        assert record["bb_epitope_precision_B"] < record["bb_epitope_recall_B"]

    def test_mpnn_argument_uses_output_numbering(self, real_inputs, real_campaign):
        design, metadata, spec = real_inputs
        record = backbone_record(design, metadata, spec, real_campaign)
        assert record["mpnn_fixed_residues"] == "A64 A65 A66"
        assert record["motif_map"] == "A77>A64;A78>A65;A79>A66"

    def test_a_tighter_threshold_fails_it_with_a_reason(self, real_inputs, raw):
        """Same design, stricter campaign: the gate layer is what changes."""
        design, metadata, spec = real_inputs
        strict = CampaignConfig.model_validate(
            build(raw, rfdiffusion3={"threshold_loop_fraction": 0.1})
        )
        record = backbone_record(design, metadata, spec, strict)
        assert record["bb_pass"] is False
        assert record["bb_loop_fraction_pass"] is False
        assert "loop_fraction" in record["fail_reasons"]
        assert record["fail_stage"] == Stage.BACKBONE_FILTERS

    def test_missing_rfd3_metric_raises(self, real_inputs, real_campaign):
        design, metadata, spec = real_inputs
        stripped = type(metadata)(
            binder_entries=metadata.binder_entries,
            target_entries=metadata.target_entries,
            sampled_contig=metadata.sampled_contig,
            metrics={},
        )
        with pytest.raises(KeyError, match="missing gated RFD3 metric"):
            backbone_record(design, stripped, spec, real_campaign)
