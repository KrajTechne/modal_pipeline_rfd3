from __future__ import annotations

import pytest

from rfd3_pipeline.config import (
    DRY_RUN_SUFFIX,
    CampaignConfig,
    apply_dry_run,
    dry_run_banner,
)


@pytest.fixture
def dry(raw):
    return CampaignConfig.model_validate(apply_dry_run(raw))


class TestScaleReduction:
    def test_keeps_only_the_first_spec(self, raw, dry):
        assert len(raw["rfdiffusion3"]["specs"]) == 2
        assert [s.name for s in dry.rfdiffusion3.specs] == ["rgd_core"]

    def test_designs_drop_to_two(self, raw, dry):
        full = CampaignConfig.model_validate(raw)
        assert full.rfdiffusion3.total_designs == 80
        assert dry.rfdiffusion3.total_designs == 2

    def test_cheapest_folding_configuration(self, dry):
        assert dry.refolding.model.value == "esmfold2"
        assert dry.refolding.mode.value == "fast"
        assert dry.cross_validation.num_seeds == 1
        assert dry.mpnn.num_seqs == 2

    def test_concurrency_collapses_to_one(self, dry):
        assert dry.rfdiffusion3.max_concurrent == 1
        assert dry.refolding.max_concurrent == 1

    def test_the_campaign_itself_is_untouched(self, raw, dry):
        """Contigs, thresholds, chains and the ligand must survive, or the dry
        run stops exercising the real campaign's shape."""
        full = CampaignConfig.model_validate(raw)
        assert dry.rfdiffusion3.specs[0].contig == full.rfdiffusion3.specs[0].contig
        assert dry.rfdiffusion3.ligand == full.rfdiffusion3.ligand
        assert dry.rfdiffusion3.hotspots == full.rfdiffusion3.hotspots
        assert dry.general.chain_targets == full.general.chain_targets
        assert (
            dry.rfdiffusion3.threshold_hotspot_contacts
            == full.rfdiffusion3.threshold_hotspot_contacts
        )


class TestItCannotBeMistakenForARealRun:
    def test_run_name_is_renamed_so_output_lands_elsewhere(self, raw, dry):
        assert raw["general"]["run_name"] == "alphav_beta3_01"
        assert dry.general.run_name == "alphav_beta3_01" + DRY_RUN_SUFFIX

    def test_renaming_is_idempotent(self, raw):
        once = apply_dry_run(raw)
        twice = apply_dry_run(once)
        assert twice["general"]["run_name"] == once["general"]["run_name"]

    def test_the_flag_is_recorded_in_the_config(self, dry):
        assert dry.run.dry_run is True

    def test_a_normal_campaign_is_not_flagged(self, raw):
        assert CampaignConfig.model_validate(raw).run.dry_run is False

    def test_banner_names_what_changed(self, dry):
        banner = dry_run_banner(dry)
        assert "DRY RUN" in banner
        assert "alphav_beta3_01-dryrun" in banner
        assert "designs  : 2" in banner
        assert "--dry-run" in banner


class TestOverridesAreValidated:
    def test_the_result_is_a_normal_validated_campaign(self, raw):
        """Overrides go through the same schema, so a bad one fails the same way
        a bad file would."""
        assert isinstance(CampaignConfig.model_validate(apply_dry_run(raw)), CampaignConfig)

    def test_input_is_not_modified(self, raw):
        before = raw["general"]["run_name"]
        apply_dry_run(raw)
        assert raw["general"]["run_name"] == before
        assert len(raw["rfdiffusion3"]["specs"]) == 2


class TestSummaryRecordsIt:
    def test_summary_json_carries_the_flag(self):
        from rfd3_pipeline.gates import Stage
        from rfd3_pipeline.report import build_summary_json

        summary = build_summary_json(
            [], "tpo", "tpo_01-dryrun", 1, {Stage.BACKBONE_FILTERS: ()}, dry_run=True
        )
        assert summary["dry_run"] is True

    def test_defaults_to_false(self):
        from rfd3_pipeline.gates import Stage
        from rfd3_pipeline.report import build_summary_json

        summary = build_summary_json([], "tpo", "tpo_01", 1, {Stage.BACKBONE_FILTERS: ()})
        assert summary["dry_run"] is False
