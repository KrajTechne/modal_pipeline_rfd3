from __future__ import annotations

import copy

import pytest
from conftest import build
from pydantic import ValidationError

from rfd3_pipeline.config import (
    MODEL_CAPABILITIES,
    CampaignConfig,
    FoldingModel,
    KitCard,
    MpnnType,
    SeedStyle,
    Weights,
    folding_variant,
    seed_sequence,
)


class TestCampaignParses:
    def test_the_real_campaign_validates(self, raw):
        config = CampaignConfig.model_validate(raw)
        assert config.general.chain_binder == "A"
        assert config.general.chain_targets == ["B", "C"]
        assert config.rfdiffusion3.ligand == "E703"
        assert config.cross_validation.model is FoldingModel.AF3_JAX

    def test_design_counts(self, raw):
        config = CampaignConfig.model_validate(raw)
        assert config.rfdiffusion3.designs_per_spec == 40
        assert config.rfdiffusion3.total_designs == 80

    def test_unknown_keys_are_rejected(self, raw):
        with pytest.raises(ValidationError, match="Extra inputs"):
            CampaignConfig.model_validate(build(raw, mpnn={"num_seqz": 8}))


class TestGeneral:
    def test_comma_string_chain_targets(self, raw):
        config = CampaignConfig.model_validate(build(raw, general={"chain_targets": "B,C"}))
        assert config.general.chain_targets == ["B", "C"]

    def test_all_chains_is_binder_then_targets(self, raw):
        assert CampaignConfig.model_validate(raw).general.all_chains == ["A", "B", "C"]

    def test_binder_cannot_be_a_target(self, raw):
        with pytest.raises(ValidationError, match="also appears in chain_targets"):
            CampaignConfig.model_validate(build(raw, general={"chain_targets": ["A", "B"]}))

    def test_duplicate_targets_rejected(self, raw):
        with pytest.raises(ValidationError, match="duplicate target chains"):
            CampaignConfig.model_validate(build(raw, general={"chain_targets": ["B", "B"]}))

    def test_comma_joined_seq_targets(self, raw):
        """The original YAML's form, paired positionally with chain_targets."""
        sequences = raw["general"]["seq_targets"]
        config = CampaignConfig.model_validate(
            build(
                raw,
                general={
                    "chain_targets": "B,C",
                    "seq_targets": f"{sequences['B']},{sequences['C']}",
                },
            )
        )
        assert set(config.general.seq_targets) == {"B", "C"}

    def test_seq_targets_must_cover_the_declared_chains(self, raw):
        with pytest.raises(ValidationError, match="seq_targets describes chains"):
            CampaignConfig.model_validate(build(raw, general={"seq_targets": {"B": "ACDE"}}))


class TestRfd3:
    def test_length_may_not_be_configured(self, raw):
        with pytest.raises(ValidationError, match="derived from the contig"):
            CampaignConfig.model_validate(build(raw, rfdiffusion3={"length": "519-539"}))

    def test_explicit_null_length_is_fine(self, raw):
        assert (
            CampaignConfig.model_validate(
                build(raw, rfdiffusion3={"length": None})
            ).rfdiffusion3.length
            is None
        )

    def test_backslash_paths_are_normalised(self, raw):
        config = CampaignConfig.model_validate(
            build(raw, rfdiffusion3={"path_input_structure": r"inputs\thing.pdb"})
        )
        assert config.rfdiffusion3.path_input_structure == "inputs/thing.pdb"

    def test_spec_names_must_be_unique(self, raw):
        specs = [dict(raw["rfdiffusion3"]["specs"][0]) for _ in range(2)]
        with pytest.raises(ValidationError, match="spec names must be unique"):
            CampaignConfig.model_validate(build(raw, rfdiffusion3={"specs": specs}))

    def test_hotspot_strategy_needs_hotspots(self, raw):
        with pytest.raises(ValidationError, match="no hotspots are given"):
            CampaignConfig.model_validate(build(raw, rfdiffusion3={"hotspots": ""}))

    def test_ligand_range_syntax_rejected(self, raw):
        """RFD3 splits on '-', so 'E703-705' would read as a range pattern."""
        with pytest.raises(ValidationError, match="parses as a range"):
            CampaignConfig.model_validate(build(raw, rfdiffusion3={"ligand": "E703-705"}))

    def test_several_ligands_listed_with_commas(self, raw):
        config = CampaignConfig.model_validate(
            build(raw, rfdiffusion3={"ligand": "E703,E704"})
        )
        assert config.rfdiffusion3.ligand == "E703,E704"

    def test_sampler_defaults_are_rfd3_stock(self, raw):
        rfd3 = CampaignConfig.model_validate(raw).rfdiffusion3
        assert (rfd3.step_scale, rfd3.gamma_0, rfd3.num_timesteps) == (1.5, 0.6, 200)


class TestTextOnlyValidation:
    """Checks that need no structure must not wait for one to be loaded."""

    def test_malformed_contig_fails_at_config_time(self, raw):
        specs = copy.deepcopy(raw["rfdiffusion3"]["specs"])
        specs[0]["contig"] = "100-120,??,B1-177"
        with pytest.raises(ValidationError, match="cannot parse contig component"):
            CampaignConfig.model_validate(build(raw, rfdiffusion3={"specs": specs}))

    def test_malformed_hotspots_fail_at_config_time(self, raw):
        with pytest.raises(ValidationError, match="cannot parse residue selection"):
            CampaignConfig.model_validate(build(raw, rfdiffusion3={"hotspots": "B59,??"}))

    def test_contig_chain_count_checked_at_config_time(self, raw):
        specs = copy.deepcopy(raw["rfdiffusion3"]["specs"])
        for spec in specs:
            spec["contig"] = "100-120,/0,B1-177"
        with pytest.raises(ValidationError, match="produces output chains"):
            CampaignConfig.model_validate(build(raw, rfdiffusion3={"specs": specs}))

    def test_fixed_atoms_without_unindex_rejected(self, raw):
        specs = copy.deepcopy(raw["rfdiffusion3"]["specs"])
        specs[0].pop("unindex")
        with pytest.raises(ValidationError, match="without unindex"):
            CampaignConfig.model_validate(build(raw, rfdiffusion3={"specs": specs}))

    def test_fixed_atoms_must_lie_inside_unindex(self, raw):
        specs = copy.deepcopy(raw["rfdiffusion3"]["specs"])
        specs[0]["select_fixed_atoms"] = {"A77-79": "ALL", "A90": "BKBN"}
        with pytest.raises(ValidationError, match="not unindexed"):
            CampaignConfig.model_validate(build(raw, rfdiffusion3={"specs": specs}))


class TestFoldingSteps:
    def test_kit_card_derived_from_gpu(self, raw):
        assert CampaignConfig.model_validate(raw).refolding.kit_config is KitCard.A100

    def test_gpu_without_a_kit_card_rejected(self, raw):
        """A mode that cannot engage exits 3 rather than falling back."""
        with pytest.raises(ValidationError, match="has no kit card"):
            CampaignConfig.model_validate(build(raw, refolding={"gpu": "L40S"}))

    def test_sharding_requires_big_mode(self, raw):
        with pytest.raises(ValidationError, match="requires mode 'big'"):
            CampaignConfig.model_validate(
                build(raw, refolding={"n_gpu_shard": 2, "mode": "fast"})
            )

    def test_sharding_allowed_under_big(self, raw):
        config = CampaignConfig.model_validate(
            build(raw, refolding={"n_gpu_shard": 2, "mode": "big"})
        )
        assert config.refolding.n_gpu_shard == 2

    def test_odd_shard_counts_rejected(self, raw):
        with pytest.raises(ValidationError, match="n_gpu_shard must be one of"):
            CampaignConfig.model_validate(
                build(raw, refolding={"n_gpu_shard": 3, "mode": "big"})
            )

    def test_hyphenated_cross_validation_key_accepted(self, raw):
        legacy = {k: v for k, v in raw.items() if k != "cross_validation"}
        legacy["cross-validation"] = raw["cross_validation"]
        assert CampaignConfig.model_validate(legacy).cross_validation.num_seeds == 5

    def test_both_spellings_rejected(self, raw):
        both = dict(raw)
        both["cross-validation"] = raw["cross_validation"]
        with pytest.raises(ValidationError, match="keep only 'cross_validation'"):
            CampaignConfig.model_validate(both)


class TestModelCapabilities:
    """Per-model facts live in one table, so no model can fall through."""

    @pytest.mark.parametrize("model", list(FoldingModel))
    def test_every_model_has_a_row(self, model):
        assert model in MODEL_CAPABILITIES

    def test_variant_may_not_be_set_where_it_is_derived(self, raw):
        with pytest.raises(ValidationError, match="derived per step"):
            CampaignConfig.model_validate(build(raw, refolding={"variant": "full_msa"}))

    def test_variant_rejected_where_the_model_takes_none(self, raw):
        with pytest.raises(ValidationError, match="takes no variant"):
            CampaignConfig.model_validate(
                build(raw, refolding={"model": "boltz2", "variant": "x"})
            )

    def test_templates_rejected_for_models_without_support(self, raw):
        for model in ("esmfold2", "esmfold2-fast", "boltz2"):
            with pytest.raises(ValidationError, match="does not support templates"):
                CampaignConfig.model_validate(
                    build(raw, refolding={"model": model, "target_template": True})
                )

    def test_af3_defaults_variant_and_weights(self, raw):
        config = CampaignConfig.model_validate(raw)
        assert config.cross_validation.variant == "p2"
        assert config.cross_validation.weights is Weights.OPENFOLD3

    def test_weights_rejected_for_single_checkpoint_models(self, raw):
        """Previously only checked on whichever branch nothing else claimed."""
        for model in ("esmfold2", "boltz2"):
            with pytest.raises(ValidationError, match="meaningful only for models"):
                CampaignConfig.model_validate(
                    build(raw, refolding={"model": model, "weights": "openfold3"})
                )

    def test_msa_consumption_read_from_the_table(self, raw):
        assert CampaignConfig.model_validate(raw).refolding.consumes_msa is True
        fast = CampaignConfig.model_validate(build(raw, refolding={"model": "esmfold2-fast"}))
        assert fast.refolding.consumes_msa is False

    def test_seed_style_read_from_the_table(self, raw):
        config = CampaignConfig.model_validate(raw)
        assert config.refolding.seed_style is SeedStyle.LIST
        assert config.cross_validation.seed_style is SeedStyle.COUNT

    @pytest.mark.parametrize(
        ("model", "msa", "expected"),
        [
            (FoldingModel.ESMFOLD2, False, "full_nomsa"),
            (FoldingModel.ESMFOLD2, True, "full_msa"),
            (FoldingModel.ESMFOLD2_FAST, False, "fast"),
            (FoldingModel.ESMFOLD2_FAST, True, "fast"),
            (FoldingModel.BOLTZ2, True, None),
            (FoldingModel.AF3_JAX, True, "p2"),
        ],
    )
    def test_variant_derivation(self, model, msa, expected):
        assert folding_variant(model, msa_present=msa) == expected


class TestSeeds:
    def test_sequence_from_base(self):
        assert seed_sequence(0, 5) == [0, 1, 2, 3, 4]
        assert seed_sequence(100, 3) == [100, 101, 102]

    def test_campaign_seeds(self, raw):
        assert CampaignConfig.model_validate(raw).cross_validation_seeds == [0, 1, 2, 3, 4]

    def test_at_least_one(self):
        with pytest.raises(ValueError, match="at least one seed"):
            seed_sequence(0, 0)


class TestMsa:
    def test_defaults_need_no_secret(self, raw):
        assert CampaignConfig.model_validate(raw).msa.needs_secret is False

    def test_credentials_flag_a_secret(self, raw):
        config = CampaignConfig.model_validate(
            build(raw, msa={"username": "u", "password": "p"})
        )
        assert config.msa.needs_secret is True

    def test_partial_credentials_rejected(self, raw):
        with pytest.raises(ValidationError, match="must be given together"):
            CampaignConfig.model_validate(build(raw, msa={"username": "u"}))

    def test_two_auth_methods_rejected(self, raw):
        with pytest.raises(ValidationError, match="not both"):
            CampaignConfig.model_validate(
                build(raw, msa={"username": "u", "password": "p", "auth_header": "k"})
            )

    def test_unknown_pairing_strategy(self, raw):
        with pytest.raises(ValidationError, match="pairing_strategy"):
            CampaignConfig.model_validate(build(raw, msa={"pairing_strategy": "eager"}))


class TestIpsaeAndMpnn:
    def test_cutoffs_are_ten_not_fifteen(self, raw):
        config = CampaignConfig.model_validate(raw)
        assert (config.ipsae.pae_cutoff, config.ipsae.dist_cutoff) == (10.0, 10.0)

    def test_mpnn_defaults_to_soluble(self, raw):
        assert CampaignConfig.model_validate(raw).mpnn.mpnn_type is MpnnType.SOLUBLE

    def test_unknown_mpnn_type_rejected(self, raw):
        with pytest.raises(ValidationError):
            CampaignConfig.model_validate(build(raw, mpnn={"mpnn_type": "magic"}))
