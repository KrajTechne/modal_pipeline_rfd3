"""Cross-validation must use a model other than the refolding model.

The stage exists to measure agreement between *independent* models. The same
model on both sides would run a second full folding pass and report a model
agreeing with itself -- no error, and ranks that read as corroboration.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from rfd3_pipeline.config import CampaignConfig


@pytest.fixture
def raw():
    import pathlib

    path = pathlib.Path("inputs_motif_scaffold_alphav_beta3/alphav_rgd_unindex_claude.yaml")
    return yaml.safe_load(path.read_text())


def set_model(block: dict, model: str) -> None:
    """Point a folding block at `model`, dropping fields it must not carry.

    The two fields have different rules, which is why they are dropped
    separately. `weights` is accepted by any model shipping several checkpoints
    -- both AF3 integrations -- while `variant` is accepted only by `af3_jax`,
    since the fork names its weights directory outright instead. A separate
    validator rejects each, and those run first, so without this the mutations
    below would fail for an unrelated reason rather than the one under test.
    """
    block["model"] = model
    if model not in ("af3_jax", "af3_fork"):
        block.pop("weights", None)
    if model != "af3_jax":
        block.pop("variant", None)


class TestDistinctModels:
    def test_the_campaign_as_shipped_is_valid(self, raw):
        campaign = CampaignConfig.model_validate(raw)
        assert campaign.refolding.model.value == "esmfold2"
        assert campaign.cross_validation.model.value == "af3_fork"

    def test_the_same_model_on_both_sides_is_rejected(self, raw):
        payload = copy.deepcopy(raw)
        set_model(payload["cross_validation"], payload["refolding"]["model"])
        with pytest.raises(ValueError, match="cross-validation must use a different model"):
            CampaignConfig.model_validate(payload)

    def test_the_message_names_the_duplicated_model(self, raw):
        payload = copy.deepcopy(raw)
        set_model(payload["refolding"], "boltz2")
        set_model(payload["cross_validation"], "boltz2")
        with pytest.raises(ValueError, match="both 'boltz2'"):
            CampaignConfig.model_validate(payload)

    def test_a_different_pairing_still_validates(self, raw):
        """The rule is "not equal", not a whitelist of pairings. `boltz2`
        rather than `esmfold2-fast`, since a refolding model has to read MSAs."""
        payload = copy.deepcopy(raw)
        set_model(payload["refolding"], "boltz2")
        campaign = CampaignConfig.model_validate(payload)
        assert campaign.refolding.model.value == "boltz2"
        assert campaign.cross_validation.model.value == "af3_fork"

    def test_distinct_models_are_accepted_either_way_round(self, raw):
        """Which side runs which model is not this validator's business."""
        payload = copy.deepcopy(raw)
        set_model(payload["refolding"], "boltz2")
        campaign = CampaignConfig.model_validate(payload)
        assert campaign.refolding.model != campaign.cross_validation.model
