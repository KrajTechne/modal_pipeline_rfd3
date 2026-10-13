"""The holo input's ligand chains are derived from config, not named in code.

Hardcoding either the chain id or the CCD code would be wrong in a way that is
hard to see. This campaign's input structure holds MN in *two* chains -- one in
D at residue 1004, three in E at 703-705 -- and names only `E703`. Lettering the ligand
"after the protein chains" would land on D, a different ion the campaign does
not use; and the chain id has to match what RFD3 emitted, because the structural
comparisons pair on `(chain, res_id)`.
"""

from __future__ import annotations

import copy
import pathlib

import pytest
import yaml

import app as modal_app
from rfd3_pipeline.config import CampaignConfig
from rfd3_pipeline.structure import load_structure

CAMPAIGN = pathlib.Path(
    "inputs_motif_scaffold_alphav_beta3/alphav_rgd_unindex_claude.yaml"
)
STRUCTURE = pathlib.Path(
    "inputs_motif_scaffold_alphav_beta3/alphav_beta3_fib10_clean.pdb"
)


@pytest.fixture(scope="module")
def structure():
    return load_structure(STRUCTURE)


@pytest.fixture
def raw():
    return yaml.safe_load(CAMPAIGN.read_text())


def campaign_with(raw: dict, ligand):
    payload = copy.deepcopy(raw)
    payload["rfdiffusion3"]["ligand"] = ligand
    return CampaignConfig.model_validate(payload)


class TestDerivation:
    def test_the_campaign_as_shipped_yields_chain_e_with_mn(self, raw, structure):
        entries = modal_app._ligand_entries(CampaignConfig.model_validate(raw), structure)
        assert entries == [{"type": "ligand", "id": "E", "ccd": ["MN"]}]

    def test_it_does_not_pick_the_chain_after_the_proteins(self, raw, structure):
        """D is the trap: it exists, holds MN, and is not the one named."""
        (entry,) = modal_app._ligand_entries(
            CampaignConfig.model_validate(raw), structure
        )
        assert entry["id"] != "D"

    def test_no_ligand_configured_yields_no_entries(self, raw, structure):
        assert modal_app._ligand_entries(campaign_with(raw, None), structure) == []

    def test_the_ccd_code_comes_from_the_structure(self, raw, structure):
        """Not from the config, which only names a position. D's MN sits at
        1004, nowhere near E's 703-705 -- so a derived id cannot be guessed
        from the chain letter either."""
        (entry,) = modal_app._ligand_entries(campaign_with(raw, "D1004"), structure)
        assert entry["id"] == "D"
        assert entry["ccd"] == ["MN"]

    def test_several_residues_are_refused(self, raw, structure):
        """ESMFold2 takes one ligand component, not a chain of them. Folding
        against a subset of a configured triad would score the binder on a
        complex it was not designed for, so this refuses rather than truncates.

        A valid RFD3 config can name the whole chain-E triad, which is why this
        is a refolding limit rather than a schema error."""
        with pytest.raises(ValueError, match="names 3 residues"):
            modal_app._ligand_entries(campaign_with(raw, "E703,E704,E705"), structure)

    def test_entries_carry_no_sequence(self, raw, structure):
        """`inputs.py:63` requires a sequence for every type *except* ligand."""
        for entry in modal_app._ligand_entries(CampaignConfig.model_validate(raw), structure):
            assert "sequence" not in entry
            assert entry["type"] == "ligand"


class TestRejection:
    def test_a_position_absent_from_the_structure_raises(self, raw, structure):
        """Silently emitting nothing would fold holo without the metal."""
        with pytest.raises(ValueError, match="not in the input structure"):
            modal_app._ligand_entries(campaign_with(raw, "E999"), structure)

    def test_a_chain_absent_from_the_structure_raises(self, raw, structure):
        with pytest.raises(ValueError, match="not in the input structure"):
            modal_app._ligand_entries(campaign_with(raw, "Z703"), structure)
