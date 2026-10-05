from __future__ import annotations

import copy

import pytest
import yaml
from conftest import REPO_ROOT, build, with_contig, with_specs

from rfd3_pipeline.config import CampaignConfig
from rfd3_pipeline.spec import build_rfd3_specs, dump_rfd3_specs, resolve_campaign


def resolve(raw, structure):
    return resolve_campaign(CampaignConfig.model_validate(raw), structure=structure)


@pytest.fixture
def resolved(raw, alphav_structure):
    return resolve(raw, alphav_structure)


class TestResolveTheRealCampaign:
    def test_contig_length_is_derived(self, resolved):
        assert all(spec.contig_length == (519, 539) for spec in resolved.specs)

    def test_expected_length_adds_only_the_ligand(self, resolved):
        """Measured against a real design: the designed chain comes out at
        exactly the sampled length, so unindexed motif residues sit inside it
        and only the ligand adds a residue."""
        core, extended = resolved.specs
        assert core.n_ligand_residues == 1  # ligand: "E703"
        assert core.expected_length == (520, 540)  # 519-539 + 1 Mn
        assert core.length_string == "520-540"
        # 530 observed on the real design (110 + 177 + 242 + 1) is inside it.
        low, high = core.expected_length
        assert low <= 530 <= high

    def test_unindexed_counts_are_recorded_but_not_summed(self, resolved):
        """They explain the metadata's num_residues_in, which counts the
        guideposts as separate input tokens before they are merged in."""
        core, extended = resolved.specs
        assert (core.n_unindexed, extended.n_unindexed) == (3, 7)
        # Same contig and same ligand, so the finished designs are the same size
        # despite the differing unindex counts.
        assert core.expected_length == extended.expected_length

    def test_output_chains_match_the_declaration(self, resolved):
        assert all(spec.output_chains == ["A", "B", "C"] for spec in resolved.specs)

    def test_every_hotspot_is_renumbered(self, resolved):
        hotspots = resolved.specs[0].hotspots
        assert len(hotspots) == 19
        # This contig takes both targets whole and in order, so identity.
        assert all(key == value for key, value in hotspots.items())

    def test_hotspots_grouped_for_the_interface_metric(self, resolved):
        grouped = resolved.specs[0].hotspots_by_output_chain()
        assert set(grouped) == {"B", "C"}
        assert grouped["B"] == sorted(grouped["B"])
        assert 59 in grouped["B"] and 225 in grouped["C"]

    def test_target_sequences_are_read_from_the_structure(self, resolved):
        assert set(resolved.target_sequences) == {"B", "C"}
        assert len(resolved.target_sequences["B"]) == 177
        assert len(resolved.target_sequences["C"]) == 242

    def test_seeds_and_totals(self, resolved):
        assert resolved.cross_validation_seeds == [0, 1, 2, 3, 4]
        assert resolved.config.rfdiffusion3.total_designs == 80

    def test_no_warnings_for_this_campaign(self, resolved):
        assert resolved.warnings == []

    def test_holds_no_structure_so_it_can_cross_a_process_boundary(self, resolved):
        assert not hasattr(resolved, "structure")
        payload = resolved.to_dict()
        assert yaml.safe_dump(payload)  # round-trips through plain YAML
        assert payload["specs"][0]["contig_length"] == (519, 539)

    def test_loads_the_structure_from_disk_when_not_given(self, raw):
        result = resolve_campaign(
            CampaignConfig.model_validate(raw), structure_root=REPO_ROOT
        )
        assert result.specs[0].contig_length == (519, 539)

    def test_missing_structure_raises(self, raw):
        merged = build(raw, rfdiffusion3={"path_input_structure": "nope/missing.pdb"})
        with pytest.raises(FileNotFoundError, match="input structure not found"):
            resolve_campaign(CampaignConfig.model_validate(merged))


class TestStructureConsistency:
    def test_reordered_targets_are_mapped_and_warned_about(self, raw, alphav_structure):
        """Listing C first renames it to output B. Handled, but surprising.

        seq_targets stays keyed by input chain, so anything downstream that
        resolves a chain to its MSA has to go through target_chain_map.
        """
        merged = with_contig(raw, "100-120,/0,C1-242,/0,B1-177")
        result = resolve(merged, alphav_structure)

        assert result.specs[0].target_chain_map == {"C": "B", "B": "C"}
        assert any("renames chains" in w for w in result.warnings)
        assert result.specs[0].hotspots[("C", 10)] == ("B", 10)

    def test_identity_chain_map_for_the_real_campaign(self, resolved):
        assert resolved.specs[0].target_chain_map == {"B": "B", "C": "C"}

    def test_contig_overrunning_a_chain_warns_with_counts(self, raw, alphav_structure):
        """Chain C ends at 242; asking for C1-300 silently designs 58 short."""
        merged = with_contig(raw, "100-120,/0,B1-177,/0,C1-300")
        result = resolve(merged, alphav_structure)

        assert any("58 residue(s) not in the input structure" in w for w in result.warnings)
        # The derived length reflects what exists, not what was asked for.
        assert result.specs[0].contig_length == (519, 539)

    def test_contig_component_matching_nothing_raises(self, raw, alphav_structure):
        merged = with_contig(raw, "100-120,/0,B1-177,/0,C900-950")
        with pytest.raises(ValueError, match="matches no residues"):
            resolve(merged, alphav_structure)

    def test_unindex_must_exist_in_the_input(self, raw, alphav_structure):
        def mutate(specs):
            specs[0]["unindex"] = "A777-779"
            specs[0]["select_fixed_atoms"] = {"A777-779": "ALL"}

        with pytest.raises(ValueError, match="unindex names 3 residue"):
            resolve(with_specs(raw, mutate), alphav_structure)

    def test_unindex_may_not_overlap_the_contig(self, raw, alphav_structure):
        """A motif residue is either indexed by the contig or unindexed."""

        def mutate(specs):
            specs[0]["unindex"] = "B59-61"
            specs[0]["select_fixed_atoms"] = {"B59-61": "ALL"}

        with pytest.raises(ValueError, match="appear in both the contig and unindex"):
            resolve(with_specs(raw, mutate), alphav_structure)

    def test_seq_targets_length_mismatch_is_caught(self, raw, alphav_structure):
        merged = copy.deepcopy(raw)
        merged["general"]["seq_targets"]["B"] = "ACDEF"
        with pytest.raises(ValueError, match="has 5 residues but chain B has 177"):
            resolve(merged, alphav_structure)

    def test_seq_targets_content_mismatch_is_caught(self, raw, alphav_structure):
        merged = copy.deepcopy(raw)
        original = merged["general"]["seq_targets"]["B"]
        merged["general"]["seq_targets"]["B"] = "A" + original[1:]
        with pytest.raises(ValueError, match="disagrees with the structure at position 0"):
            resolve(merged, alphav_structure)


class TestWarnings:
    def test_esmfold2_fast_warns_the_msa_is_unused(self, raw, alphav_structure):
        merged = build(raw, refolding={"model": "esmfold2-fast"})
        result = resolve(merged, alphav_structure)
        assert any("does not read MSAs" in w for w in result.warnings)

    def test_credentials_warn_about_the_secret(self, raw, alphav_structure):
        merged = build(raw, msa={"username": "u", "password": "p"})
        result = resolve(merged, alphav_structure)
        assert any("Modal secret" in w for w in result.warnings)

    def test_empty_seq_targets_warns_and_reads_them_anyway(self, raw, alphav_structure):
        merged = build(raw, general={"seq_targets": {}})
        result = resolve(merged, alphav_structure)
        assert any("seq_targets is empty" in w for w in result.warnings)
        assert len(result.target_sequences["B"]) == 177


class TestSpecEmission:
    def test_one_key_per_spec(self, resolved):
        assert sorted(build_rfd3_specs(resolved)) == ["rgd_core", "rgd_extended"]

    def test_length_is_never_emitted(self, resolved):
        """It is optional given a contig, and a second copy is a second thing
        to keep in sync."""
        assert all("length" not in entry for entry in build_rfd3_specs(resolved).values())

    def test_shared_fields_are_copied_onto_every_key(self, resolved):
        for entry in build_rfd3_specs(resolved).values():
            assert entry["dialect"] == 2
            assert entry["infer_ori_strategy"] == "hotspots"
            assert entry["is_non_loopy"] is True
            assert entry["ligand"] == "E703"
            assert entry["select_hotspots"].startswith("B59,")
            assert entry["input"].endswith("alphav_beta3_fib10_clean.pdb")

    def test_per_spec_fields_differ(self, resolved):
        specs = build_rfd3_specs(resolved)
        assert specs["rgd_core"]["unindex"] == "A77-79"
        assert specs["rgd_extended"]["unindex"] == "A75-81"
        assert specs["rgd_core"]["select_fixed_atoms"] == {"A77-79": "ALL"}
        assert specs["rgd_extended"]["select_fixed_atoms"]["A75-76"] == "BKBN"

    def test_optional_fields_omitted_when_unset(self, raw, alphav_structure):
        merged = build(raw, rfdiffusion3={"ligand": None, "is_non_loopy": None})
        entry = build_rfd3_specs(resolve(merged, alphav_structure))["rgd_core"]
        assert "ligand" not in entry
        assert "is_non_loopy" not in entry

    def test_round_trips_through_yaml(self, resolved):
        assert yaml.safe_load(dump_rfd3_specs(resolved)) == build_rfd3_specs(resolved)

    def test_emitted_yaml_uses_block_style(self, resolved):
        text = dump_rfd3_specs(resolved)
        assert "rgd_core:" in text
        assert "{" not in text
