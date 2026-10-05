from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import ALPHAV_CONTIG, make_atoms

from rfd3_pipeline.contig import (
    IndeterminateMapping,
    expected_output_chains,
    parse_residue_key,
    renumber_hotspots,
)
from rfd3_pipeline.spec import (
    assert_output_chains,
    cross_check_target_mapping,
    load_design_metadata,
    parse_design_metadata,
    renumber_hotspots_for_design,
)


def alphav_payload(**overrides):
    """A design metadata JSON shaped like the alphaV/beta3 campaign's output.

    The RGD motif lands at A34-36 in the design; target chains map to themselves
    because this contig takes both targets whole and in order.
    """
    payload = {
        "diffused_index_map": {
            "A77": "A34",
            "A78": "A35",
            "A79": "A36",
            "B59": "B59",
            "B86": "B86",
            "C10": "C10",
            "C225": "C225",
        },
        # Nested exactly as a real design's metadata nests it.
        "specification": {"extra": {"sampled_contig": "112,/0,B1-177,/0,C1-242"}},
        "metrics": {"loop_fraction": 0.25, "radius_of_gyration": 12.5},
    }
    payload.update(overrides)
    return payload


def alphav_metadata(**overrides):
    return parse_design_metadata(alphav_payload(**overrides), binder_chain="A")


class TestParseResidueKey:
    def test_single_letter_chain(self):
        assert parse_residue_key("A125") == ("A", 125)

    def test_multi_character_chain(self):
        assert parse_residue_key("AA7") == ("AA", 7)

    def test_tolerates_whitespace(self):
        assert parse_residue_key("  B59 ") == ("B", 59)

    @pytest.mark.parametrize("bad", ["", "125", "A", "A12B"])
    def test_rejects_malformed(self, bad):
        with pytest.raises(ValueError, match="cannot parse residue key"):
            parse_residue_key(bad)


class TestDesignMetadata:
    def test_splits_binder_from_target_entries_in_one_pass(self):
        metadata = alphav_metadata()
        assert set(metadata.binder_entries) == {("A", 77), ("A", 78), ("A", 79)}
        assert metadata.target_entries[("B", 59)] == ("B", 59)
        assert ("A", 77) not in metadata.target_entries

    def test_sampled_contig_read_from_specification_extra(self):
        assert alphav_metadata().sampled_contig == "112,/0,B1-177,/0,C1-242"

    def test_sampled_contig_absent_is_none(self):
        payload = alphav_payload()
        payload.pop("specification")
        assert parse_design_metadata(payload, "A").sampled_contig is None

    def test_numeric_metrics_are_kept(self):
        assert alphav_metadata().metrics["loop_fraction"] == 0.25

    def test_raises_when_the_map_is_absent(self):
        with pytest.raises(KeyError, match="diffused_index_map"):
            parse_design_metadata({"something_else": {}}, "A")

    def test_raises_when_the_map_is_not_an_object(self):
        with pytest.raises(ValueError, match="should be an object"):
            parse_design_metadata({"diffused_index_map": []}, "A")


class TestMotifMap:
    def test_output_residues_are_sorted(self):
        assert alphav_metadata().motif_map("A").output_residues == (34, 35, 36)

    def test_compact_records_the_translation(self):
        assert alphav_metadata().motif_map("A").compact() == "A77>A34;A78>A35;A79>A36"

    def test_raises_when_nothing_lands_on_the_binder(self):
        metadata = parse_design_metadata(alphav_payload(), binder_chain="Z")
        with pytest.raises(ValueError, match="no entries landing on binder chain"):
            metadata.motif_map("Z")


class TestCrossCheck:
    def test_agreement_returns_the_comparison_count(self):
        derived = {("B", 59): ("B", 59), ("C", 10): ("C", 10)}
        assert cross_check_target_mapping(derived, derived) == 2

    def test_no_overlap_compares_nothing(self):
        assert cross_check_target_mapping({("B", 1): ("B", 1)}, {("C", 1): ("C", 1)}) == 0

    def test_off_by_one_is_caught(self):
        """The failure this whole cross-check exists to catch."""
        derived = {("B", 59): ("B", 59), ("B", 86): ("B", 86)}
        reference = {("B", 59): ("B", 60), ("B", 86): ("B", 87)}
        with pytest.raises(ValueError, match="disagrees with diffused_index_map"):
            cross_check_target_mapping(derived, reference)

    def test_error_names_the_offending_residues(self):
        with pytest.raises(ValueError, match=r"B59: contig says B59, RFD3 says B60"):
            cross_check_target_mapping({("B", 59): ("B", 59)}, {("B", 59): ("B", 60)})


class TestRenumberHotspots:
    def test_identity_for_the_campaign_contig(self):
        assert renumber_hotspots("B59,C225", ALPHAV_CONTIG) == {
            ("B", 59): ("B", 59),
            ("C", 225): ("C", 225),
        }

    def test_offset_contig_shifts_hotspots(self):
        assert renumber_hotspots("B77", "100-120,/0,B5-181") == {("B", 77): ("B", 73)}

    def test_reordered_contig_changes_the_chain_letter(self):
        assert renumber_hotspots("C10", "100-120,/0,C1-242,/0,B1-177") == {
            ("C", 10): ("B", 10)
        }

    def test_accepts_an_explicit_residue_list(self):
        assert renumber_hotspots([("B", 59)], ALPHAV_CONTIG) == {("B", 59): ("B", 59)}

    def test_empty_hotspots(self):
        assert renumber_hotspots("", ALPHAV_CONTIG) == {}

    def test_raises_when_a_hotspot_is_outside_the_contig(self):
        with pytest.raises(ValueError, match="not taken from the input"):
            renumber_hotspots("B59,D999", ALPHAV_CONTIG)


class TestRenumberForDesign:
    def test_cross_checks_against_the_design(self):
        out = renumber_hotspots_for_design("B59,C10", ALPHAV_CONTIG, alphav_metadata())
        assert out == {("B", 59): ("B", 59), ("C", 10): ("C", 10)}

    def test_disagreement_with_the_design_raises(self):
        wrong = alphav_metadata(
            diffused_index_map={"A77": "A34", "B59": "B60", "C10": "C11"}
        )
        with pytest.raises(ValueError, match="disagrees with diffused_index_map"):
            renumber_hotspots_for_design("B59", ALPHAV_CONTIG, wrong)

    def test_sampled_contig_overrides_the_declared_one(self):
        """A variable designed region is indeterminate until RFD3 samples it."""
        metadata = parse_design_metadata(
            {
                "diffused_index_map": {"B30": "A20"},
                "specification": {"extra": {"sampled_contig": "B1-5,14,B30-32"}},
            },
            binder_chain="Z",  # nothing lands on the binder in this stub
        )
        out = renumber_hotspots_for_design("B30", "B1-5,10-20,B30-32", metadata)
        assert out == {("B", 30): ("A", 20)}  # 5 + 14 fixed residues precede it

    def test_without_the_sampled_contig_it_refuses_to_guess(self):
        with pytest.raises(IndeterminateMapping):
            renumber_hotspots("B30", "B1-5,10-20,B30-32")


class TestOutputChains:
    def test_expected_chains_from_the_campaign_contig(self):
        assert expected_output_chains(ALPHAV_CONTIG) == ["A", "B", "C"]

    def test_assert_passes_on_a_matching_design(self):
        design = (
            make_atoms("A", [1, 2], np.zeros((2, 3)))
            + make_atoms("B", [1, 2], np.zeros((2, 3)))
            + make_atoms("C", [1, 2], np.zeros((2, 3)))
        )
        assert assert_output_chains(design, ["A", "B", "C"]) == []

    def test_ligand_chains_are_tolerated_and_reported(self):
        """A real design with `ligand: "E703"` comes out as A/B/C/E.

        The ligand keeps its *input* chain id rather than being renumbered with
        the polymers, so comparing every chain against the declared list would
        fail on every design that uses one.
        """
        metal = make_atoms("E", [703], np.zeros((1, 3)), atom_name="MN")
        metal.res_name = np.array(["MN"])
        metal.element = np.array(["MN"])
        design = (
            make_atoms("A", [1, 2], np.zeros((2, 3)))
            + make_atoms("B", [1, 2], np.zeros((2, 3)))
            + make_atoms("C", [1, 2], np.zeros((2, 3)))
            + metal
        )
        assert assert_output_chains(design, ["A", "B", "C"]) == ["E"]

    def test_a_missing_polymer_chain_still_fails_with_a_ligand_present(self):
        metal = make_atoms("E", [703], np.zeros((1, 3)), atom_name="MN")
        metal.res_name = np.array(["MN"])
        design = make_atoms("A", [1], np.zeros((1, 3))) + metal
        with pytest.raises(ValueError, match="polymer chains"):
            assert_output_chains(design, ["A", "B", "C"])

    def test_assert_catches_a_missing_chain(self):
        design = make_atoms("A", [1], np.zeros((1, 3))) + make_atoms(
            "B", [1], np.zeros((1, 3))
        )
        with pytest.raises(ValueError, match="declares"):
            assert_output_chains(design, ["A", "B", "C"])


class TestLoadDesignMetadata:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "design.json"
        path.write_text(json.dumps(alphav_payload()))
        metadata = load_design_metadata(path, binder_chain="A")
        assert metadata.motif_map("A").output_residues == (34, 35, 36)

    def test_rejects_a_json_array(self, tmp_path):
        path = tmp_path / "design.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="expected a JSON object"):
            load_design_metadata(path, binder_chain="A")
