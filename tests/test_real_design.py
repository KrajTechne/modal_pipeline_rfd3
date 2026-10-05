"""Against a real RFD3 design, not a synthetic stub.

`tests/fixtures/specs_rgd_core_0_model_0.{cif.gz,json}` is one design from the
alphaV/beta3 campaign, generated on Modal with `n_batches=1
diffusion_batch_size=2`. Every assumption `spec/design.py` was written against
documentation is pinned here against the file instead.
"""

from __future__ import annotations

import pytest
from conftest import ALPHAV_CONTIG, FIXTURES

from rfd3_pipeline.contig import parse_contig, renumber_hotspots
from rfd3_pipeline.metrics import rg_report
from rfd3_pipeline.spec import (
    assert_output_chains,
    load_design_metadata,
    renumber_hotspots_for_design,
)
from rfd3_pipeline.structure import ca_only, chain_subset, chains_present, load_structure

DESIGN_CIF = FIXTURES / "specs_rgd_core_0_model_0.cif.gz"
DESIGN_JSON = FIXTURES / "specs_rgd_core_0_model_0.json"


@pytest.fixture(scope="module")
def design():
    if not DESIGN_CIF.exists():
        pytest.skip(f"design fixture not present: {DESIGN_CIF}")
    return load_structure(DESIGN_CIF)


@pytest.fixture(scope="module")
def metadata():
    if not DESIGN_JSON.exists():
        pytest.skip(f"design fixture not present: {DESIGN_JSON}")
    return load_design_metadata(DESIGN_JSON, binder_chain="A")


class TestStructure:
    def test_gzipped_cif_loads(self, design):
        """RFD3 writes .cif.gz, which biotite's loader cannot open directly."""
        assert design.array_length() == 4046

    def test_ligand_keeps_its_input_chain(self, design):
        """The reason assert_output_chains compares polymers only."""
        assert chains_present(design) == ["A", "B", "C", "E"]
        assert assert_output_chains(design, ["A", "B", "C"]) == ["E"]

    def test_the_metal_survived_into_the_design(self, design):
        metal = chain_subset(design, "E")
        assert metal.array_length() == 1
        assert str(metal.res_name[0]) == "MN"

    def test_binder_is_exactly_the_sampled_length(self, design, metadata):
        """Unindexed motif residues are placed within the designed chain, not
        added to it -- `cleanup_guideposts` merges them before output."""
        sampled_designed = parse_contig(metadata.sampled_contig)[0]
        binder_residues = int((ca_only(design).chain_id == "A").sum())
        assert binder_residues == sampled_designed.minimum == 110

    def test_targets_came_through_unchanged(self, design):
        alpha_carbons = ca_only(design)
        assert int((alpha_carbons.chain_id == "B").sum()) == 177
        assert int((alpha_carbons.chain_id == "C").sum()) == 242


class TestMetadata:
    def test_diffused_index_map_is_top_level_and_split_correctly(self, metadata):
        assert set(metadata.binder_entries) == {("A", 77), ("A", 78), ("A", 79)}
        assert metadata.target_entries[("B", 59)] == ("B", 59)
        assert len(metadata.target_entries) == 419

    def test_motif_landed_where_only_the_map_can_say(self, metadata):
        """The scaffolded motif's position is chosen by the model; no arithmetic
        predicts it, which is why diffused_index_map is ground truth."""
        assert metadata.motif_map("A").output_residues == (64, 65, 66)
        assert metadata.motif_map("A").compact() == "A77>A64;A78>A65;A79>A66"

    def test_fixed_residues_for_mpnn_use_output_numbering(self, metadata):
        residues = metadata.motif_map("A").output_residues
        assert [f"A{r}" for r in residues] == ["A64", "A65", "A66"]

    def test_sampled_contig_is_nested_under_specification(self, metadata):
        """Not at the top level, which is where the first guess looked."""
        assert metadata.sampled_contig is not None
        assert metadata.sampled_contig.startswith("110P,/0,B1,B2")

    def test_motif_residues_lie_inside_the_binder(self, metadata, design):
        present = {
            int(r)
            for r in ca_only(chain_subset(design, "A")).res_id
        }
        assert set(metadata.motif_map("A").output_residues) <= present

    def test_rfd3_metrics_carry_the_backbone_filter_values(self, metadata):
        for name in (
            "n_clashing.interresidue_clashes_w_backbone",
            "n_clashing.interresidue_clashes_w_sidechain",
            "loop_fraction",
            "radius_of_gyration",
        ):
            assert name in metadata.metrics

    def test_nan_metrics_are_kept_as_floats_not_dropped(self, metadata):
        """`join_point_rmsd` is NaN on this design; it must not become a string."""
        assert isinstance(metadata.metrics["join_point_rmsd"], float)


class TestCrossChecks:
    def test_contig_arithmetic_agrees_with_rfd3_on_every_target_residue(self, metadata):
        """419 shared residues, and the cross-check passes on all of them."""
        out = renumber_hotspots_for_design(
            "B59,B86,C10,C225", ALPHAV_CONTIG, metadata
        )
        assert out == {
            ("B", 59): ("B", 59),
            ("B", 86): ("B", 86),
            ("C", 10): ("C", 10),
            ("C", 225): ("C", 225),
        }

    def test_the_real_sampled_contig_renumbers_identically(self, metadata):
        """It is fully expanded and type-annotated, so it has to parse too."""
        declared = renumber_hotspots("B59,C225", ALPHAV_CONTIG)
        sampled = renumber_hotspots("B59,C225", metadata.sampled_contig)
        assert declared == sampled

    def test_our_rg_tracks_rfd3s_to_within_two_percent(self, design, metadata):
        """Suggests RFD3's radius_of_gyration describes the diffused region, not
        the complex -- the plan assumed the opposite."""
        ours = rg_report(design, "A")["rg"]
        theirs = metadata.metrics["radius_of_gyration"]
        assert abs(ours - theirs) / theirs < 0.02

    def test_binder_is_compact_by_the_filter_definition(self, design):
        report = rg_report(design, "A")
        assert report["binder_length"] == 110
        assert report["rg_ratio"] < 1.1  # the campaign's multiplier
