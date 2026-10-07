"""`_mpnn_records` against MPNN's real FASTA shape.

The header and segment layout here are copied from actual `soluble_mpnn` output
(`experiments/mpnn_image.py --stage run_one`, 2026-10-07), shortened. The two
properties worth protecting are that the first record is the *input* sequence
rather than a design, and that the binder is the first `:`-separated segment.
Getting either wrong produces a full set of plausible records that are off by
one design or carry a target's sequence as the binder's.
"""

from __future__ import annotations

import pytest

import app as modal_app

# Three chains, as the alphaV/beta3 campaign emits: binder A, targets B and C.
BINDER_NATIVE = "ASSHVPLASA"
BINDER_1 = "GMGIVPVESL"
BINDER_2 = "APGIEPVYSE"
TARGET_B = "QWFGASVRSK"
TARGET_C = "VDIYYLMDLS"

FASTA = f"""\
>input, T=0.1, seed=7, num_res=521, num_ligand_res=521, use_ligand_context=True, batch_size=2
{BINDER_NATIVE}:{TARGET_B}:{TARGET_C}
>input, id=1, T=0.1, seed=7, overall_confidence=0.3873, ligand_confidence=0.3412, seq_rec=0.3493
{BINDER_1}:{TARGET_B}:{TARGET_C}
>input, id=2, T=0.1, seed=7, overall_confidence=0.3866, ligand_confidence=0.3401, seq_rec=0.3762
{BINDER_2}:{TARGET_B}:{TARGET_C}
"""


class _Mpnn:
    """Stand-in for `MpnnConfig`, which only supplies a temperature fallback."""

    mpnn_sampling_temp = 0.1


BACKBONE = {
    "design_id": "rgd_core_0_model_0",
    "campaign": "motif_scaffold_alphav_beta3",
    "run_name": "alphav_beta3_01",
    "bb_pass": True,
    "bb_binder_length": 10,
    "motif_map": "A77>A49;A78>A50;A79>A51",
    "mpnn_fixed_residues": "A49 A50 A51",
}


@pytest.fixture
def records():
    return modal_app._mpnn_records(FASTA, "rgd_core_0_model_0", BACKBONE, _Mpnn())


class TestRecordSet:
    def test_the_native_sequence_is_not_a_design(self, records):
        """MPNN's first FASTA record is the input, so two designs means three
        records in the file."""
        assert len(records) == 2
        assert all(r["sequence"] != BINDER_NATIVE for r in records)

    def test_designs_are_numbered_from_one(self, records):
        assert [r["mpnn_index"] for r in records] == [1, 2]

    def test_seq_id_is_the_design_id_plus_the_index(self, records):
        assert [r["seq_id"] for r in records] == [
            "rgd_core_0_model_0_seq1",
            "rgd_core_0_model_0_seq2",
        ]

    def test_sequences_are_in_file_order(self, records):
        assert [r["sequence"] for r in records] == [BINDER_1, BINDER_2]


class TestBinderExtraction:
    def test_only_the_binder_segment_is_kept(self, records):
        """The targets are context, not product; a record holding a target's
        sequence would be refolded as though it were the binder."""
        for record in records:
            assert ":" not in record["sequence"]
            assert record["sequence"] not in (TARGET_B, TARGET_C)

    def test_seq_length_matches_the_binder(self, records):
        for record in records:
            assert record["seq_length"] == len(record["sequence"]) == len(BINDER_1)


class TestHeaderMetadata:
    def test_temperature_and_seed_come_from_the_header(self, records):
        assert all(r["mpnn_temp"] == 0.1 for r in records)
        assert all(r["mpnn_seed"] == 7 for r in records)

    def test_confidences_are_floats(self, records):
        assert records[0]["mpnn_overall_confidence"] == pytest.approx(0.3873)
        assert records[0]["mpnn_ligand_confidence"] == pytest.approx(0.3412)
        assert records[0]["mpnn_seq_rec"] == pytest.approx(0.3493)
        assert records[1]["mpnn_overall_confidence"] == pytest.approx(0.3866)

    def test_missing_metadata_is_none_not_an_error(self):
        """A header without the confidence fields must still yield a record."""
        bare = f">input, T=0.1, seed=1\n{BINDER_NATIVE}\n>input, id=1\n{BINDER_1}\n"
        (record,) = modal_app._mpnn_records(bare, "d", BACKBONE, _Mpnn())
        assert record["mpnn_overall_confidence"] is None
        assert record["mpnn_seq_rec"] is None
        assert record["mpnn_seed"] is None
        # Falls back to the configured temperature rather than None.
        assert record["mpnn_temp"] == 0.1

    def test_unparseable_numbers_do_not_raise(self):
        bad = f">x, T=hot, seed=n/a\n{BINDER_NATIVE}\n>x, id=1, seq_rec=?\n{BINDER_1}\n"
        (record,) = modal_app._mpnn_records(bad, "d", BACKBONE, _Mpnn())
        assert record["mpnn_seed"] is None
        assert record["mpnn_seq_rec"] is None
        assert record["mpnn_temp"] == 0.1


class TestBackboneCarriedForward:
    def test_bb_columns_survive_onto_the_sequence_row(self, records):
        """`summary.csv` is one row per sequence, so a sequence has to keep its
        backbone's metrics or they are lost from the report."""
        for record in records:
            assert record["bb_pass"] is True
            assert record["bb_binder_length"] == 10
            assert record["motif_map"] == "A77>A49;A78>A50;A79>A51"
            assert record["campaign"] == "motif_scaffold_alphav_beta3"

    def test_stage_advances_to_mpnn(self, records):
        from rfd3_pipeline.gates import Stage

        assert all(r["stage_reached"] == Stage.MPNN for r in records)

    def test_later_stage_columns_are_stripped(self):
        """Re-running MPNN over shards that already carry refolding columns must
        not copy them onto the new rows."""
        stale = dict(BACKBONE, apo_rmsd=1.0, holo_ipsae=0.5, cv_mean=2.0, rank_overall=3)
        records = modal_app._mpnn_records(FASTA, "d", stale, _Mpnn())
        for record in records:
            assert not any(
                key.startswith(("apo_", "holo_", "cv_", "rank_", "cluster_"))
                for key in record
            )


class TestDegenerateInput:
    def test_only_a_native_record_yields_nothing(self):
        text = f">input, T=0.1\n{BINDER_NATIVE}\n"
        assert modal_app._mpnn_records(text, "d", BACKBONE, _Mpnn()) == []

    def test_empty_text_yields_nothing(self):
        assert modal_app._mpnn_records("", "d", BACKBONE, _Mpnn()) == []

    def test_blank_lines_between_records_are_ignored(self):
        text = f">input, T=0.1\n{BINDER_NATIVE}\n\n>input, id=1\n{BINDER_1}\n\n"
        (record,) = modal_app._mpnn_records(text, "d", BACKBONE, _Mpnn())
        assert record["sequence"] == BINDER_1
