"""Reading a campaign has to mean the same thing every time it is read.

These tables were scratch scripts first, and two readings of one run already
disagreed: a bucket median came out 0.513 in a throwaway script and 0.511 from
the module, because the script took `values[len(values) // 2]` and the module
takes a nearest-rank quantile. The number reached `Notes.md` before anyone
noticed. So the conventions are pinned here rather than left to whoever is
holding the keyboard.
"""

from __future__ import annotations

import json

import pytest

from rfd3_pipeline import analysis

# Four cross-validated sequences across two designs, plus one that stopped at
# apo -- enough shape for every function here and small enough to read.
RECORDS = [
    {
        "seq_id": "d0_seq1", "design_id": "d0", "spec_name": "s",
        "apo_plddt_binder": "72.0", "apo_pass": "True",
        "holo_plddt_binder": "85.0", "holo_iptm": "0.91", "holo_ipsae_min": "0.60",
        "holo_rmsd_binder_on_target": "1.8", "holo_rmsd_motif_on_target": "0.7",
        "holo_pass": "True", "holo_iptm_pass": "True",
        "holo_rmsd_binder_on_target_pass": "True",
        "cv_plddt_binder": "92.0", "cv_iptm_mean": "0.83",
        "cv_ipsae_min": "0.58", "cv_rmsd_binder_on_target": "2.2",
        "cv_rmsd_motif_on_target": "0.8",
    },
    {
        "seq_id": "d0_seq2", "design_id": "d0", "spec_name": "s",
        "apo_plddt_binder": "75.0", "apo_pass": "True",
        "holo_plddt_binder": "82.0", "holo_iptm": "0.80", "holo_ipsae_min": "0.0",
        "holo_rmsd_binder_on_target": "4.0", "holo_rmsd_motif_on_target": "1.0",
        "holo_pass": "True", "holo_iptm_pass": "True",
        "holo_rmsd_binder_on_target_pass": "True",
        "cv_plddt_binder": "80.0", "cv_iptm_mean": "0.66",
        "cv_ipsae_min": "0.0", "cv_rmsd_binder_on_target": "3.0",
        # No motif measured: must not be read as a motif failure.
        "cv_rmsd_motif_on_target": "",
    },
    {
        "seq_id": "d1_seq1", "design_id": "d1", "spec_name": "s",
        "apo_plddt_binder": "71.0", "apo_pass": "True",
        "holo_plddt_binder": "79.0", "holo_iptm": "0.75", "holo_ipsae_min": "0.0",
        "holo_rmsd_binder_on_target": "22.0", "holo_rmsd_motif_on_target": "8.0",
        "holo_pass": "False", "holo_iptm_pass": "True",
        "holo_rmsd_binder_on_target_pass": "False",
        "cv_plddt_binder": "70.0", "cv_iptm_mean": "0.60",
        "cv_ipsae_min": "0.0", "cv_rmsd_binder_on_target": "25.0",
        "cv_rmsd_motif_on_target": "9.0",
    },
    {
        "seq_id": "d1_seq2", "design_id": "d1", "spec_name": "s",
        "apo_plddt_binder": "60.0", "apo_pass": "False",
        "apo_plddt_binder_pass": "False",
    },
]


class TestCoercion:
    def test_absent_is_not_zero(self):
        """ipSAE 0.0 is a verdict; an empty cell is not measured at all."""
        assert analysis.value(RECORDS[1], "cv_ipsae_min") == 0.0
        assert analysis.value(RECORDS[1], "cv_rmsd_motif_on_target") is None
        assert analysis.value(RECORDS[0], "nonexistent") is None

    def test_verdicts_round_trip_through_the_csv(self):
        assert analysis.truth(RECORDS[0], "holo_pass") is True
        assert analysis.truth(RECORDS[2], "holo_pass") is False
        assert analysis.truth(RECORDS[3], "holo_pass") is None


class TestQuantile:
    def test_nearest_rank_returns_a_value_some_design_had(self):
        values = [1.0, 2.0, 3.0, 4.0]
        assert analysis.quantile(values, 0.5) in values
        assert analysis.quantile(values, 0.0) == 1.0
        assert analysis.quantile(values, 1.0) == 4.0

    def test_empty_is_none_not_an_error(self):
        assert analysis.quantile([], 0.5) is None


class TestColumns:
    def test_cv_iptm_falls_back_to_the_seed_mean(self):
        """The bug this exists for: `cv_iptm` is absent, `cv_iptm_mean` is not."""
        column, qualifier = analysis.column_for(RECORDS, "cv", "iptm")
        assert column == "cv_iptm_mean"
        assert qualifier == " (mean)"

    def test_a_direct_column_is_not_relabelled(self):
        assert analysis.column_for(RECORDS, "holo", "iptm") == ("holo_iptm", "")

    def test_distribution_labels_the_averaged_column(self):
        rows = analysis.distribution(RECORDS, "cv")
        assert "ipTM (mean)" in {row["metric"] for row in rows}

    def test_distribution_omits_metrics_a_stage_never_recorded(self):
        """Apo records no ipTM -- the column is gone, not zero."""
        rows = analysis.distribution(RECORDS, "apo")
        assert "ipTM" not in {row["metric"] for row in rows}


class TestGateTerms:
    def test_sole_reason_ignores_designs_that_failed_several_terms(self):
        work = analysis.gate_terms(RECORDS, "holo")
        assert (work["n"], work["passed"]) == (3, 2)
        assert work["rejected"]["rmsd_binder_on_target"] == 1
        assert work["sole"] == {"rmsd_binder_on_target": 1}

    def test_a_term_that_never_fires_is_reported_as_zero_not_omitted(self):
        """The whole point: a gate term doing no work must be visible."""
        assert analysis.gate_terms(RECORDS, "holo")["rejected"]["iptm"] == 0

    def test_an_ungated_stage_has_no_terms(self):
        assert analysis.gate_terms(RECORDS, "cv")["rejected"] == {}


class TestBucket:
    def test_zeros_are_counted_beside_the_median(self):
        rows = analysis.bucket(
            RECORDS, "holo_ipsae_min", "holo_rmsd_binder_on_target", edges=(5.0, 15.0)
        )
        first = rows[0]
        assert (first["band"], first["n"], first["zeros"]) == ("0-5", 2, 1)
        assert rows[-1]["band"] == ">15"

    def test_empty_bands_are_dropped(self):
        rows = analysis.bucket(
            RECORDS, "holo_ipsae_min", "holo_rmsd_binder_on_target", edges=(5.0, 6.0)
        )
        assert "5-6" not in {row["band"] for row in rows}


class TestAgreed:
    def test_an_unmeasured_motif_does_not_reject(self):
        """d0_seq2 has no cv motif; placement alone decides it."""
        assert {r["seq_id"] for r in analysis.agreed(RECORDS)} == {"d0_seq1", "d0_seq2"}

    def test_defaults_match_the_holo_gate(self):
        """So the question is whether the second model agrees, not some other bar."""
        from rfd3_pipeline.config import FoldingModel, RefoldingConfig

        defaults = RefoldingConfig(model=FoldingModel.ESMFOLD2)
        assert defaults.threshold_rmsd_binder_on_target == 5.0
        assert defaults.threshold_rmsd_motif == 2.0

    def test_yield_is_reported_per_design(self):
        assert analysis.by_design(analysis.agreed(RECORDS)) == [
            {"design_id": "d0", "n_sequences": 2}
        ]


class TestReached:
    def test_a_row_that_stopped_at_apo_did_not_reach_cv(self):
        assert analysis.reached(RECORDS[3], "apo") is True
        assert analysis.reached(RECORDS[3], "cv") is False
        assert analysis.reached(RECORDS[0], "holo", "cv") is True


class TestFunnel:
    def test_stage_reached_is_not_printed_as_a_pass_rate(self):
        """It counts casualties for every stage but the last."""
        text = "\n".join(
            analysis.funnel(
                {
                    "requested": {"n_batches": 5},
                    "steps": {"apo": {"unit": "sequences", "n_in": 10, "n_pass": 4}},
                    "stage_reached": {"apo": 6},
                }
            )
        )
        assert "n_pass=4" in text
        assert "stage_reached" not in text


class TestFindRun:
    def test_the_error_names_every_path_it_tried(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="summary.csv"):
            analysis.find_run("camp", "run", roots=(str(tmp_path),))

    def test_the_first_root_holding_a_summary_wins(self, tmp_path):
        later = tmp_path / "b" / "camp" / "run"
        later.mkdir(parents=True)
        (later / "summary.csv").write_text("seq_id\n", encoding="utf-8")
        found = analysis.find_run(
            "camp", "run", roots=(str(tmp_path / "a"), str(tmp_path / "b"))
        )
        assert found == later


class TestRoundTrip:
    def test_records_and_summary_load_from_a_directory(self, tmp_path):
        """A notebook passes the run directory, not two file paths."""
        (tmp_path / "summary.csv").write_text(
            "seq_id,holo_iptm\nd0_seq1,0.9\n", encoding="utf-8"
        )
        (tmp_path / "summary.json").write_text(
            json.dumps({"run": "r", "steps": {}}), encoding="utf-8"
        )
        assert analysis.load_records(tmp_path)[0]["seq_id"] == "d0_seq1"
        assert analysis.load_summary(tmp_path)["run"] == "r"


class TestRender:
    def test_nothing_is_an_explicit_message_not_a_blank(self):
        assert analysis.render([]) == "(no rows)"

    def test_missing_cells_print_as_a_dash(self):
        text = analysis.render([{"a": 1.0, "b": None}])
        assert "-" in text.splitlines()[-1]
