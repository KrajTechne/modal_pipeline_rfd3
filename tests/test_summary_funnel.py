"""The funnel has to count the right things at each stage.

Three bugs lived here, all of which reported plausible numbers rather than
failing. They are pinned individually because each was invisible under the
conditions that existed when the code was written, and became wrong only when a
later stage was added:

* `n_in` was `stage_reached == stage`, which counts only what *stopped* at a
  stage. Correct while `backbone_filters` was the only gated stage; said
  `n_in: 0` for apo the moment a sequence survived apo.
* design-unit stages counted rows, and after MPNN one design occupies several
  rows carrying its `bb_*` columns forward -- so three designs entering the
  filters were reported as five.
* a stage with no gates still emitted `n_fail`, which read as "two designs
  failed cross-validation" when nothing can fail it.
"""

from __future__ import annotations

from rfd3_pipeline.gates import Gate, Stage
from rfd3_pipeline.report import build_summary_json

BACKBONE = (Gate("loop_fraction", "<=", 0.5),)
APO = (Gate("plddt_binder", ">=", 70.0),)
HOLO = (Gate("iptm", ">=", 0.3), Gate("rmsd_apo_holo", "<=", 4.0))

# One design failing the filters, one design yielding two sequences that reach
# cv, one yielding two that stop at holo. The shape of a real dry run.
RECORDS = [
    {
        "design_id": "d0",
        "bb_pass": False,
        "bb_loop_fraction_pass": False,
        "stage_reached": Stage.BACKBONE_FILTERS,
    },
    *[
        {
            "design_id": "d1",
            "seq_id": f"d1_seq{i}",
            "bb_pass": True,
            "bb_loop_fraction_pass": True,
            "apo_pass": True,
            "apo_plddt_binder_pass": True,
            "holo_pass": True,
            "holo_iptm_pass": True,
            "holo_rmsd_apo_holo_pass": True,
            "cv_n_seeds_ok": 5,
            "stage_reached": Stage.CV,
        }
        for i in (1, 2)
    ],
    *[
        {
            "design_id": "d2",
            "seq_id": f"d2_seq{i}",
            "bb_pass": True,
            "bb_loop_fraction_pass": True,
            "apo_pass": True,
            "apo_plddt_binder_pass": True,
            "holo_pass": False,
            "holo_iptm_pass": True,
            "holo_rmsd_apo_holo_pass": False,
            "stage_reached": Stage.HOLO,
        }
        for i in (1, 2)
    ],
]

GATES = {
    Stage.BACKBONE_FILTERS: BACKBONE,
    Stage.APO: APO,
    Stage.HOLO: HOLO,
}


def summary(records=None, requested=None) -> dict:
    return build_summary_json(
        records if records is not None else RECORDS,
        "campaign",
        "run",
        n_specs=1,
        stage_gates=GATES,
        dry_run=True,
        requested=requested,
    )


class TestEveryStageAppears:
    def test_all_six_stages_are_reported(self):
        """The summary described two stages out of six, because only the
        backbone gates were passed to it."""
        assert set(summary()["steps"]) == set(Stage.ORDER)


class TestReachedMeansAtOrBeyond:
    def test_apo_counts_sequences_that_survived_it(self):
        # All four reached apo; two went on to cv, so equality would see none.
        assert summary()["steps"][Stage.APO]["n_in"] == 4

    def test_holo_counts_both_its_survivors_and_its_failures(self):
        holo = summary()["steps"][Stage.HOLO]
        assert (holo["n_in"], holo["n_pass"], holo["n_fail"]) == (4, 2, 2)

    def test_the_failing_metric_is_the_one_blamed(self):
        metrics = summary()["steps"][Stage.HOLO]["metrics"]
        assert metrics["rmsd_apo_holo"]["n_fail"] == 2
        assert metrics["iptm"]["n_fail"] == 0

    def test_the_rule_is_recorded_beside_the_count(self):
        """So a reader can tell 2 failures at <= 4.0 from 2 at <= 1.0."""
        assert summary()["steps"][Stage.HOLO]["metrics"]["iptm"]["rule"] == ">= 0.3"


class TestUnits:
    def test_backbone_filters_counts_designs_not_rows(self):
        backbone = summary()["steps"][Stage.BACKBONE_FILTERS]
        assert backbone["unit"] == "designs"
        # Three designs entered, not the five rows that carry bb_* columns.
        assert (backbone["n_in"], backbone["n_pass"], backbone["n_fail"]) == (3, 2, 1)

    def test_refolding_counts_sequences(self):
        assert summary()["steps"][Stage.APO]["unit"] == "sequences"

    def test_mpnn_reports_its_fan_out_rather_than_a_pass_rate(self):
        mpnn = summary()["steps"][Stage.MPNN]
        assert mpnn["n_in_designs"] == 2
        assert mpnn["n_out"] == 4
        assert "n_pass" not in mpnn


class TestGatelessStage:
    def test_cv_reports_no_failures_because_nothing_can_fail_it(self):
        cv = summary()["steps"][Stage.CV]
        assert cv["n_in"] == 2
        assert cv["n_reported"] == 2
        assert "n_fail" not in cv and "n_pass" not in cv


class TestRequested:
    def test_what_the_campaign_asked_for_is_recorded(self):
        """Without it, three designs could mean "three configured" or "the rest
        failed", and at dry-run scale those look identical."""
        asked = {"n_batches": 1, "diffusion_batch_size": 3, "total_designs": 3,
                 "n_mpnn_seqs_per_design": 2, "max_sequences": 6}
        assert summary(requested=asked)["requested"] == asked

    def test_it_is_omitted_rather_than_guessed_when_absent(self):
        assert "requested" not in summary()

    def test_totals_separate_designs_from_sequences(self):
        out = summary()
        assert (out["n_designs"], out["n_sequences"]) == (3, 4)
