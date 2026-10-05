from __future__ import annotations

import csv
import io
import json

import pytest

from rfd3_pipeline.gates import Gate, Stage
from rfd3_pipeline.report import (
    build_summary_csv,
    build_summary_json,
    order_columns,
    read_shards,
    write_shard,
    write_summaries,
)

GATES = (Gate("loop_fraction", "<=", 0.5), Gate("rg_ratio", "<=", 1.1))


def record(design_id, *, passed=True, loop_ok=True, rg_ok=True):
    return {
        "design_id": design_id,
        "campaign": "tpo",
        "run_name": "tpo_01",
        "spec_name": "rgd_core",
        "stage_reached": Stage.BACKBONE_FILTERS,
        "bb_pass": passed,
        "bb_loop_fraction": 0.25 if loop_ok else 0.9,
        "bb_loop_fraction_pass": loop_ok,
        "bb_rg_ratio": 0.94 if rg_ok else 1.8,
        "bb_rg_ratio_pass": rg_ok,
        "fail_reasons": "" if passed else "something",
    }


class TestShards:
    def test_round_trip(self, tmp_path):
        write_shard(record("d1"), tmp_path)
        write_shard(record("d2"), tmp_path)
        assert [r["design_id"] for r in read_shards(tmp_path)] == ["d1", "d2"]

    def test_named_by_seq_id_when_present(self, tmp_path):
        write_shard({**record("d1"), "seq_id": "d1_seq03"}, tmp_path)
        assert (tmp_path / "d1_seq03.json").exists()

    def test_write_is_atomic(self, tmp_path):
        """Staged under a .partial name and renamed, so an interrupted write
        cannot leave a half-file for the reducer to parse."""
        write_shard(record("d1"), tmp_path)
        assert list(p.name for p in tmp_path.iterdir()) == ["d1.json"]

    def test_unnameable_record_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="design_id or seq_id"):
            write_shard({"campaign": "tpo"}, tmp_path)

    def test_missing_directory_reads_as_empty(self, tmp_path):
        assert read_shards(tmp_path / "nothing") == []


class TestColumnOrder:
    def test_identity_columns_lead(self):
        columns = order_columns([record("d1")])
        assert columns[:4] == ["campaign", "run_name", "spec_name", "design_id"]

    def test_stage_groups_are_contiguous_and_sorted(self):
        columns = order_columns([record("d1")])
        bb = [c for c in columns if c.startswith("bb_")]
        assert bb == sorted(bb)
        first, last = columns.index(bb[0]), columns.index(bb[-1])
        assert last - first == len(bb) - 1

    def test_unknown_columns_are_kept_at_the_end(self):
        columns = order_columns([{**record("d1"), "zzz_extra": 1}])
        assert columns[-1] == "zzz_extra"

    def test_order_is_stable_regardless_of_record_key_order(self):
        forward = record("d1")
        reversed_keys = dict(reversed(list(forward.items())))
        assert order_columns([forward]) == order_columns([reversed_keys])


class TestSummaryCsv:
    def test_header_and_rows(self):
        text = build_summary_csv([record("d1"), record("d2")])
        rows = list(csv.DictReader(io.StringIO(text)))
        assert len(rows) == 2
        assert rows[0]["design_id"] == "d1"
        assert rows[0]["bb_pass"] == "True"

    def test_empty_input(self):
        assert build_summary_csv([]) == ""

    def test_missing_values_become_empty_cells(self):
        """Designs that died early have no columns for later stages."""
        text = build_summary_csv([record("d1"), {"design_id": "d2"}])
        rows = list(csv.DictReader(io.StringIO(text)))
        assert rows[1]["bb_pass"] == ""


class TestSummaryJson:
    def test_funnel_counts(self):
        records = [
            record("d1"),
            record("d2", passed=False, loop_ok=False),
            record("d3", passed=False, rg_ok=False),
        ]
        summary = build_summary_json(
            records, "tpo", "tpo_01", 2, {Stage.BACKBONE_FILTERS: GATES}
        )
        step = summary["steps"][Stage.BACKBONE_FILTERS]
        assert step["n_in"] == 3
        assert step["n_pass"] == 1
        assert step["n_fail"] == 2

    def test_per_metric_failure_counts_and_rules(self):
        records = [
            record("d1", passed=False, loop_ok=False),
            record("d2", passed=False, loop_ok=False),
            record("d3", passed=False, rg_ok=False),
        ]
        summary = build_summary_json(
            records, "tpo", "tpo_01", 2, {Stage.BACKBONE_FILTERS: GATES}
        )
        metrics = summary["steps"][Stage.BACKBONE_FILTERS]["metrics"]
        assert metrics["loop_fraction"] == {"rule": "<= 0.5", "n_fail": 2}
        assert metrics["rg_ratio"] == {"rule": "<= 1.1", "n_fail": 1}

    def test_rules_come_from_the_same_gates_the_filter_applied(self):
        """So the recorded rule cannot drift from the rule that was enforced."""
        summary = build_summary_json(
            [record("d1")], "tpo", "tpo_01", 1, {Stage.BACKBONE_FILTERS: GATES}
        )
        assert summary["steps"][Stage.BACKBONE_FILTERS]["metrics"]["rg_ratio"]["rule"] == (
            GATES[1].rule
        )

    def test_survivors_follow_stage_order(self):
        summary = build_summary_json(
            [record("d1")], "tpo", "tpo_01", 1, {Stage.BACKBONE_FILTERS: GATES}
        )
        assert list(summary["survivors"]) == [Stage.BACKBONE_FILTERS]

    def test_campaign_identity(self):
        summary = build_summary_json(
            [record("d1")], "tpo", "tpo_01", 5, {Stage.BACKBONE_FILTERS: GATES}
        )
        assert (summary["campaign"], summary["run"], summary["n_specs"]) == (
            "tpo",
            "tpo_01",
            5,
        )


class TestWriteSummaries:
    def test_both_files_land_and_parse(self, tmp_path):
        paths = write_summaries(
            tmp_path,
            [record("d1"), record("d2", passed=False, loop_ok=False)],
            campaign="tpo",
            run_name="tpo_01",
            n_specs=2,
            stage_gates={Stage.BACKBONE_FILTERS: GATES},
        )
        assert set(paths) == {"summary.csv", "summary.json"}
        assert list(csv.DictReader(io.StringIO(paths["summary.csv"].read_text())))
        parsed = json.loads(paths["summary.json"].read_text())
        assert parsed["steps"][Stage.BACKBONE_FILTERS]["n_fail"] == 1

    def test_rebuilding_is_idempotent(self, tmp_path):
        """Summaries are derived, so a restart regenerates them for free."""
        args = dict(
            campaign="tpo",
            run_name="tpo_01",
            n_specs=1,
            stage_gates={Stage.BACKBONE_FILTERS: GATES},
        )
        first = write_summaries(tmp_path, [record("d1")], **args)
        text = first["summary.csv"].read_text()
        second = write_summaries(tmp_path, [record("d1")], **args)
        assert second["summary.csv"].read_text() == text
