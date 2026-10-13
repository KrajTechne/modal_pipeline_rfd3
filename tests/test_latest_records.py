"""`latest_records`: one row per sequence, plus backbones that died earlier.

The bug this exists to prevent was real, not hypothetical. On 2026-10-07 the
first MPNN run reported `n_passing_backbone_filters: 3` for a single passing
backbone, because its two sequence shards each carried `bb_pass` forward and the
backbone shard was still counted alongside them.
"""

from __future__ import annotations

from rfd3_pipeline.report import latest_records


def backbone(design_id: str, *, passed: bool = True) -> dict:
    return {
        "design_id": design_id,
        "bb_pass": passed,
        "bb_binder_length": 105,
        "stage_reached": "backbone_filters",
    }


def sequence(design_id: str, index: int) -> dict:
    return {
        "design_id": design_id,
        "seq_id": f"{design_id}_seq{index}",
        "mpnn_index": index,
        "bb_pass": True,
        "bb_binder_length": 105,
        "stage_reached": "mpnn",
    }


class TestSupersession:
    def test_backbone_is_dropped_once_its_sequences_exist(self):
        records = latest_records([backbone("d1"), sequence("d1", 1), sequence("d1", 2)])
        assert [r["seq_id"] for r in records] == ["d1_seq1", "d1_seq2"]

    def test_the_inflated_pass_count_is_gone(self):
        """The exact shape of the 2026-10-07 miscount."""
        shards = [
            backbone("pass", passed=True),
            backbone("fail", passed=False),
            sequence("pass", 1),
            sequence("pass", 2),
        ]
        records = latest_records(shards)
        assert len(records) == 3  # two sequences + the failed backbone
        designs = {r["design_id"] for r in records if r.get("bb_pass") is True}
        assert designs == {"pass"}

    def test_a_failed_backbone_keeps_its_row(self):
        """It never reaches MPNN, so it is its own terminal record and must
        still appear in the summary."""
        records = latest_records([backbone("died", passed=False)])
        assert len(records) == 1
        assert records[0]["design_id"] == "died"
        assert "seq_id" not in records[0]

    def test_designs_are_independent(self):
        records = latest_records(
            [backbone("a"), backbone("b"), sequence("a", 1)]
        )
        kinds = {r.get("seq_id") or r["design_id"] for r in records}
        assert kinds == {"a_seq1", "b"}


class TestPassThrough:
    def test_empty_input(self):
        assert latest_records([]) == []

    def test_sequences_only_are_all_kept(self):
        records = latest_records([sequence("d", 1), sequence("d", 2)])
        assert len(records) == 2

    def test_order_is_preserved(self):
        shards = [backbone("z", passed=False), sequence("a", 1), sequence("a", 2)]
        assert [r.get("seq_id") or r["design_id"] for r in latest_records(shards)] == [
            "z",
            "a_seq1",
            "a_seq2",
        ]

    def test_records_are_copied_not_aliased(self):
        """The reducer should not be able to mutate the caller's shards."""
        original = sequence("d", 1)
        (returned,) = latest_records([original])
        returned["sequence"] = "MUTATED"
        assert "sequence" not in original

    def test_a_record_without_a_design_id_is_kept(self):
        """Defensive: nothing should produce one, but dropping it silently
        would be worse than carrying it into the summary."""
        odd = {"seq_id": "orphan", "stage_reached": "mpnn"}
        assert latest_records([odd]) == [odd]
