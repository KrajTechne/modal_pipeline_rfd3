"""A refold stage must not inherit its own previous attempt's columns.

The bug this exists to prevent was real. An apo re-run that computed pLDDT
successfully still carried the earlier attempt's `apo_plddt_error` forward from
the shard, so the summary row asserted both a value and a failure:

    apo_plddt_binder = 58.114
    apo_plddt_error  = "AttributeError: 'AtomArray' object has no attribute 'b_factor'"

Diagnostic fields are the dangerous ones, because a stale number looks like a
measurement and a stale error looks like a live problem.
"""

from __future__ import annotations

import app as modal_app


def strip(record: dict, stage: str) -> dict:
    """What `_refold` keeps from a shard before writing its own columns."""
    superseded = modal_app.REFOLD_SUPERSEDES[stage]
    return {k: v for k, v in record.items() if not k.startswith(superseded)}


SHARD = {
    "seq_id": "d_seq1",
    "design_id": "d",
    "sequence": "MGIV",
    "bb_pass": True,
    "bb_binder_length": 105,
    "mpnn_seq_rec": 0.38,
    "apo_plddt_binder": 58.1,
    "apo_plddt_error": "AttributeError: ...",
    "apo_rmsd_to_design": 3.6,
    "apo_pass": False,
    "holo_iptm": 0.42,
    "cv_mean_ipsae": 0.31,
    "rank_overall": 7,
    "cluster_id": 2,
}


class TestApo:
    def test_its_own_stale_columns_are_dropped(self):
        kept = strip(SHARD, "apo")
        assert not any(k.startswith("apo_") for k in kept)

    def test_the_contradictory_error_field_is_gone(self):
        """The exact shape of the 2026-10-08 bug."""
        assert "apo_plddt_error" not in strip(SHARD, "apo")

    def test_downstream_stages_are_dropped_too(self):
        kept = strip(SHARD, "apo")
        for key in ("holo_iptm", "cv_mean_ipsae", "rank_overall", "cluster_id"):
            assert key not in kept

    def test_upstream_columns_survive(self):
        """A sequence row keeps its backbone and MPNN metrics -- the summary is
        one row per sequence, so losing them would lose them entirely."""
        kept = strip(SHARD, "apo")
        assert kept["bb_pass"] is True
        assert kept["bb_binder_length"] == 105
        assert kept["mpnn_seq_rec"] == 0.38
        assert kept["sequence"] == "MGIV"
        assert kept["seq_id"] == "d_seq1"


class TestHolo:
    def test_holo_keeps_apo(self):
        """Holo must not drop apo: the apo/holo RMSD gate needs both, and apo
        is a completed upstream stage rather than a previous attempt."""
        kept = strip(SHARD, "holo")
        assert kept["apo_plddt_binder"] == 58.1
        assert kept["apo_rmsd_to_design"] == 3.6

    def test_holo_drops_its_own_and_downstream(self):
        kept = strip(SHARD, "holo")
        assert "holo_iptm" not in kept
        assert "cv_mean_ipsae" not in kept
        assert "rank_overall" not in kept


class TestCoverage:
    def test_every_refold_stage_has_a_supersedes_entry(self):
        """A stage added to REFOLD_STAGES without one would silently inherit
        everything."""
        assert set(modal_app.REFOLD_STAGES) == set(modal_app.REFOLD_SUPERSEDES)

    def test_each_stage_supersedes_itself(self):
        for stage, prefixes in modal_app.REFOLD_SUPERSEDES.items():
            assert f"{stage}_" in prefixes, f"{stage} does not supersede its own columns"
