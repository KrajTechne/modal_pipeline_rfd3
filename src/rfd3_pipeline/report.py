"""Shards in, `summary.csv` and `summary.json` out.

Each worker writes one shard per item and never touches a shared file: Modal
volumes are not POSIX-consistent under concurrent writers, so an
append-with-lock does not serialise the writes, it just makes you believe they
were serialised. The summaries are derived artifacts, rebuilt from the shards by
a single reducer, which also means a crash mid-write costs nothing and
`--restart-from` regenerates them for free.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .gates import Stage

__all__ = [
    "build_summary_csv",
    "build_summary_json",
    "order_columns",
    "latest_records",
    "read_shards",
    "write_shard",
    "write_summaries",
]

# Identity and outcome columns, in the order they should read left to right.
# Everything else is grouped by stage prefix and sorted within the group, so the
# header is stable across runs without being hand-maintained.
_LEADING = (
    "campaign",
    "run_name",
    "spec_name",
    "design_id",
    "seq_id",
    "mpnn_index",
    "stage_reached",
    "passes_filters",
    "fail_stage",
    "fail_reasons",
)
_STAGE_PREFIXES = ("bb_", "mpnn_", "apo_", "holo_", "cv_", "rank_", "cluster_")


def write_shard(record: Mapping[str, object], directory: str | Path) -> Path:
    """Write one record as its own file, named by the id it describes.

    Written to a temporary name and renamed, so an interrupted write cannot
    leave a half-written shard that the reducer would then parse.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    name = record.get("seq_id") or record.get("design_id")
    if not name:
        raise ValueError("a shard record needs a design_id or seq_id to be named by")

    final = directory / f"{name}.json"
    staging = final.with_suffix(".json.partial")
    staging.write_text(json.dumps(record, indent=2, default=str))
    staging.replace(final)
    return final


def read_shards(directory: str | Path) -> list[dict]:
    """Every shard in a directory, sorted by file name for a stable order."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return [
        json.loads(path.read_text())
        for path in sorted(directory.glob("*.json"))
    ]


def order_columns(records: Sequence[Mapping[str, object]]) -> list[str]:
    """A stable header: known identity columns, then stage groups, then the rest."""
    present = {key for record in records for key in record}

    ordered = [name for name in _LEADING if name in present]
    seen = set(ordered)

    for prefix in _STAGE_PREFIXES:
        group = sorted(k for k in present if k.startswith(prefix) and k not in seen)
        ordered.extend(group)
        seen.update(group)

    ordered.extend(sorted(present - seen))
    return ordered


def build_summary_csv(records: Sequence[Mapping[str, object]]) -> str:
    """`summary.csv`, one row per shard."""
    if not records:
        return ""
    columns = order_columns(records)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow({name: record.get(name, "") for name in columns})
    return buffer.getvalue()


def _funnel(records: Sequence[Mapping[str, object]], stage: str, gates) -> dict:
    """Counts for one stage, plus per-metric failure counts.

    `n_in` counts designs that reached the stage at all, so a design that died
    earlier is absent rather than counted as a failure here.
    """
    reached = [r for r in records if r.get("stage_reached") == stage]
    passed = [r for r in reached if r.get(f"{_stage_prefix(stage)}pass") is True]

    per_metric: dict[str, dict[str, object]] = {}
    for gate in gates:
        failed = sum(
            1
            for r in reached
            if r.get(f"{_stage_prefix(stage)}{gate.metric}_pass") is False
        )
        per_metric[gate.metric] = {"rule": gate.rule, "n_fail": failed}

    return {
        "n_in": len(reached),
        "n_pass": len(passed),
        "n_fail": len(reached) - len(passed),
        "metrics": per_metric,
    }


def _stage_prefix(stage: str) -> str:
    return {
        Stage.BACKBONE_FILTERS: "bb_",
        Stage.APO: "apo_",
        Stage.HOLO: "holo_",
        Stage.CV: "cv_",
    }.get(stage, "")


def build_summary_json(
    records: Sequence[Mapping[str, object]],
    campaign: str,
    run_name: str,
    n_specs: int,
    stage_gates: Mapping[str, Iterable],
    dry_run: bool = False,
) -> dict:
    """The funnel: how many designs reached each stage, and what failed them.

    `stage_gates` maps a stage name to its gates, so the per-metric failure
    counts come from the same declarations the filter evaluated -- the rules in
    this file and the rules applied to the data cannot drift apart.
    """
    counts = {
        "campaign": campaign,
        "run": run_name,
        # Recorded so the artifact itself says whether these numbers describe a
        # real campaign, independent of whoever saw the console banner.
        "dry_run": dry_run,
        "n_specs": n_specs,
        "n_designs": len(records),
        "steps": {
            Stage.RFD3: {
                "n_in": n_specs,
                "n_designs": len(records),
                "n_fail": 0,
            }
        },
    }
    for stage, gates in stage_gates.items():
        counts["steps"][stage] = _funnel(records, stage, list(gates))

    counts["survivors"] = {
        stage: sum(1 for r in records if r.get("stage_reached") == stage)
        for stage in Stage.ORDER
        if any(r.get("stage_reached") == stage for r in records)
    }
    return counts


def latest_records(
    records: Sequence[Mapping[str, object]],
) -> list[dict]:
    """One row per sequence, plus one per backbone that never got that far.

    A design's backbone shard and its sequence shards describe the same design
    at different stages, and the sequence records carry the backbone's `bb_*`
    columns forward. Keeping both would double count: a passing backbone would
    appear once as itself and once per sequence, so `summary.csv` would mix
    granularities and any `bb_pass` tally would be inflated by the number of
    sequences.

    The backbone record is therefore dropped once sequences exist for that
    design, and kept otherwise -- which is exactly the set the plan describes,
    since a backbone that failed the filters never reaches MPNN and stays its
    own terminal row.

    The shards themselves are left alone, so restarting at `mpnn` still sees
    every backbone verdict.
    """
    superseded = {
        record.get("design_id")
        for record in records
        if record.get("seq_id") and record.get("design_id")
    }
    return [
        dict(record)
        for record in records
        if record.get("seq_id") or record.get("design_id") not in superseded
    ]


def write_summaries(
    directory: str | Path,
    records: Sequence[Mapping[str, object]],
    campaign: str,
    run_name: str,
    n_specs: int,
    stage_gates: Mapping[str, Iterable],
    dry_run: bool = False,
) -> dict[str, Path]:
    """Write both summaries beside the shards, returning where they landed."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    csv_path = directory / "summary.csv"
    json_path = directory / "summary.json"
    csv_path.write_text(build_summary_csv(records))
    json_path.write_text(
        json.dumps(
            build_summary_json(
                records, campaign, run_name, n_specs, stage_gates, dry_run
            ),
            indent=2,
            default=str,
        )
    )
    return {"summary.csv": csv_path, "summary.json": json_path}
