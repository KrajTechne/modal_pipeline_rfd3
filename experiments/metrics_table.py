"""Cross-stage metric tables from a campaign's `summary.csv`.

    python experiments/metrics_table.py <summary.csv> [summary.json] [--rows N]

(`--rows N` and `--rows=N` both work.)

Written because the hand-made four-sequence table that drove the 2026-10-09
gate revision will be wanted again every campaign, and because at campaign
scale the per-sequence form stops being readable: 400 rows is a dump, not a
table. So it prints three things, cheapest question first.

1. **Distributions per stage.** What calibration actually needs -- the spread
   of each metric, not any one design's value. A threshold is defensible when
   it sits somewhere defensible in a distribution.
2. **Which gate metric did the rejecting.** A gate whose terms never fail is
   not gating; `threshold_ipTM` sat at 0.3 for months while every sequence
   measured scored 0.735 or better, and nothing said so.
3. **The per-sequence table**, bounded by `--rows`, ordered worst-first on the
   binder's weakest interface so the interesting designs are the ones shown.

Deliberately local and read-only: it takes the CSV the pipeline already writes
rather than recomputing anything, so it cannot disagree with the artifact.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# (column suffix, label). Stage prefix is applied per stage.
METRICS = [
    ("plddt_binder", "pLDDT"),
    ("iptm", "ipTM"),
    ("ipsae_min", "ipSAE"),
    ("rmsd_to_design", "fold->design"),
    ("rmsd_apo_holo", "fold apo/holo"),
    ("rmsd_binder_on_target", "placement"),
    ("rmsd_motif_on_target", "motif"),
    ("rmsd_to_holo", "fold->holo"),
    ("rmsd_placement_vs_holo", "placement->holo"),
    ("pae_floor_B", "PAE floor B"),
    ("pae_floor_C", "PAE floor C"),
]

STAGES = ["apo", "holo", "cv"]


def numbers(rows, column):
    out = []
    for row in rows:
        raw = row.get(column)
        if raw in (None, ""):
            continue
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return sorted(out)


def quantile(values, q):
    """Nearest-rank, so every printed number is one a design actually had."""
    if not values:
        return None
    return values[min(len(values) - 1, max(0, round(q * (len(values) - 1))))]


def fmt(value, width=8):
    if value is None:
        return f"{'-':>{width}}"
    return f"{value:{width}.3f}"


def distributions(rows):
    print("=" * 94)
    print("1. DISTRIBUTIONS  (nearest-rank quantiles: every value is one a design had)")
    print("=" * 94)
    head = f"{'stage':6s} {'metric':18s} {'n':>5s} {'min':>8s} {'p25':>8s} {'med':>8s} {'p75':>8s} {'max':>8s}"
    for stage in STAGES:
        shown = False
        for suffix, label in METRICS:
            values = numbers(rows, f"{stage}_{suffix}")
            if not values:
                continue
            if not shown:
                print()
                print(head)
                print("-" * len(head))
                shown = True
            print(
                f"{stage:6s} {label:18s} {len(values):5d} {fmt(values[0])}"
                f" {fmt(quantile(values, 0.25))} {fmt(quantile(values, 0.5))}"
                f" {fmt(quantile(values, 0.75))} {fmt(values[-1])}"
            )


def gate_work(rows):
    """Which gate terms actually reject anything."""
    print()
    print("=" * 94)
    print("2. WHAT EACH GATE TERM REJECTED")
    print("=" * 94)
    for stage in STAGES:
        verdicts = [r for r in rows if r.get(f"{stage}_pass") not in (None, "")]
        if not verdicts:
            continue
        passed = sum(1 for r in verdicts if r[f"{stage}_pass"] == "True")
        print()
        print(f"  {stage}: {passed}/{len(verdicts)} passed")
        terms = sorted(
            column
            for column in rows[0]
            if column.startswith(f"{stage}_") and column.endswith("_pass")
            and column != f"{stage}_pass"
        )
        if not terms:
            print("    (no per-metric gate columns -- stage is ungated)")
            continue
        for term in terms:
            failed = sum(1 for r in verdicts if r.get(term) == "False")
            metric = term[len(stage) + 1 : -len("_pass")]
            verdict = "never rejected anything" if failed == 0 else f"rejected {failed}"
            print(f"    {metric:26s} {verdict}")
        # A design can fail several terms; the sole reason is the actionable one.
        sole = {}
        for row in verdicts:
            bad = [t for t in terms if row.get(t) == "False"]
            if len(bad) == 1:
                key = bad[0][len(stage) + 1 : -len("_pass")]
                sole[key] = sole.get(key, 0) + 1
        if sole:
            print("    sole reason for failure:")
            for metric, count in sorted(sole.items(), key=lambda kv: -kv[1]):
                print(f"      {metric:24s} {count}")


def per_sequence(rows, limit):
    print()
    print("=" * 94)
    print(f"3. PER SEQUENCE  (worst holo ipSAE first, {limit} shown)")
    print("=" * 94)
    folded = [r for r in rows if r.get("seq_id") and r.get("holo_iptm")]

    def key(row):
        raw = row.get("holo_ipsae_min")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 1e9

    folded.sort(key=key)
    head = (
        f"{'seq_id':26s} {'stage':5s} {'fold':>7s} {'place':>7s} {'motif':>7s}"
        f" {'ipTM':>6s} {'ipSAE':>6s} {'gate':>5s}"
    )
    print(head)
    print("-" * len(head))
    for row in folded[:limit]:
        for stage in ("holo", "cv"):
            if not row.get(f"{stage}_iptm") and not row.get(f"{stage}_iptm_mean"):
                continue
            iptm = row.get(f"{stage}_iptm") or row.get(f"{stage}_iptm_mean")
            verdict = {"True": "pass", "False": "FAIL", "": "-"}.get(
                row.get(f"{stage}_pass", ""), "-"
            )
            print(
                f"{row['seq_id'][:26]:26s} {stage:5s}"
                f" {fmt(_f(row.get(f'{stage}_rmsd_to_design')), 7)}"
                f" {fmt(_f(row.get(f'{stage}_rmsd_binder_on_target')), 7)}"
                f" {fmt(_f(row.get(f'{stage}_rmsd_motif_on_target')), 7)}"
                f" {fmt(_f(iptm), 6)} {fmt(_f(row.get(f'{stage}_ipsae_min')), 6)}"
                f" {verdict:>5s}"
            )


def _f(raw):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def funnel(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    print("=" * 94)
    print(f"0. FUNNEL  {payload.get('campaign')} / {payload.get('run')}"
          f"{'  [DRY RUN]' if payload.get('dry_run') else ''}")
    print("=" * 94)
    asked = payload.get("requested")
    if asked:
        print("  requested: " + ", ".join(f"{k}={v}" for k, v in asked.items()))
    for stage, step in payload.get("steps", {}).items():
        unit = step.get("unit", "")
        counts = " ".join(
            f"{k}={v}" for k, v in step.items()
            if k not in ("unit", "metrics")
        )
        print(f"  {stage:18s} {unit:10s} {counts}")
        for metric, detail in step.get("metrics", {}).items():
            print(f"      {metric:26s} {detail['rule']:>8s}  n_fail={detail['n_fail']}")
    print()


def main(argv: list[str]) -> int:
    limit = 40
    args: list[str] = []
    pending = False
    for a in argv:
        if pending:
            limit, pending = int(a), False
        elif a == "--rows":
            pending = True          # `--rows 40`
        elif a.startswith("--rows="):
            limit = int(a.split("=", 1)[1])
        elif not a.startswith("--"):
            args.append(a)
    if not args:
        print(__doc__)
        return 2

    csv_path = Path(args[0])
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        print(f"{csv_path}: no rows")
        return 1

    if len(args) > 1:
        funnel(Path(args[1]))
    distributions(rows)
    gate_work(rows)
    per_sequence(rows, limit)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
