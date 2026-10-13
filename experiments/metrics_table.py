"""Cross-stage metric tables from a campaign's `summary.csv`.

    python experiments/metrics_table.py <summary.csv> [summary.json] [--rows N]

(`--rows N` and `--rows=N` both work; `--rows 0` drops the per-sequence table.)

The analysis lives in `rfd3_pipeline.analysis`; this only arranges it into
printed sections. `experiments/campaign_analysis.ipynb` is the other caller,
and the two must not drift -- so a new question goes in the module, not here.

Sections, cheapest question first:

0. **The funnel**, straight from `summary.json`.
1. **Distributions per stage.** What calibration needs: the spread of each
   metric, not any one design's value.
2. **Which gate term did the rejecting.** A gate whose terms never fail is not
   gating, and nothing else says so.
3. **Holo against CV**, then the agreed set -- the campaign's actual result,
   and meant to be rare. See `analysis.agreed`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rfd3_pipeline import analysis  # noqa: E402


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main(argv: list[str]) -> int:
    limit, args, pending = 40, [], False
    for arg in argv:
        if pending:
            limit, pending = int(arg), False
        elif arg == "--rows":
            pending = True
        elif arg.startswith("--rows="):
            limit = int(arg.split("=", 1)[1])
        elif not arg.startswith("--"):
            args.append(arg)
    if not args:
        print(__doc__)
        return 2

    records = analysis.load_records(args[0])
    if not records:
        print(f"{args[0]}: no rows")
        return 1

    if len(args) > 1:
        banner("0. FUNNEL")
        print("\n".join(analysis.funnel(analysis.load_summary(args[1]))))

    banner("1. DISTRIBUTIONS  (nearest-rank: every value is one a design had)")
    for stage in analysis.STAGES:
        rows = analysis.distribution(records, stage)
        if rows:
            print()
            print(analysis.render(rows))

    banner("2. WHAT EACH GATE TERM REJECTED")
    for stage in analysis.STAGES:
        work = analysis.gate_terms(records, stage)
        if not work["n"]:
            continue
        print(f"\n  {stage}: {work['passed']}/{work['n']} passed")
        if not work["rejected"]:
            print("    (ungated -- its metrics inform ranking only)")
            continue
        for metric, count in work["rejected"].items():
            verdict = "never rejected anything" if not count else f"rejected {count}"
            print(f"    {metric:26s} {verdict}")
        for metric, count in work["sole"].items():
            print(f"    sole reason: {metric:26s} {count}")

    banner("3. HOLO AGAINST CV")
    print(analysis.render(analysis.compare(records)))
    agreed = analysis.agreed(records)
    folded = [r for r in records if analysis.reached(r, "cv")]
    print(
        f"\n  CV confirms holo on {len(agreed)}/{len(folded)} sequences."
        "\n  Meant to be few: two models that usually disagree, agreeing."
    )
    if agreed:
        print()
        print(analysis.render(analysis.by_design(agreed)))

    if limit and agreed:
        banner(f"4. THE AGREED SEQUENCES  ({min(limit, len(agreed))} shown)")
        ranked = sorted(
            agreed, key=lambda r: analysis.value(r, "cv_rmsd_motif_on_target") or 0.0
        )
        print(
            analysis.render(
                [
                    {
                        "seq_id": r["seq_id"],
                        "holo_place": analysis.value(r, "holo_rmsd_binder_on_target"),
                        "cv_place": analysis.value(r, "cv_rmsd_binder_on_target"),
                        "holo_motif": analysis.value(r, "holo_rmsd_motif_on_target"),
                        "cv_motif": analysis.value(r, "cv_rmsd_motif_on_target"),
                        "cv_iptm": analysis.value(r, "cv_iptm_mean"),
                        "cv_ipsae": analysis.value(r, "cv_ipsae_min"),
                    }
                    for r in ranked[:limit]
                ]
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
