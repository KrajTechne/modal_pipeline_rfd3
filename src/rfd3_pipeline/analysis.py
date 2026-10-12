"""Reading a finished campaign: distributions, gate work, cross-stage compares.

Everything here takes the `summary.csv` / `summary.json` that `reduce` already
wrote and recomputes nothing, so an analysis cannot disagree with the artifact
it describes. `experiments/metrics_table.py` is the CLI over this module and
`experiments/campaign_analysis.ipynb` the interactive one; both exist because
the same handful of questions get asked of every campaign, and answering them
in a throwaway script each time is how two readings of one run end up
disagreeing.

**Standard library only, deliberately.** A Modal notebook kernel with the
volume attached can import this and work immediately, with no image to build
and no numpy to install.

Every function returns rows -- lists of plain dicts -- rather than printing.
`render` turns rows into a table when that is what is wanted, and a notebook
can hand the same rows to a DataFrame or a plot instead.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

# Folding stages, in pipeline order. The backbone filters are design-unit and
# differently shaped, so they are read from `summary.json`, not from here.
STAGES = ("apo", "holo", "cv")

# (column suffix, label). The stage prefix is applied per stage, so one row
# covers `apo_plddt_binder`, `holo_plddt_binder` and `cv_plddt_binder`.
METRICS = (
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
)

# Where a campaign's artifacts turn up, most specific first: inside a pipeline
# container, in a Modal notebook with the volume attached, then a local pull
# (`structures/` is gitignored and is where `modal volume get` output goes),
# then the working directory.
_VOLUME_ROOTS = ("/data", "/mnt/rfd3-pipeline", "structures", ".")


def find_run(campaign: str, run: str, roots: Iterable[str] = _VOLUME_ROOTS) -> Path:
    """The directory holding one run's `summary.csv`, wherever it is mounted.

    Saves every notebook from hard-coding a mount point it cannot know: the
    volume is at `/data` in a pipeline container and `/mnt/rfd3-pipeline` in a
    Modal notebook. Raises rather than returning a path that holds nothing,
    because a later `FileNotFoundError` on `summary.csv` is the harder error to
    read.
    """
    tried = []
    for root in roots:
        candidate = Path(root) / campaign / run
        tried.append(candidate)
        if (candidate / "summary.csv").is_file():
            return candidate
    raise FileNotFoundError(
        "no summary.csv under any of: " + ", ".join(str(p) for p in tried)
    )


def load_records(path: str | Path) -> list[dict]:
    """`summary.csv` as rows, strings left as written.

    Coercion is `value`'s job, per column, because the CSV is deliberately
    sparse -- a metric a stage never reached is an empty string, and guessing a
    type for it here would turn "not measured" into 0.0.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "summary.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_summary(path: str | Path) -> dict:
    """`summary.json`: the funnel, the requested scale, the per-gate counts."""
    path = Path(path)
    if path.is_dir():
        path = path / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def value(record: Mapping[str, object], column: str) -> float | None:
    """One cell as a float, or None for absent, empty or non-numeric.

    None means "not measured" and must stay distinguishable from 0.0, which for
    ipSAE is a real and common verdict.
    """
    raw = record.get(column)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def truth(record: Mapping[str, object], column: str) -> bool | None:
    """A gate verdict. The CSV writes Python bools, so "True"/"False"."""
    raw = record.get(column)
    if raw in (None, "", "None"):
        return None
    if isinstance(raw, bool):
        return raw
    return str(raw) == "True"


def numbers(records: Sequence[Mapping[str, object]], column: str) -> list[float]:
    """Every measured value in one column, sorted."""
    return sorted(v for v in (value(r, column) for r in records) if v is not None)


def quantile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank, so every number reported is one a design actually had.

    Interpolating would print values no design scored, which is a poor basis
    for arguing about a threshold.
    """
    if not values:
        return None
    index = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[index]


def column_for(
    records: Sequence[Mapping[str, object]], stage: str, suffix: str
) -> tuple[str, str]:
    """`{stage}_{suffix}`, or the `_mean` aggregate when that is what exists.

    Cross-validation folds several seeds and reports `cv_iptm_mean/_best/
    _worst` rather than a bare `cv_iptm`, so a straight lookup silently leaves
    CV's ipTM out. The returned qualifier labels the fallback, so an average
    over seeds is never mistaken for a single fold.
    """
    direct = f"{stage}_{suffix}"
    head = records[0] if records else {}
    if direct in head:
        return direct, ""
    averaged = f"{direct}_mean"
    if averaged in head:
        return averaged, " (mean)"
    return direct, ""


def distribution(records: Sequence[Mapping[str, object]], stage: str) -> list[dict]:
    """Spread of each of `stage`'s metrics: what calibrating a threshold needs.

    A threshold is defensible when it sits somewhere defensible in a
    distribution, which no single design's value can show.
    """
    rows = []
    for suffix, label in METRICS:
        column, qualifier = column_for(records, stage, suffix)
        values = numbers(records, column)
        if not values:
            continue
        rows.append(
            {
                "stage": stage,
                "metric": f"{label}{qualifier}",
                "n": len(values),
                "min": values[0],
                "p25": quantile(values, 0.25),
                "med": quantile(values, 0.5),
                "p75": quantile(values, 0.75),
                "max": values[-1],
            }
        )
    return rows


def gate_terms(records: Sequence[Mapping[str, object]], stage: str) -> dict:
    """Which of `stage`'s gate terms rejected anything, and which did it alone.

    A gate whose terms never fail is not gating. `threshold_ipTM` sat at 0.3
    for months while every sequence measured scored 0.735 or better, and
    nothing said so; raised to 0.7, it rejected 43 of 292 holo rows and was the
    sole reason for none of them. `sole` is the actionable column, because a
    design that failed three terms is not evidence about any one of them.
    """
    verdicts = [r for r in records if truth(r, f"{stage}_pass") is not None]
    head = records[0] if records else {}
    terms = sorted(
        column
        for column in head
        if column.startswith(f"{stage}_")
        and column.endswith("_pass")
        and column != f"{stage}_pass"
    )
    name = lambda term: term[len(stage) + 1 : -len("_pass")]  # noqa: E731

    sole: Counter[str] = Counter()
    for record in verdicts:
        failed = [t for t in terms if truth(record, t) is False]
        if len(failed) == 1:
            sole[name(failed[0])] += 1

    return {
        "stage": stage,
        "n": len(verdicts),
        "passed": sum(1 for r in verdicts if truth(r, f"{stage}_pass")),
        "rejected": {
            name(t): sum(1 for r in verdicts if truth(r, t) is False) for t in terms
        },
        "sole": dict(sole.most_common()),
    }


def compare(
    records: Sequence[Mapping[str, object]],
    left: str = "holo",
    right: str = "cv",
    suffixes: Iterable[str] = (
        "rmsd_to_design",
        "rmsd_binder_on_target",
        "rmsd_motif_on_target",
        "iptm",
        "ipsae_min",
        "plddt_binder",
        "pae_floor_B",
    ),
) -> list[dict]:
    """Two stages' metrics side by side, on the rows that reached both.

    The point of cross-validating is this table: on `alphav_beta3_02` the two
    models agreed on fold (1.11 -> 1.69 A median) and disagreed on placement
    (2.72 -> 13.10 A), which no single-model column could have shown.
    """
    both = [r for r in records if reached(r, left, right)]
    rows = []
    for suffix in suffixes:
        left_column, _ = column_for(records, left, suffix)
        right_column, qualifier = column_for(records, right, suffix)
        left_values = numbers(both, left_column)
        right_values = numbers(both, right_column)
        if not left_values or not right_values:
            continue
        rows.append(
            {
                "metric": f"{suffix}{qualifier}",
                f"{left}_p25": quantile(left_values, 0.25),
                f"{left}_med": quantile(left_values, 0.5),
                f"{right}_p25": quantile(right_values, 0.25),
                f"{right}_med": quantile(right_values, 0.5),
            }
        )
    return rows


def reached(
    record: Mapping[str, object], *stages: str, suffix: str = "plddt_binder"
) -> bool:
    """Did this record reach every one of `stages`?

    Keyed on a metric every folding stage records rather than on
    `stage_reached`, so it stays true for a row that went further still.
    """
    return all(
        value(record, column_for([record], stage, suffix)[0]) is not None
        for stage in stages
    )


def bucket(
    records: Sequence[Mapping[str, object]],
    metric: str,
    by: str,
    edges: Sequence[float] = (5.0, 15.0),
) -> list[dict]:
    """`metric`'s spread within bands of `by`, with the exact zeros counted.

    Built for the question of whether ipSAE tracks placement. `zeros` is
    reported beside the median because a median of 0.000 hides whether it was
    half the band or all of it, and for ipSAE an exact 0.0 means "no residue
    pair fell under the PAE cutoff" -- a verdict, not missing data.
    """
    bounds = [0.0, *edges, float("inf")]
    rows = []
    for low, high in zip(bounds, bounds[1:]):
        band = [
            r
            for r in records
            if (v := value(r, by)) is not None and low <= v < high
        ]
        values = numbers(band, metric)
        if not values:
            continue
        rows.append(
            {
                "band": f"{low:g}-{high:g}" if high != float("inf") else f">{low:g}",
                "n": len(band),
                "zeros": sum(1 for v in values if v == 0.0),
                "med": quantile(values, 0.5),
                "max": values[-1],
            }
        )
    return rows


def agreed(
    records: Sequence[Mapping[str, object]],
    placement: float = 5.0,
    motif: float = 2.0,
) -> list[dict]:
    """The rows where cross-validation confirms holo's placement and motif.

    **These are the campaign's result, and they are meant to be few.** Holo and
    CV disagree more often than not; agreement is the signal precisely because
    two independent models concurred where they usually do not. On
    `alphav_beta3_02`, 16 of 138. Do not read the other 122 as a yield problem,
    and do not try to predict membership from holo -- if that worked, CV would
    be measuring nothing.

    Defaults match the holo gate so the question is "does the second model
    agree", not "does it clear some other bar".
    """
    out = []
    for record in records:
        place = value(record, "cv_rmsd_binder_on_target")
        if place is None or place > placement:
            continue
        motif_rmsd = value(record, "cv_rmsd_motif_on_target")
        if motif_rmsd is not None and motif_rmsd > motif:
            continue
        out.append(dict(record))
    return out


def by_design(records: Sequence[Mapping[str, object]]) -> list[dict]:
    """Sequences per design, most first -- yield at the unit that gets ordered.

    Sequence counts overstate how much a campaign found: on
    `alphav_beta3_02`'s 16 agreed sequences, 7 came from two backbones. A
    design with four surviving sequences is evidence about the backbone; four
    designs with one each are four weaker pieces of evidence.
    """
    counts = Counter(r.get("design_id") for r in records if r.get("design_id"))
    return [
        {"design_id": design, "n_sequences": count}
        for design, count in counts.most_common()
    ]


def top(
    records: Sequence[Mapping[str, object]], column: str, n: int = 10
) -> list[dict]:
    """The `n` rows scoring highest on `column`; negate it for lowest."""
    scored = [(value(r, column), r) for r in records]
    ranked = sorted(
        ((v, r) for v, r in scored if v is not None), key=lambda pair: -pair[0]
    )
    return [
        {"seq_id": r.get("seq_id"), "design_id": r.get("design_id"), column: v}
        for v, r in ranked[:n]
    ]


def record_for(records: Sequence[Mapping[str, object]], seq_id: str) -> dict:
    """One sequence's full record, empty cells dropped."""
    for record in records:
        if record.get("seq_id") == seq_id:
            return {k: v for k, v in record.items() if v not in ("", None)}
    raise KeyError(seq_id)


def funnel(summary: Mapping[str, object]) -> list[str]:
    """`summary.json`'s stage counts as lines, gate terms indented beneath.

    `stage_reached` is deliberately not printed: it counts where records came
    to rest, so for every stage but the last those are the casualties. The
    per-stage `n_pass`/`n_fail` are the pass rates.
    """
    lines = []
    asked = summary.get("requested")
    if isinstance(asked, Mapping):
        lines.append("requested: " + ", ".join(f"{k}={v}" for k, v in asked.items()))
    steps = summary.get("steps")
    if not isinstance(steps, Mapping):
        return lines
    for stage, step in steps.items():
        if not isinstance(step, Mapping):
            continue
        counts = " ".join(
            f"{k}={v}" for k, v in step.items() if k not in ("unit", "metrics")
        )
        lines.append(f"{stage:18s} {str(step.get('unit', '')):10s} {counts}")
        metrics = step.get("metrics")
        if isinstance(metrics, Mapping):
            for metric, detail in metrics.items():
                lines.append(
                    f"    {metric:26s} {detail['rule']:>8s}  n_fail={detail['n_fail']}"
                )
    return lines


def render(rows: Sequence[Mapping[str, object]], places: int = 3) -> str:
    """Rows as a fixed-width table, columns taken from the first row.

    Deliberately plain: these tables get pasted into `Notes.md` and commit
    messages, where a markdown grid or a DataFrame repr is noise.
    """
    if not rows:
        return "(no rows)"
    columns = list(rows[0])

    def cell(item: object) -> str:
        if isinstance(item, float):
            return f"{item:.{places}f}"
        return "-" if item is None else str(item)

    widths = {
        column: max(len(column), *(len(cell(row.get(column))) for row in rows))
        for column in columns
    }
    head = "  ".join(f"{c:>{widths[c]}}" for c in columns)
    body = [
        "  ".join(f"{cell(row.get(c)):>{widths[c]}}" for c in columns) for row in rows
    ]
    return "\n".join([head, "-" * len(head), *body])
