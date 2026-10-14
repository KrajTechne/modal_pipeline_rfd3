# modal_pipeline_rfd3

A protein binder design pipeline on [Modal](https://modal.com), driven end to
end from one campaign YAML:

```
rfd3 → backbone_filters → mpnn → apo → holo → cv → reduce
```

RFDiffusion3 generates backbones against a target and a motif, cheap geometric
filters throw out the unbuildable ones, LigandMPNN designs sequences for the
survivors (soluble, protein or ligand model per campaign), and each sequence is
then folded three times — alone, with its target, and by a second independent
model — with a gate between every stage so expensive work only runs on what
passed the cheap work.

Two full campaigns have run at 80 designs each. See [`runs/`](runs/) for what
they measured.

## Running one

```bash
export PYTHONPATH=src                 # see "Two environment traps" below
modal run app.py::run --campaign inputs_.../campaign.yaml --dry-run
modal run --detach app.py::run --campaign inputs_.../campaign.yaml --spawn
```

The first is a smoke test: one spec, two designs, one cross-validation seed,
written to its own run directory so it cannot be confused with a real campaign.
The second is the real thing, ~3 hours at 80 designs.

**`--detach` *and* `--spawn`, never one alone.** Two campaigns were lost to
this. `--detach` with a blocking call preserves only the *last triggered* call,
so the orchestrator dies and its children outlive it; `--spawn` without
`--detach` runs in an ephemeral app that tears down the moment the entrypoint
returns. `experiments/detach_probe.py` demonstrates all three forms on a CPU
container for pennies.

Everything lands on the `rfd3-pipeline` volume under
`<campaign>/<run_name>/`, including `summary.csv` and `summary.json`. The
orchestrator reduces server-side, so the artifacts are written whether or not a
client is attached — **read those, never the logs.**

### Other entrypoints

| | |
|---|---|
| `run` | the whole campaign; `--start-at <stage>` resumes |
| `refold_only` / `cross_validate_only` / `filter_only` | one stage over existing shards, for proving a change |
| `reduce_only` | rebuild the summaries from shards; safe any time |
| `cache_msas` | build the target MSA cache |
| `install_weights`, `install_mpnn_weights`, `install_af3_fork_weights`, `warm_esm_jit` | one-time setup |
| `publish_tools` | copy the analysis module to the volume for a Modal notebook |

## Configuration

One YAML per campaign, validated by `pydantic` before anything launches —
including that the named models are actually wired, so a typo costs a second
rather than four hours of GPU. See
[`inputs_motif_scaffold_alphav_beta3/`](inputs_motif_scaffold_alphav_beta3/)
for a worked example with the reasoning in comments.

Sections: `general`, `rfdiffusion3`, `mpnn`, `refolding`, `cross_validation`,
`msa`, `ipsae`, `run`.

**Folding models** are chosen per stage, and the two stages must differ —
cross-validation only means something if a second model does it.
`esmfold2` and `esmfold2-fast` run through the Anthropic inference-optimization
kit; `af3_fork` drives [sokrypton/alphafold3](https://github.com/sokrypton/alphafold3)
directly. `boltz2` is planned, see [`boltz2_plan.md`](boltz2_plan.md).

Each model declares what it can do in one place (`MODEL_CAPABILITIES`): whether
it reads unpaired or paired alignments, whether it templates, how it takes
seeds, and whether it ships several checkpoints. A campaign asking a model for
something it cannot do is rejected rather than silently given something else.

## Gates

Every threshold is declared in `gates.py`, so the funnel in `summary.json`
counts the same rule the stage enforced. A missing measurement fails closed:
unknown is not a pass, because the next stage is always the expensive one.

| stage | gate |
|---|---|
| `backbone_filters` | clashes, loop fraction, radius of gyration, hotspot contacts |
| `apo` | `plddt_binder` — fold-to-design is recorded and watched, not gated |
| `holo` | placement RMSD on target, motif RMSD when a spec scaffolds one, ipTM |
| `cv` | **none** — fold, placement, motif, ipTM and ipSAE inform ranking |

Cross-validation is ungated on purpose. Holo and CV are *supposed* to disagree;
the designs where they agree are the result, and they are meant to be few — 16
of 138 and 28 of 144 across the two campaigns. Nothing holo measures predicts
which, and if it did, CV would be measuring nothing.

## Reading a finished run

```bash
python experiments/metrics_table.py <run>/summary.csv <run>/summary.json
```

The funnel, per-stage metric distributions, which gate terms actually rejected
anything, and holo against CV. `experiments/campaign_analysis.ipynb` is the
same functions interactively, and runs in a Modal notebook with the volume
attached.

Both are thin callers of `rfd3_pipeline.analysis`, which reads only the
artifacts `reduce` wrote and recomputes nothing — so an analysis cannot
disagree with what it describes. **Add a question to the module, not to a
caller**: these tables started as throwaway scripts and two of them already
disagreed about a median.

## Layout

```
app.py                     the Modal app: images, volumes, stages, entrypoints
src/rfd3_pipeline/         everything testable, with no Modal dependency
  config.py                the campaign schema and the model capability table
  gates.py                 every threshold, declared once
  filters.py  report.py    backbone filters; shards and summaries
  analysis.py              reading a finished campaign
  metrics/                 rmsd, ipsae
experiments/               probes and one-off tools, each documenting a trap
tests/                     508 tests, no GPU, no network
runs/                      one record per campaign
Notes.md                   gotchas by step, dated, marked verified or not
development/               dated work logs
```

`app.py` keeps its imports inside function bodies so the module stays importable
without Modal's dependencies. A module-level helper therefore does **not** see a
caller's function-local imports — that cost a campaign once.

## Tests

```bash
python -m pytest tests/ -q        # 508, ~3 s
```

No GPU and no network: `src/` holds the logic and `app.py` holds the Modal
wiring, which is what keeps the suite fast enough to run on every change.

## Two environment traps

**`PYTHONPATH=src`.** Every image ends with
`add_local_python_source("rfd3_pipeline")`, which imports the package *locally*
at build time. It lives under `src/` and is not pip-installed, so without this
every entrypoint fails with `ModuleNotMountable('rfd3_pipeline has no spec')`
before reaching Modal at all.

**`PYTHONIOENCODING=utf-8` on Windows.** The Modal CLI prints a `✓` that a
cp1252 console cannot encode, and the resulting `charmap` traceback looks like a
failed run when the function actually succeeded.

## Where the knowledge lives

- [`Notes.md`](Notes.md) — gotchas by campaign step, each dated and marked
  verified or unconfirmed. **Separates what held across both campaigns from
  what did not**: pass rates swing ±8 points between runs, so a single run's
  number is not pipeline behaviour.
- [`runs/`](runs/) — one record per campaign, fixed section order so two can be
  read side by side.
- [`boltz2_plan.md`](boltz2_plan.md) — adding a third folding model, verified
  against the kit source before any code.
- `modal_rfd3_pipeline_plan*.md` — the original design, now wrong in places the
  logs correct. Prefer `Notes.md`.

## Third party

`LigandMPNN/` is vendored rather than cloned, with the minimal fixes needed to
run on Python 3.12 and numpy 2.x — upstream pins `numpy==1.23.5`, which has no
cp312 wheels. The ESMFold2 and AlphaFold3 inference kits live outside this
repo.
