---
name: RFD3 Modal pipeline (revised)
overview: "Build a new Modal pipeline from scratch for protein design: one campaign YAML drives RFD3, backbone filters, SolubleMPNN, structure prediction, then cross validation structure prediction with five seeds, and ranks by cross-model ipSAE."
---

# RFD3 pipeline on Modal

Revision of `modal_rfd3_pipeline_plan.md`. Same pipeline, same metrics, same
gating. What changed: commands that would not have parsed, two config keys whose
names collided with real upstream flags, the MSA cache (which as originally
sketched would have re-queried ColabFold on every prediction), and the
orchestration shape, which did not survive contact with the fan-out this
campaign actually needs.

Everything asserted about upstream behaviour below is cited to a file and line
in the repo it came from, so it can be re-checked when those repos move.

One campaign YAML is the source of truth. Modal volume `rfd3_pipeline` holds all
run data; weights live on a separate volume. The pipeline runs detached — no
local computer in the loop after launch — and can restart at any step.

```
flowchart
  yaml[Campaign YAML]
  msa[Target MSA cache]
  rfd3[RFD3 backbones]
  bb[Backbone filters]
  mpnn[SolubleMPNN]
  apo[binder only structure prediction]
  holo[complex structure prediction]
  cv[cross-validation x5 seeds default]
  rank[Rank and label clusters]
  yaml --> msa
  yaml --> rfd3 --> bb --> mpnn --> apo --> holo --> cv --> rank
  msa --> holo
  msa --> cv
```

The MSA cache is off the main chain: it depends only on the target sequences, so
it runs once at campaign start, in parallel with RFD3, and both holo and
cross-validation read it.

Sources:
- RFD3: https://github.com/RosettaCommons/foundry/tree/production/models/rfd3
- MPNN: https://github.com/dauparas/LigandMPNN
- ESMFold2, Boltz2, AF3 kits: https://github.com/anthropics/uplifting-biomolecular-modeling
- AF3 fork (weights + MSA server): https://github.com/sokrypton/alphafold3
- Prior art to lift from: https://github.com/KrajTechne/Optimus

---

## Changes from the original plan

| # | Original | Revised | Why |
|---|---|---|---|
| 1 | `--n-batches 8`, diffusion batch size as a flag | `n_batches=8 diffusion_batch_size=8` | RFD3 is Hydra. There are no `--` flags and no dashes in key names. `input.md:65-66` |
| 2 | `length: 1050-1150 # binder only, need to add target lengths` | `length` derived from contig, then asserted | The value was already correct and already total: (120-220) + 465 + 465. The comment was wrong. `length` and `contig` are redundant, so a disagreement is a silent-wrong-result. `input.md:127` |
| 3 | `step_scale=3`, `gamma_0=0.2` | upstream defaults (`1.5` / `0.6`), both exposed in YAML | Default `step_scale` is 1.5 and the config annotates the useful band as "1.5 - 1.0", so 3 was 2x the top of that; `gamma_0` default is 0.6. Start at stock, tune deliberately. `rfdiffusion3.yaml:45,48` |
| 4 | `n_gpu: 4` meaning Modal concurrency | `max_concurrent` (Modal) + `n_gpu_shard` (kit) | `--n_gpu P` in the kits splits **one** prediction across P GPUs. Same name, opposite meaning. `esmfold2/README.md:9`, `af3_jax/README.md:11` |
| 5 | `model: esmfold2` | `model: esmfold2` or `esmfold2-fast`; `--variant` derived | `--variant` conflates model choice with MSA presence. Naming the model in the YAML lets the orchestrator derive the variant per step, so no field can contradict the step's MSA policy. `esmfold2/README.md:72-73,103,107` |
| 6 | `model: af3` | `model: openfold3` or `af3_native` | The Anthropic kit ships only OpenFold3 ported weights (`--variant p2`). Official AF3 weights are a different path with different licence terms. `af3_jax/README.md:4,63` |
| 7 | contigs + unindex as two parallel lists | one `specs:` list of objects | `contigs[i]` paired with `unindex[i]` by position is a bug waiting to happen; also there was no place to put `select_fixed_atoms`. |
| 8 | (absent) | `select_fixed_atoms` per spec | For unindexed motifs only the atoms marked fixed are carried over from the input. `ALL` vs `BKBN` vs `TIP` changes the designs. `input.md:151,215` |
| 9 | MSA cache "define how you are going to implement this" | one hybrid client (Optimus auth/modes/tarball + fork parsing/pairing), content-addressed cache, never `--use_msa_server` | See [MSA cache](#msa-cache). `--use_msa_server` is not cached despite its help text, and each existing client has what the other lacks. |
| 10 | ipSAE "PAE cutoff 10, distance cutoff 10" | passed explicitly | Optimus defaults are 15/15, so the spec would have been silently ignored. `StrucTools.py:390,412` |
| 11 | separate Modal apps per step, orchestrator calls in order | one app, orchestration in a remote function | Cross-app lookups give version skew; a local orchestrator cannot be detached. |
| 12 | `summary.csv`, "avoid concurrent appends" | per-item shard files + reduce step | Volumes are not POSIX-consistent under concurrent writers. Locking does not fix this. |
| 13 | (absent) | JIT/compile cache volume + warm at build | Kits compile Triton kernels on first run per machine. On ephemeral containers that is paid per container, ~2500 times. `esmfold2/README.md:74-79` |
| 14 | (absent) | `dry_run` profile, GPU-seconds accounting | Needed before scaling to a real campaign. |

---

## Architecture

**One Modal app, not several.** `modal.App("rfd3-pipeline")`, with a separate
`modal.Image` per step. Images are per-function, so nothing is lost by
consolidating, and consolidating avoids `Function.from_name` lookups across
independently-deployed apps, where the orchestrator can silently call a stale
version of a step.

**Orchestration runs remotely.** This is what makes the run detachable:

```python
@app.function(image=orchestrator_image, timeout=24 * 60 * 60,
              volumes={DATA: data_volume})
def orchestrate(campaign_yaml: str, start_at: str = "rfd3", dry_run: bool = False):
    ...   # calls each step with .map(), writes state.json between steps
```

Launch either as `modal run --detach app.py::main` (thin local entrypoint that
only calls `orchestrate.spawn()`) or, once deployed, `orchestrate.spawn()` from
anywhere. The orchestrator itself needs no GPU.

Optimus does this differently and it is worth being explicit about why not to
copy it: `modal_run_refiner.py:171` makes `refiner()` an
`@app.local_entrypoint()` that calls `.remote()` and then runs
`_run_af3_validation` locally. That works for one design, but the laptop is part
of the control flow, so closing it ends the run. It also does not fan out: the
whole refinement loop lives in one GPU container.

**Fan-out with `.map()`, not a loop of `.remote()`.** The scale demands it:
5 specs x 8 batches x 8 = 320 backbones, ~8 sequences each after filters, so
order 1000-2500 refolds, then x5 seeds for cross-validation.

**But map over *chunks*, not individual sequences.** Every folding kit pays a
fixed per-call cost — Boltz2's README measures it at roughly 25 s of start-up,
checkpoint load and featurization *in every mode*, and notes the gain grows "with
several inputs or seeds per call" (`boltz2/README.md:144-146`). At ~2500
sequences, one container per sequence is on the order of 17 GPU-hours of pure
start-up, paid before any prediction happens.

All three folding paths accept multiple inputs per call — esmfold2 takes a
multi-complex input JSON (`esmfold2/README.md:84`), boltz2 runs "every input at
every seed in one worker" (`boltz2/README.md:91`), and the AF3 fork takes
`--input_dir` (`run_alphafold.py:105`). So:

```python
CHUNK = 24   # tune in the dry run
chunks = [ids[i:i + CHUNK] for i in range(0, len(ids), CHUNK)]
results = fold_step.with_options(gpu=cfg.gpu).map(chunks)
```

Size the chunk so a container runs roughly 10-30 minutes: long enough to amortize
the load, short enough that a preemption loses little and `skip_existing` picks up
cheaply. `max_concurrent` from the YAML caps breadth. The reduce step is
unaffected — each item still writes its own shard.

**Per-step GPU from YAML** via `.with_options(gpu=...)`, which rebinds per call
without redefining the function. Optimus already does this
(`modal_run_refiner.py:210`) and it is the right pattern — it also means
`mode: big` with `n_gpu_shard: 2` can be dispatched to a `"A100-80GB:2"`
function variant at call time.

### Volumes

| Volume | Mount | Holds |
|---|---|---|
| `rfd3_pipeline` | `/data` | campaign runs, shards, summaries, MSA cache |
| `rfd3_weights` | `/weights` | model weights, one subfolder per model |
| `rfd3_jit` | `/jit` | Triton / XLA compile caches (see below) |

One weights volume, **one subfolder per model, each created only when that model
is first needed** — so a campaign that never runs Boltz2 never pays for its
checkpoint:

```
/weights/
  rfd3/           foundry install rfd3 --checkpoint-dir /weights/rfd3
  esmfold2/       run.sh install --weights ...   (HF_HOME layout; ~24 GiB)
  boltz2/         boltz's own cache layout
  af3_jax/p2/     of3_ported_weights.bin.zst     (~2.3 GB fetched + converted)
  ligandmpnn/     get_model_params.sh
```

Populating it is **its own Modal function**, not a side effect of a step: the
installers run inside containers, so `populate_weights(models=[...])` runs once,
commits, and is idempotent — each installer hash-checks an existing directory
rather than refetching (`af3_jax/README.md:53`, `esmfold2/README.md:61`), so a
complete directory can be mounted read-only afterwards.

RFD3's installer sets `FOUNDRY_CHECKPOINT_DIRS` to the directory it was given
(`models/rfd3/README.md:26-30`).

No Modal secrets, unless the MSA client is pointed at an authenticated endpoint.

**Call `volume.commit()` at the end of every step**, and periodically inside long
ones. Writes are not visible to other containers until committed. This is the
single most common way a multi-step Modal volume pipeline breaks: step N+1 sees
an empty directory and reports zero inputs rather than failing.

### Compile caches

The kits compile Triton kernels on first run per machine, and keep them under
`MODEL_OPT_JIT_ROOT` (`esmfold2/README.md:74-79`). Containers are ephemeral, so
without a persistent root every one of ~2500 predictions pays a cold compile.
Two mitigations, use both:

- Mount `rfd3_jit` and set `MODEL_OPT_JIT_ROOT=/jit/<kit>`.
- Run `bash run.sh warm --config <card> --variant <v> --mode <m>` at **image
  build time** so the cache ships inside the image.

For the AF3 fork the equivalents are `--cache_dir` (JAX compile + tokamax
autotune, `run_alphafold.py:364-370`), `--precompile=<token counts>` with
`--buckets` to move the cold compile to install time
(`run_alphafold.py:563-574`), and `--lowercache_dir` for serialized executables,
which additionally skips trace-and-lower — measured at 10.75 s of a 24.22 s
first fold (`run_alphafold.py:551-562`).

**Bucketing matters more here than in a typical run**, because compiled kernels
are keyed by input shape as well as by GPU. Binder lengths vary across
`120-220`, so complexes land anywhere in ~1050-1150 tokens: left alone, that is
close to one distinct shape per design and therefore a recompile per design,
which defeats the cache entirely. Set `--buckets` to a small fixed ladder
covering the complex size range, and `--precompile` those same token counts at
image build. Then every prediction hits a warm shape.

### Restart

`state.json` per run, holding per-step status **and per-item completion** — so a
restart re-runs only missing designs, not the whole step. Each step is
idempotent: the output path is a deterministic function of the item ID, and the
step skips any item whose output already exists.

RFD3 gives this away free: `skip_existing` defaults to `True` and skips any
system whose output files already exist in `out_dir` (`input.md:72`). Build on
that rather than inventing a step-level flag. `--restart-from apo` then means
"run from apo and let idempotency skip finished work", and it validates that the
prior step's outputs exist before starting.

`summary.csv` and `summary.json` are **derived artifacts**, regenerated by the
reduce step from shards. They are never the source of truth, so a crash
mid-write costs nothing.

---

## Layout

```
/data/{campaign}/
  msa_cache/                        # content-addressed; campaign-wide, never auto-invalidated
    {key}_raw.tar.gz                # server response, kept for re-derivation
    {key}_unpaired.a3m
    {key}_paired.a3m
    index.json                      # key -> {sequence, chain_id, length, mode,
                                    #         pairing_strategy, host_url, created_utc}
  {run_name}/
    state.json                      # step + per-item status, for restart
    campaign.resolved.yaml          # after schema validation + derived fields
    rfd3/
      specs.yaml                    # all spec keys, one file
      out/                          # *.cif.gz + *.json per design
    filters/shards/{design_id}.json
    mpnn/{design_id}/seqs.fa
    apo/{seq_id}/
    holo/{seq_id}/
    cv/{seq_id}/seed_{n}/
    shards/{seq_id}.json            # one metric record per MPNN sequence
    summary.csv
    summary.json
```

---

## Campaign YAML

Validated with a Pydantic schema at orchestrator start, before any GPU is
allocated. Fail fast on a bad campaign file rather than 40 minutes into RFD3.
The resolved config is written back to `campaign.resolved.yaml` so a run records
exactly what it ran with, including derived fields.

```yaml
general:
  chain_binder: A
  chain_targets: [B, C]          # list, not a comma string
  name_design_campaign: tpo
  run_name: tpo_01

rfdiffusion3:
  gpu: A100-80GB
  max_concurrent: 5              # Modal containers; was `n_gpu`
  path_input_structure: in/rfdiffusion/tpo_alphafold.pdb

  # One object per spec. Replaces the parallel contigs[]/unindex[] lists:
  # pairing by index was fragile and left nowhere to put select_fixed_atoms.
  # `name` is explicit, not derived from list position, so the RFD3 output key
  # for a spec is stable when specs are added, removed or reordered. Derived
  # names (tpo_spec_00, _01, ...) would shift on an insert, and a resumed run
  # would then match `skip_existing` against the wrong prior outputs.
  specs:
    - name: motif_full
      contig: "120-220,/0,B1-465,/0,C1-465"
      unindex: "A13-25,A53-60,A125,A141-149,A86,A106-111"
      select_fixed_atoms: {default: BKBN}    # see note below
    - name: motif_ext141
      contig: "120-220,/0,B1-465,/0,C1-465"
      unindex: "A13-25,A53-60,A125,A141-153,A86,A106-111"
      select_fixed_atoms: {default: BKBN}
    - name: motif_no86
      contig: "120-220,/0,B1-465,/0,C1-465"
      unindex: "A13-25,A53-60,A125,A141-149,A106-111"
      select_fixed_atoms: {default: BKBN}
    - name: motif_no86_no125
      contig: "120-220,/0,B1-465,/0,C1-465"
      unindex: "A13-25,A53-60,A141-149,A106-111"
      select_fixed_atoms: {default: BKBN}
    - name: motif_minimal
      contig: "120-220,/0,B1-465,/0,C1-465"
      unindex: "A13-25,A141-149,A106-111"
      select_fixed_atoms: {default: BKBN}

  hotspots: "B77,B79,B80,B101,B103,B139,C74,C77,C79,C80,C135,C139"
  infer_ori_strategy: hotspots
  is_non_loopy: true
  dialect: 2
  length: null                   # DERIVED from contig + input; see below

  n_batches: 8                   # RFD3 `n_batches`
  diffusion_batch_size: 8        # RFD3 `diffusion_batch_size`
  # Sampler knobs. All three are upstream defaults, exposed here so a future
  # run can tune them without a code change. Higher step_scale and lower
  # gamma_0 both trade diversity for designability (input.md:69,98).
  num_timesteps: 200             # RFD3 default
  step_scale: 1.5                # RFD3 default (was 3 in the original plan)
  gamma_0: 0.6                   # RFD3 default (was 0.2 in the original plan)

  threshold_interresidue_clashes_sidechain: 3
  threshold_interresidue_clashes_backbone: 0
  threshold_loop_fraction: 0.5
  threshold_radius_gyration_multiplier: 1.1
  threshold_hotspot_contacts: 0.4

mpnn:
  gpu: L40S
  max_concurrent: 8
  mpnn_sampling_temp: 0.1
  num_seqs: 8
  rm_aa: C
  use_solubleMPNN: true

refolding:
  model: esmfold2                # esmfold2 | esmfold2-fast | boltz2 | af3_jax
                                 # --variant is derived per step, not set here
  mode: fast                     # off | exact | fast | big
  gpu: A100-80GB
  kit_config: a100               # must match `gpu`; validated as a pair
  n_gpu_shard: 1                 # only >1 under mode: big; 2|4|8
  max_concurrent: 4              # was `n_gpu`
  target_template: false
  threshold_plddt_binder: 70
  threshold_rmsd_apo_holo: 4.0
  threshold_rmsd_motif: 2.0
  threshold_hotspot_contacts: 0.4
  threshold_ipTM: 0.3

cross-validation:
  model: openfold3               # openfold3 | af3_native | boltz2
  variant: p2
  mode: exact
  gpu: H100
  kit_config: h100
  n_gpu_shard: 1
  max_concurrent: 4
  target_template: false
  num_seeds: 5

msa:
  host_url: https://api.colabfold.com
  use_env: true                  # env | all
  use_filter: true               # false -> *-nofilter modes
  pairing_strategy: greedy       # greedy | complete
  # Unauthenticated against the public endpoint. Both auth paths stay plumbed
  # through so moving to a private/self-hosted MMseqs2 server is config-only.
  # Setting either is the one case that needs a Modal secret.
  username: null
  password: null
  auth_header: null

ipsae:
  pae_cutoff: 10                 # passed explicitly; Optimus defaults to 15
  dist_cutoff: 10

run:
  dry_run: false
```

### `length` is derived, not entered

`length` is the **total** design length (`input.md:127`). For the contig above:
`120-220` designed, plus `B1-465` and `C1-465` from the input, giving 1050-1150.
The original plan's number was already right; only its comment was wrong.

Because `length` and `contig` encode overlapping information, compute `length`
from the contig plus the input structure and assert agreement if the user also
supplied one. A hand-typed mismatch here produces designs of the wrong size with
no error.

The contig grammar, for the parser: comma-separated components; a dash is a
range; `/0` is a chain break; a leading chain label means "from the input
structure"; no chain label means "design this many residues, uniformly random
within the range" (`input.md:189-207`).

### `select_fixed_atoms` is a real decision

Unindexed atoms are always fixed unless `select_fixed_atoms` says otherwise, at
least one atom per unindexed residue must be fixed, and **only the atoms marked
fixed are carried over from the input** (`input.md:151,215`). So `ALL` / `BKBN` /
`TIP` materially changes what is being scaffolded. The original plan was silent
on this, which means it would have taken the default without anyone choosing it.
For hotspot-directed binder design against a protein target, `BKBN` per
unindexed residue is the usual starting point, with `TIP` where a sidechain
contact is the point of the motif. Set it per spec and record it.

### Things to verify before a production run

- **Sampler settings start at stock.** `step_scale: 1.5`, `gamma_0: 0.6`,
  `num_timesteps: 200` are all RFD3 defaults, exposed in the YAML so a later run
  can tune them without touching code. The original plan's `step_scale=3` /
  `gamma_0=0.2` came from "the Foundry binder examples", but
  `models/rfd3/docs/protein_binder_design.md` 404s at the path its own README
  links, so neither value could be verified. Both move in the
  designability-over-diversity direction (`input.md:69,98`), which is plausible
  for binder design — so treat them as a deliberate A/B against stock on a small
  batch, not as the starting point.
- **The 5 contigs are identical**; only `unindex` differs. Presumably deliberate
  (same length envelope, progressively fewer motif residues) — worth confirming
  it is not a copy-paste artifact.
- **Trailing `/0,/0`** on the original `unindex` strings has been dropped here.
  Breaks between unindexed components are inferred and logged
  (`input.md:222`), so the trailing breaks were at best inert. Run with
  `prevalidate_inputs=True` to confirm.
- **The YAML above is one example, not the shape of every campaign.** The number
  of target chains varies and nothing may assume two. But it is known from the
  config, not discovered later: **the count is `len(chain_targets)`, and output
  chain IDs run `A` for the binder then `B`, `C`, ... for the targets in contig
  order**, because RFD3 renumbers output chains deterministically by contig
  position. So the column set is fixed once the YAML is resolved — see
  [ipSAE columns](#ipsae-columns).

  What is *not* known from the config is sequence **identity**: any subset of
  target chains may share a sequence, and that is read from the input structure
  at runtime. It affects MSA pairing only — see [MSA cache](#msa-cache).
- **`gpu` and `kit_config` must agree.** Kit modes print
  `NOT ACTIVE: <reason>` and **exit 3 rather than falling back**
  (`esmfold2/README.md:117`). Cards are `h100|h200|a100` only, so a folding step
  cannot be pointed at a GPU with no card. Validate the pair in the schema.
  (`L40S` for MPNN is fine — MPNN is not a kit.)

---

## Data storage and tracking

`summary.csv` — one row per MPNN sequence. Backbones that die before MPNN still
get a row with empty sequence columns. `summary.json` — per-step counts and
per-metric failure counts, schema unchanged from the original plan.

### summary.csv columns

Prefixed by stage so the funnel reads left to right. `{T}` expands once per
target chain, from the resolved `chain_targets` (see
[ipSAE columns](#ipsae-columns)).

**Identity and outcome** — always populated

`campaign`, `run_name`, `spec_name`, `design_id`, `seq_id`, `mpnn_index`,
`stage_reached`, `passes_filters`, `fail_stage`, `fail_reasons`

`stage_reached` is one of `rfd3 | backbone_filters | mpnn | apo | holo | cv` and
records how far the design got. It makes the funnel a `value_counts()` rather
than a join, and makes "where did everything die" answerable at a glance.

**Backbone** — populated for every design, including ones that die here

`bb_binder_length`, `bb_clashes_backbone`, `bb_clashes_sidechain`,
`bb_loop_fraction`, `bb_rg`, `bb_rg_expected`, `bb_rg_ratio`,
`bb_hotspot_fraction`, `bb_n_hotspots_hit`, `bb_n_hotspots_total`, `bb_pass`,
`motif_map`, `mpnn_fixed_residues`

`motif_map` is the compact form of the binder-chain `diffused_index_map`
(`A77>A34;A78>A35;A79>A36`) and `mpnn_fixed_residues` is the literal argument
handed to MPNN (`A34 A35 A36`). Both exist so the input→output numbering
translation is auditable from the CSV rather than only from code.

**MPNN** — empty for backbones that died earlier

`sequence`, `seq_length`, `mpnn_temp`, `mpnn_seed`

**Apo**

`apo_plddt_binder`, `apo_pass`

**Holo**

`holo_plddt_binder`, `holo_iptm`, `holo_ptm`, `holo_ipsae_min_{T}`,
`holo_ipsae_min`, `holo_max_ipsae_min`, `holo_rmsd_motif_on_target`,
`holo_rmsd_binder_on_target`, `holo_rmsd_binder_apo_holo`,
`holo_paratope_length_{T}`, `holo_epitope_length_{T}`, `holo_epitope_recall_{T}`,
`holo_epitope_precision_{T}`, `holo_epitope_f1_{T}`, `holo_epitope_indices_{T}`,
`holo_paratope_indices_{T}`, `holo_pass`

**Cross-validation** — aggregates across seeds only

`cv_model`, `cv_n_seeds`, and for each of
`plddt_binder`, `iptm`, `ipsae_min_{T}`, `max_ipsae_min`, `rmsd_motif_on_target`,
`rmsd_binder_apo_holo`: a `_mean`, `_max` and `_sd` column.

The `_sd` columns are cheap and carry real signal — a design whose ipSAE varies
widely across seeds is a different proposition from one that scores the same
every time, even at equal means.

**Ranking** — written by the final reduce pass

`rank_holo_ipsae`, `rank_cv_ipsae`, `cluster_id`, `cluster_size`

### cross_validation_seeds.csv

Per-seed detail lives in its own file, one row per `(seq_id, seed)`:

`seq_id`, `seed`, `plddt_binder`, `iptm`, `ptm`, `ipsae_min_{T}`,
`max_ipsae_min`, `rmsd_motif_on_target`, `rmsd_binder_apo_holo`

Keeping it separate holds `summary.csv` at a fixed width: inlining five seeds
would add ~30 columns *and* change the schema whenever `num_seeds` changes.
Both files are regenerated from the same shards by the reduce step.

**How concurrent writes are avoided:** they are not attempted. Each worker
writes `shards/{seq_id}.json`; a CPU-only reduce function globs the shards and
rewrites `summary.csv` and `summary.json` at the end of each step. Modal volumes
are not POSIX-consistent under concurrent writers, so append-with-lock corrupts
rows rather than serializing them. This also makes both files regenerable after
a crash, and makes `--restart-from` cheap.

Add to `summary.json`, beyond the original schema:

```json
"cost": {
  "rfd3":    {"gpu_seconds": 0, "gpu": "A100-80GB", "n_containers": 0},
  "apo":     {"gpu_seconds": 0, "gpu": "A100-80GB", "n_containers": 0},
  "holo":    {"gpu_seconds": 0, "gpu": "A100-80GB", "n_containers": 0},
  "cross_validation": {"gpu_seconds": 0, "gpu": "H100", "n_containers": 0}
},
"survivors": {"rfd3": 320, "backbone_filters": 0, "mpnn": 0, "apo": 0, "holo": 0}
```

The survivor counts are the load-bearing numbers: the apo pLDDT gate and then
the holo gate are what keep the x5-seed cross-validation affordable, so they
should be visible at a glance rather than derived from the CSV.

### `dry_run`

`run.dry_run: true` overrides to 1 spec, `n_batches=1`,
`diffusion_batch_size=2`, `num_seqs=2`, `num_seeds=1`, `model: esmfold2`,
`mode: fast`. Exercises every step and both reduce passes in roughly ten
minutes. Run this after any change to the spec generator or the renumbering
logic, before spending A100-hours.

---

## RFD3

Multiple specs are written to one RFD3 input file, one key per spec:
`tpo_spec_00`, `tpo_spec_01`, ... Shared fields (`input`, `select_hotspots`,
`infer_ori_strategy`, `is_non_loopy`, `ligand` if set, `dialect: 2`) are copied
onto every key. `hotspots` in the campaign YAML becomes `select_hotspots`
(`input.md:136`).

Validation before launch: unindex residues must exist in the input, and must not
overlap that spec's contig. RFD3 checks the overlap case itself and errors early
(`input.md:156`), and `prevalidate_inputs=True` checks the whole input file
before loading checkpoints (`input.md:75`) — cheap, so always on.

### Command

Hydra `key=value`, no dashes, no `--`:

```bash
rfd3 design \
  out_dir=/data/tpo/tpo_01/rfd3/out \
  inputs=/data/tpo/tpo_01/rfd3/specs.yaml \
  json_keys_subset='[tpo_spec_00]' \
  global_prefix='' \
  n_batches=8 \
  diffusion_batch_size=8 \
  inference_sampler.num_timesteps=200 \
  inference_sampler.step_scale=1.5 \
  inference_sampler.gamma_0=0.6 \
  ckpt_path=rfd3 \
  skip_existing=True \
  prevalidate_inputs=True \
  dump_prediction_metadata_json=True
```

`n_batches=8` x `diffusion_batch_size=8` = 64 designs per spec, 320 across five
specs, matching the original plan's `summary.json` (`input.md:65-66`).

### Sharding

RFD3 supplies the two mechanisms needed to fan out across containers:

- `json_keys_subset` runs only a subset of the input file's keys
  (`input.md:80`), so each container takes one spec key from the one shared
  `specs.yaml`.
- `global_prefix` renames outputs from the input-file basename to a string of
  your choosing, and the config annotates the chunked form as
  "(pipelines usage)" (`rfdiffusion3.yaml:58-63`) — it exists for exactly this.

So: `.map()` over the spec keys, `max_concurrent` containers, each writing into
the shared `out_dir` under a distinct prefix. With `skip_existing=True` a
re-run after a preemption regenerates only what is missing.

### Motif mapping

Extract motif positions from the per-design JSON (`dump_prediction_metadata_json`
is on by default, `input.md:105`), key `diffused_index_map`: input-structure
residue on the left, output-structure position on the right.

```json
"diffused_index_map": {"A13": "A125", ...}
```

Keep only entries whose **value** is on the binder chain (`chain_binder: A`);
ignore the target-chain mappings. Store the mapping per design for MPNN fixed
residues and for motif RMSD.

`diffused_index_map` is ground truth for motif positions. Contig arithmetic is a
fallback only, because the contig refers to the input structure while the map
describes the actual output. Where both are available, assert they agree — a
silent off-by-one here invalidates every downstream RMSD and contact count.

**Hotspot renumbering** has no equivalent map, so it must come from contig
arithmetic: the output structure starts at 1 and orders chains by contig
position, so a target hotspot's new chain and index are computed from where its
input chain lands in the contig.

(ligands-the-midas-metal)=
### Ligands: the MIDAS metal

`alphav_beta3_fib10_clean.pdb` holds five chains, not three: A/B/C protein plus
four Mn(2+) ions, one on chain D and three on chain E. A contig naming only B and
C drops all of them.

Measured coordination in the input (heavy atoms within 3.0 A, and distance to the
RGD aspartate A79 carboxylate):

| Metal | Coordinated by | To RGD Asp79 | Identity |
|---|---|---|---|
| E703 | A79(ASP), C10(SER), C12(SER), C109(GLU) | **2.12 A** | MIDAS |
| E704 | C12, C15(ASP), C16(ASP), C140(ASP) | 7.71 A | ADMIDAS |
| E705 | C47, C104, C106, C108, C109 | 5.90 A | SyMBS |
| D1004 | B139, B141, B143, B145, B147 | 41.42 A | alphaV propeller |

E703 is the one that matters: the RGD aspartate completes its coordination
sphere at bonding distance, alongside the DXSXS serines of the beta3 betaI
domain. Scaffolding RGD without it omits the motif's principal interaction.
D1004 is 41 A away on the other subunit and is simply excluded.

**Two constraints from the Foundry source**, both of which answer the obvious
questions:

- **Ligands do not affect `length`.** `length` is optional and documented as
  constraining the *contig* (`input_parsing.py:141`), and `_append_ligand` runs
  after the contig is built (`input_parsing.py:516`). Adding metals changes
  nothing about length — and since the contig already determines the polymer,
  the cleanest course is to omit `length` from the emitted spec entirely and keep
  the derived value as a validation assertion.
- **Each ligand residue must sit on its own chain.** `_append_ligand` raises on
  more than one ligand residue per chain (`input_parsing.py:696-706`). E703,
  E704 and E705 all share chain E, so asking for the full triad against this file
  fails. The escape hatch `allow_ligand_on_existing_chain: true` exists but its
  own docstring warns that chain ID is leaked to the model, and it re-enables
  legacy residue renumbering — not worth it.

Naming is flexible: `ligand` is split on commas and each component may match
several instances (`inference.py:105-108`), so `"MN"` would select all four —
and then fail the per-chain check. `ligand: "E703"` selects one residue on one
chain and passes as the file stands.

To use the full beta3 triad, split E703/E704/E705 onto separate chains (E, F, G)
in the input PDB and set `ligand: "E703,E704,E705"`. Whether the two secondary
metals earn that edit is a modelling judgement: they shape the MIDAS site's
geometry and electrostatics but do not contact the binder.

Ligands default to a fully fixed motif (`set_defaults` with `motif=True`,
`inference.py:113-116`), which is the right treatment for a structural metal, and
`select_unfixed_sequence` excludes them by construction
(`input_parsing.py:165`).

### `rfd3_spec` — a GPU-free, unit-tested module

Spec generation, contig parsing, `length` derivation, hotspot renumbering and
`diffused_index_map` extraction are all pure functions over text and structure
files. They go in one module with **no GPU, no Modal, and no network**, tested
locally against a checked-in example RFD3 output.

This is the highest-risk logic in the pipeline and the cheapest to get right. A
silent off-by-one in renumbering invalidates every hotspot contact count, every
motif RMSD, and therefore the ranking the whole campaign exists to produce — and
it produces no error, just plausible wrong numbers. Tests worth having:

- Contig round-trip: parse `120-220,/0,B1-465,/0,C1-465`, derive
  `length == "1050-1150"`, and assert against a hand-checked value.
- Hotspot renumbering against a real output: for the checked-in design, map each
  input hotspot to its output chain/index and compare to positions read directly
  out of the output structure.
- Cross-check: where `diffused_index_map` and contig arithmetic both cover a
  residue, assert they agree. Any disagreement fails the build.
- Rejection cases: unindex residue absent from the input; unindex overlapping the
  contig; `length` supplied and contradicting the contig.

Keep the example output small (one spec, one design) so it can live in the repo.

---

## RFD3 filters

Applied to the binder chain before MPNN, reading the per-design JSON that shares
a name with each structure file. Thresholds come from the YAML; defaults below.
Fail reasons go into `summary.json`.

- Backbone inter-residue clashes `<= 0`
- Side-chain inter-residue clashes `<= 3` (soft relative to backbone; MPNN
  replaces side chains)
- Loop fraction `<= 0.5`
- Binder Rg, computed with biotite on the binder chain, since the number in the
  JSON is for the whole complex: `Rg <= 1.1 * 2.38 * N^0.365`, where `N` is the
  binder residue count and the multiplier comes from the YAML. Exponent stays
  `0.365`. Store `rg / rg_expected`.
- Hotspot contact fraction `>= 0.4`

**Hotspot contact fraction, stated once and unambiguously** — the original
phrasing ("fail if less than `threshold_hotspot_contacts` of hotspots have zero
contacts") inverted itself:

> Of the hotspots listed in the YAML, the fraction having **at least 1 heavy-atom
> contact within 4.5 Å** of the binder chain must be **>= 0.4**.

This is exactly `epitope_coverage_recall` from
`determine_binding_interface` — "what fraction of the desired epitope residues
were hit", at a 4.5 Å heavy-atom cutoff (`StrucTools.py:317`). One
implementation serves both the backbone filter and the holo interface metrics;
do not write a second.

The same call also returns `precision` (how much of the actual epitope fell
outside the hotspot list) and `f1`. Neither gates, but both are recorded —
precision is the off-target-spillover signal and is often the more informative of
the two when ranking otherwise-similar designs.

Interface contacts go into `summary.csv`.

Reuse `StrucTools.determine_binding_interface(pdb_file_path, hotspots,
binder_chain_id, target_chain_id)` from Optimus for this. Optimus derives target
chains as `chr(ord('B') + i)` (`StructurePredictionInputs.py:70`), which matches
this pipeline's output convention — so the derivation is right and only two
things change: the binder is taken from `chain_binder` rather than hardcoded to
`"A"`, and the chain count comes from the resolved `chain_targets` rather than
from `len(seq_list) - 1`. Both come from the same resolved config that drives the
[ipSAE columns](#ipsae-columns).

---

## MPNN

Only structures that pass the backbone filters. Temperature `0.1`,
`num_seqs: 8`, `omit_AA: C`. No filter at this step.

`mpnn_type` selects the model — `soluble` (the default), `protein` or `ligand` —
rather than the boolean `use_solubleMPNN`, so adding a ligand campaign later is a
config change rather than a schema change.

`--fixed_residues` is **derived, never configured**: take the binder-chain entries
of `diffused_index_map`, keep the values (output numbering), and emit them
chain-prefixed and space-separated, e.g. `A34 A35 A36`. This is the one place
where using input numbering instead of output numbering would silently fix the
wrong residues and produce sequences that discard the motif — so the conversion
lives in `rfd3_spec` with the rest of the numbering logic, and the emitted list
is recorded per design in `summary.csv` for audit.

`.map()` over passing design IDs. MPNN is seconds per design even at ~1150
residues, and Optimus notes it runs acceptably on CPU torch
(`modal_common.py:194`), so `gpu: L40S` is generous — consider CPU-only with
higher `max_concurrent` and measure in the dry run.

---

## MSA cache

Depends only on the target sequences, which do not change across the campaign,
so it runs **once** per campaign and is shared by holo and cross-validation.
Stored in `/data/{campaign}/msa_cache/`.

### Do not use `--use_msa_server`

The AF3 fork's `--use_msa_server` flag says "Results are cached in `--cache_dir`
so subsequent runs skip the network" (`run_alphafold.py:582-588`). **That is not
what it does.** `run_alphafold.py:1728-1734` calls `fill_missing_msas` on every
fold input on every invocation, with no cache lookup; `--cache_dir` is used only
for the JAX compile and tokamax autotune caches (`run_alphafold.py:364-370,700`).
`save_msas` at `:1417` writes the a3m files into the *output* directory
afterwards, which is a record of what was used, not a cache.

Left on, this pipeline would POST to `api.colabfold.com` once per prediction —
order 1000+ submissions for holo and cross-validation — hammer the rate limit,
and send every de novo binder sequence to a third party. Keep it **off** and
pre-populate instead.

### Build the cache: one client, taken from both

Neither existing implementation is a superset of the other. Optimus's
`mmseqs2.py` (ColabFold's `run_mmseqs2`) has the operational features; the AF3
fork's `msa_server.py` has the parsing correctness. Write one
`msa_client.py` that takes each side's strengths.

**From Optimus's `mmseqs2.py`:**

- **Authentication.** `msa_server_username`/`msa_server_password` for HTTP basic
  auth, or `auth_headers` for an API key, mutually exclusive and validated
  (`mmseqs2.py:28-56`). The fork has none — only `host_url` and `user_agent`
  (`msa_server.py:29-30`). This is the feature that matters most long-term: the
  public ColabFold endpoint is rate-limited and shared, and an authenticated or
  self-hosted MMseqs2 server is the path off it. Keep the parameters plumbed
  through from the start even while running unauthenticated, so switching is a
  config change and not a rewrite. (Credentials would be the one thing needing a
  Modal secret — the "no secrets" constraint holds only while the endpoint stays
  anonymous.)
- **Database/filter control.** `use_filter` selects among `env`, `all`,
  `env-nofilter`, `nofilter` (`mmseqs2.py:163-166`); the fork hardcodes `env`/
  `all` and cannot request an unfiltered search (`msa_server.py:43`).
- **Pairing strategy.** `greedy` or `complete` (`mmseqs2.py:170-176`); the fork
  hardcodes `pairgreedy` (`msa_server.py:40`).
- **Keeping the raw tarball.** Optimus writes `out.tar.gz` to disk and
  short-circuits on its existence (`mmseqs2.py:184,193`). The fork extracts into
  a `tempfile.TemporaryDirectory` and throws the response away
  (`msa_server.py:117`). Keep it: the tarball holds `uniref.a3m` and
  `bfd.mgnify30.metaeuk30.smag30.a3m` separately, so a different merge or depth
  can be re-derived months later without re-querying a database that will have
  moved on.

**From the fork's `msa_server.py`:**

- **Correct a3m block parsing.** It splits on `\x00` explicitly so hits from one
  query cannot bleed into another (`msa_server.py:137`), and appends a trailing
  newline so the last sequence of a block cannot be concatenated onto the next
  block's header (`:145`). Optimus instead strips `\x00` and tracks blocks with
  an `update_M` flag (`mmseqs2.py:274-283`), which is more fragile.
- **The merge fix, which is a genuine correctness difference.** When merging bfd
  hits onto uniref, the fork skips both the `>M` header *and* the query sequence
  line, with the reason recorded inline: a bare sequence line with no preceding
  header gets concatenated onto the previous hit by AF3's parser
  (`msa_server.py:156-160`). Optimus appends every line including the repeated
  query (`mmseqs2.py:283`). Since these MSAs are consumed by AF3, take the
  fork's version.
- **A usable download timeout.** The fork allows 60 s for the result tarball
  (`msa_server.py:107`); Optimus uses `6.02` s for the download as well as the
  submit (`mmseqs2.py:140`), which is too short for a large multi-MB tarball and
  will retry-loop on exactly the deep MSAs you most want.
- **Pairing actually wired through.** `_query_server(..., use_pairing=True)`
  works; Optimus's `run_mmseqs2` supports pairing but its only helper,
  `generate_msa`, hardcodes `use_pairing=False` (`mmseqs2.py:293-300`).
- **`FoldInput` integration.** `fill_missing_msas` and `save_msas` for the AF3
  path (`msa_server.py:165,272`).

**Fix in neither, added here: the cache key.** Optimus keys the cache on
`f"{prefix}_{mode}"` (`mmseqs2.py:179`), and its `generate_msa` passes the
caller's `msa_dir` as that prefix (`mmseqs2.py:293`). Two different sequences
sharing an `msa_dir` therefore resolve to the same cached tarball, and the second
silently receives the first one's MSA. Key on content instead:

```
cache_key = sha256(f"{sequence}|{mode}|{pairing_strategy}|{use_env}").hexdigest()[:16]
```

`index.json` stores, per key: the full query sequence (so a collision or a stale
entry is detectable rather than silent), chain ID, length, mode parameters, host
URL, and a UTC timestamp. The timestamp matters because the ColabFold databases
change over time — the same query re-run next quarter returns a different
alignment — so entries are **never invalidated automatically**. A campaign's
predictions stay comparable to each other, and a deliberate refresh is a new
cache key.

### The build step

One CPU-only Modal function, run once at campaign start, in parallel with RFD3:

1. Read target chain sequences from the input structure.
2. Compute `cache_key` per unique target sequence; skip any whose
   `{key}_unpaired.a3m` already exists.
3. One `_query_server` call for all missing targets — it deduplicates internally
   (`msa_server.py:46-53`), so a single call covers every target chain.
4. Deduplicate the target sequences and count the **unique** ones. If more than
   one, a second call with `use_pairing=True`; if exactly one, pairing does not
   apply and no call is made. `fill_missing_msas` itself skips pairing below two
   unique chains (`msa_server.py:215`), so matching that behaviour keeps the
   cache and the model consistent.

   This is decided at runtime from the input structure, never from the YAML. A
   campaign may have any number of target chains, and any subset of them may be
   identical — two chains of a homodimer collapse to one cache entry and no
   paired MSA, while two distinct chains need both. The same dedup makes the
   cache cheap: identical chains share one entry, and a chain already cached from
   an earlier campaign is skipped entirely.
5. Write the raw tarball, the parsed `_unpaired.a3m` / `_paired.a3m`, and the
   `index.json` entry; then `volume.commit()`.

Write each file to a temporary name and rename into place, so an interrupted
build cannot leave a truncated a3m that a later run treats as a cache hit.

### Injecting it, per model

The cache stores a3m text. Each model needs it in its own shape:

- **AF3 fork / OpenFold3.** MSAs are carried **inline** on the chain, as
  `unpaired_msa` and `paired_msa` a3m strings (`msa_server.py:247-255`), and the
  fork always runs with the data pipeline off (`af3_jax/README.md:99`). Set both
  fields from the cache. `fill_missing_msas` leaves any chain whose MSA is
  already set untouched (`msa_server.py:195,243-246`), so pre-populating makes it
  a no-op even if the flag is on — but leave the flag off regardless.
- **Boltz2.** Takes an a3m path per chain; point it at the cache file.
- **ESMFold2.** Selected by `--variant`, not by an MSA argument — see below.

### The binder must be explicitly suppressed

This is the trap. A de novo binder has no homologs, and its sequence is the
thing you least want to send to a third-party server. `fill_missing_msas` only
skips chains whose MSA is **already set** — leave the binder at `None` and it is
batched into both the unpaired query and, because it counts toward
`unique_protein_seqs`, the paired query as well (`msa_server.py:211-215`).

So for every prediction, set explicitly:

```python
unpaired_msa = f">query\n{binder_seq}\n"   # single-sequence
paired_msa   = ""                          # no pairing
```

which is the same convention Optimus expresses as `msa_options` `"empty"` for
the binder and `""` for each target (`modal_common.py:249-250`). Assert it
before dispatch: a run that leaks ~2500 designed sequences is not recoverable
after the fact.

### ESMFold2: the model is the choice, the variant is derived

`--variant fast|full_msa|full_nomsa` is required and has no default, but it is
not three peers. The kit describes it as picking "the model: ESMFold2-Fast, or
ESMFold2 with / without MSAs" (`esmfold2/README.md:72-73`) — so it encodes two
independent things: **which model** (Fast or full) and **whether an MSA is
supplied**. Two further details confirm the split: only `full_msa` reads a
chain's `"msa"` field (`:103`), and the kit's own output filenames carry
`__fast__` or `__full__`, not the three variant names (`:107`).

So the campaign YAML names the model, and the orchestrator derives the variant:

| `model` | Step | MSA present | `--variant` |
|---|---|---|---|
| `esmfold2` | apo | no | `full_nomsa` |
| `esmfold2` | holo | yes (targets) | `full_msa` |
| `esmfold2-fast` | apo | no | `fast` |
| `esmfold2-fast` | holo | ignored | `fast` |

This removes a field from the YAML rather than adding one, and removes the
possibility of setting a variant that contradicts the step's MSA policy. It also
matches Optimus, which already treats ESMFold2 and ESMFold2-Fast as separate
model names.

**Warn on the silent case.** `esmfold2-fast` does not consume MSAs, so choosing
it means the cached target MSA is built and then ignored for holo. That is a
legitimate choice — it is the cheap screening configuration — but it should be
logged at config resolution rather than discovered from a flat ipTM distribution.
The cache is still built, since the cross-validation model needs it regardless.

Note that `variant` remains meaningful for af3_jax, where it names a checkpoint
(`p2`) rather than a model/MSA combination. Same flag name, unrelated axes.

---

## Refolding metrics

Every sequence from MPNN is refolded.

### ipSAE: in-process, not a subprocess

Compute ipSAE **in process from the PAE array**, using the generalized
`calculate_ipsae_complex_poc`-style implementation rather than shelling out to
the `ipsae` CLI.

The reason is throughput. `StrucTools.calculate_ipSAE` runs the CLI
(`StrucTools.py:390`), which needs a PAE file and a structure file on disk per
call. This pipeline computes ipSAE for every holo prediction plus every
cross-validation seed — order 480 x (1 + 5) calls for the alphaV/beta3 campaign —
so per-call subprocess spawn and temp-file I/O is pure overhead inside a chunked
worker that already holds the PAE array in memory.

**Cutoffs are passed explicitly, always.** Both existing implementations default
to 15 (`StrucTools.py:390,412`) while this pipeline specifies 10, so a default
would be silently wrong. They come from the `ipsae:` block in the YAML.

**Validate once against the reference before trusting it.** The in-process
version is PAE-only; the Dunbrack CLI additionally takes a distance cutoff, so
the two need not agree. Run both on a handful of complexes during the validation
script (which already runs known complexes through every metric) and record the
agreement. If they diverge, the reference wins and the fast path gets fixed — but
this is exactly the check the validation script exists to make cheap.

Implementation requirements, beyond the reference version:

- **Assert the PAE matrix matches the sequences.**
  `sum(len(s) for s in seq_list) == pae.shape[0]`, checked before slicing.
  Without it, a model returning per-token PAE with extra tokens mis-slices every
  chain and yields plausible wrong numbers with no error. This is the single most
  important addition.
- **Take chain IDs explicitly**, from the resolved `chain_binder` /
  `chain_targets`, rather than deriving them as `chr(ord("A") + index)`. The
  derivation happens to be right under this pipeline's convention, but it couples
  the metric to the convention silently.
- **Mask directly** — `np.ma.masked_array(pae_subset, mask=~pae_pass_mask)` —
  instead of writing a `-1000` sentinel and calling `masked_values`. It removes
  both the float-comparison semantics of `masked_values` and any chance of a real
  PAE value colliding with the sentinel.
- **Key the binder results by the non-binder chain explicitly**, not by whichever
  element of the pair is second. The positional form is correct only while the
  binder sorts first.
- Silence per-call printing by default; at this call volume it is log noise.

ipSAE needs PAE, pLDDT and pTM matrices, so each folding step must persist raw
confidence output, not just scalars. Make that an explicit per-model output
contract — the kits' output layouts differ, and a single-sequence apo run has no
meaningful inter-chain PAE at all.

Alignment on CA atoms only, via `biotite.structure.superimpose`. Optimus has all
three of these already (`StrucTools.py:342,367`):

- `rmsd_motif_on_target` — align refolded and RFD3-designed structures on the
  target chains, then RMSD over motif residues only, using the
  `diffused_index_map` mapping. Gate: `<= threshold_rmsd_motif`.
- `rmsd_binder_on_target` — same alignment, RMSD over binder residues. Recorded,
  not gated.
- `rmsd_binder_apo_holo` — **refolded apo vs refolded holo**, superimposed
  **binder-on-binder** (extract the binder chain from holo, superimpose the apo
  binder onto it), RMSD over binder residues. Gate:
  `<= threshold_rmsd_apo_holo`.

  *Corrected from the original plan*, which specified aligning on target chains.
  That cannot work: the apo prediction is binder-only and has no target chains to
  align on. The metric's purpose is to measure whether the binder adopts the same
  fold free as it does bound, so binder-on-binder superposition is both the only
  computable option and the right one. Optimus already implements it this way —
  `run_superimpose_and_calculate_rmsd_apo_holo_pipeline` pulls the binder chain
  out of holo and superimposes the apo structure onto it
  (`StrucTools.py:380-386`). Note its `assert len(fixed) == len(mobile)`
  (`:358`): apo and holo binders must have identical residue counts, which holds
  since they are the same sequence, but will fail loudly if a model drops
  residues.

Note that `rmsd_motif_on_target` requires target chains, so it is undefined for
apo. The metric table must not try to compute it there.

---

## Step 1 — apo (binder only)

Model and `mode` from the YAML. `max_concurrent` sets how many predictions run
at once, one GPU each; `n_gpu_shard` is separate and only meaningful under
`mode: big`, where it splits **one** prediction across 2, 4 or 8 GPUs of one
host (`esmfold2/README.md:143`, `af3_jax/README.md:11`). These are different
axes and the YAML now names them differently.

Binder is single-sequence and not templated.

Pass if binder pLDDT `>= threshold_plddt_binder`.

## Step 2 — holo (complex)

Same model as step 1. Runs only the designs that passed apo.

Binder single-sequence and untemplated; target chains always use the cached MSA.
`target_template: true` optionally supplies the input structure as a template for
target chains only, never the binder. ESMFold2 does not support templates, so
the schema must reject `target_template: true` with `model: esmfold2` rather
than silently ignoring it.

Pass if all of:
- binder pLDDT `>= threshold_plddt_binder`
- `rmsd_motif_on_target <= threshold_rmsd_motif`
- `rmsd_binder_apo_holo <= threshold_rmsd_apo_holo`
- ipTM `>= threshold_ipTM`

Record complex ipSAE as `ipSAE_min` of the two directions.

## Cross-validation

A different model from the refolding steps, on the designs that passed holo. No
gating changes the set — this step only adds columns.

`num_seeds: 5`, but **how seeds are requested is per-model** — there is no shared
flag, and the semantics differ, not just the name:

| Model | Flag | Semantics | Per-seed output |
|---|---|---|---|
| esmfold2 | `--seeds a,b,c` | explicit list, one worker | `cif_all/<id>__<variant>__s<seed>_x<k>.cif` + `…_pae.npz` (`esmfold2/README.md:94,107`) |
| boltz2 | `--seeds a,b` | explicit list, every input at every seed, one worker | `by_seed/<name>/s<seed>/` (`boltz2/README.md:91,103`) |
| af3 / openfold3 | `--num_seeds N` | **generates** N seeds in sequence from the single seed in the input JSON | AF3's own per-seed dirs (`run_alphafold.py:441-447`, applied `:1725-1727`) |

Two consequences for the seed adapter each model wrapper has to implement:

- **The cost win generalizes.** All three run multiple seeds inside one worker
  and one model load, so five seeds cost one container, not five.
- **Reproducibility does not.** With esmfold2 and boltz2 the seeds are yours and
  reproducible by construction. With the AF3 fork you pass a *count* and it
  derives the seeds, so the seeds actually used are only discoverable from the
  output JSON it writes. Either way, record the concrete seed on every
  cross-validation row of `summary.csv` — otherwise a re-run is not comparable to
  the run it is meant to reproduce.

Where a model takes an explicit list, generate it from a campaign-level base seed
(`base_seed + i`) and store it in `campaign.resolved.yaml`.

Binder single-sequence and untemplated; targets use the MSA from the cache.
Optional target templating as in step 2.

Per seed, store `ipSAE_min_{target_chain} = min(binder_to_target,
target_to_binder)` — the `ipsae` package computes both directions. With more
than one target, also store `max_ipSAE_min`. Across seeds, store mean and max.
Also compute the RMSDs defined above.

(ipsae-columns)=
### ipSAE columns

The column set is derived when the YAML is resolved, not from any output file:

```python
binder  = cfg.general.chain_binder                    # default "A"
targets = cfg.general.chain_targets                   # e.g. ["B", "C"]
cols  = [f"ipSAE_min_{t}" for t in targets]
cols += ["max_ipSAE_min"] if len(targets) > 1 else []
```

Output chain IDs follow RFD3's deterministic renumbering: the binder is the first
contig component and the targets follow in contig order, so they are `A`, then
`B`, `C`, ... The binder is `A` by convention but stays configurable via
`chain_binder` rather than being hardcoded.

Because the convention is derived rather than guaranteed, **assert it once per
run** instead of trusting it: after the first RFD3 design is written, read the
chain IDs actually present in the output and check they equal
`[chain_binder] + chain_targets`. A contig whose ordering differs from the
declared chains fails loudly at design one, rather than silently producing ipSAE
columns attributed to the wrong chain for the whole campaign. This belongs in
`rfd3_spec` and is unit-testable without a GPU.

The same resolved list drives `determine_binding_interface`, the hotspot
renumbering and the `summary.csv` header, so there is one source of truth for
chain identity across the pipeline.

`summary.csv` gets a `passes_filters` boolean combining this step's gates with
everything upstream (anything reaching this step has already passed the rest):
binder pLDDT `>= threshold_plddt_binder`, `rmsd_motif_on_target <=
threshold_rmsd_motif`, `rmsd_binder_apo_holo <= threshold_rmsd_apo_holo`.

---

## Rank and report

Sort by ipSAE, for both refolding and cross-validation.

Cluster backbones by TM-score and write a cluster ID onto every final structure.
US-align is not pip-installable — either compile it from source in the image, or
use `tmtools` (Python bindings for TM-align). Specify the clustering explicitly:
binder chain only, TM-score threshold 0.5, greedy single-linkage, largest
cluster first.

---

## Not in the pipeline: model selection validation

A separate script, built **first**. Two reasons, and the second is the bigger one.

**It chooses the campaign's config.** `model`, `variant`, `mode` and
`target_template` are currently guesses in the YAML. This script measures them on
*this target* rather than inheriting someone else's benchmark.

**It de-risks everything except RFD3 and MPNN.** By the time it runs green you
have exercised: the per-model folding wrappers, the MSA cache end to end, the
alignment and RMSD code, ipSAE with the right cutoffs, the kit `mode`/`card`/
`variant` plumbing including the exit-3 failure path, the JIT warm path, and the
shard→reduce plumbing. That is most of the expensive half of the pipeline,
reachable without generating a single backbone.

### What it varies

One run per combination of: `model` x `variant` x `mode` x `target_template` x
`seed`, on the known input complex. Plus an apo prediction of the native binder,
because `rmsd_binder_apo_holo` is a gate and you need to know what that number
looks like for a binder that genuinely folds and binds.

Including `mode` is deliberate. The kits claim `fast` sits "within stock's
seed-to-seed variation" and `exact` reproduces `off` bit-for-bit
(`esmfold2/README.md:14`, `boltz2/README.md:131-137`). That is a testable claim on
your target: run `off`/`exact`/`fast` across seeds, and if `fast` differs from
`exact` by more than the seed-to-seed spread, you have found a reason not to use
`fast` for gating. If it does not, you have justified the cheaper mode with
evidence.

Also run the target with and without its MSA. If the MSA does not improve
target-chain accuracy, the cache is less load-bearing than assumed and holo can
be cheaper.

### What it measures

RMSD to the input structure, three alignments — whole complex, target chains only,
binder only — as originally planned. But the more valuable output is the second
group: **every metric the pipeline gates on**, computed on a complex known to be
real. Binder pLDDT, ipTM, ipSAE (both directions and `ipSAE_min`), hotspot contact
fraction, `rmsd_motif_on_target`, `rmsd_binder_apo_holo`.

This is what calibrates the thresholds. The plan gates at pLDDT >= 70 and
ipTM >= 0.3 — numbers carried over from general practice. If the *native* complex
scores ipTM 0.35 with this target under this model, then a 0.3 gate admits
essentially everything and the filter is doing no work. If it scores 0.85, there
is room to tighten and save cross-validation GPU time. You cannot know which
without this run, and every threshold in the campaign depends on it.

Record wall time and GPU-seconds per combination too — that feeds both the cost
model and the chunk size for `.map()`.

### The trap: the input structure is itself a prediction

`path_input_structure: in/rfdiffusion/tpo_alphafold.pdb` is an AlphaFold model. So
"RMSD to the original structure" measures agreement with AlphaFold, not accuracy.
Used naively to pick a model, this selects whichever model most resembles
AlphaFold — which is circular, and biased toward AF3/OpenFold3 over ESMFold2 for
reasons that have nothing to do with which is the better filter.

Handle it explicitly:

- If an experimental structure of the target complex exists, validate against
  **that**, and keep the AlphaFold model only as the RFD3 input.
- If none exists, report the RMSDs as *self-consistency* and say so in the CSV
  header. Then lean on the calibration numbers above, which do not depend on the
  input being correct — a model that gives a real complex high confidence and
  low apo-holo drift is informative regardless of what the reference coordinates
  came from.
- Either way, the target-chains-only alignment is the most trustworthy of the
  three, since the target is where the MSA and the template actually constrain
  the prediction.

### Output

One row per (model, variant, mode, target_template, msa, seed), with the three
RMSDs, the full gate metric set, wall time and GPU-seconds. Then a reduce pass
giving mean and standard deviation across seeds per
(model, mode, templating) — the sd is what tells you whether a difference between
two models is real or noise.

Same shard→reduce plumbing, same metric functions, same MSA cache as the
pipeline. It is not a throwaway script; it is the pipeline's folding half driven
through a different entrypoint.

### Decision rule

Pick the `model`/`mode`/`target_template` that (1) reproduces the reference
complex within tolerance, (2) gives gate metrics with clear headroom above
threshold on a known-good complex, and (3) fits the budget at campaign scale.
Then write those values into the campaign YAML, and set the thresholds from the
calibration rather than from convention.

---

## Open questions

1. **Splitting the chain-E metals** — optional, see
   [Ligands](#ligands-the-midas-metal). `ligand: "E703"` works against the input
   as it stands; the full beta3 triad needs a one-line PDB edit first.

2. **Sampler A/B** — `step_scale` / `gamma_0` now start at stock. Worth one
   small-batch comparison against the original plan's 3 / 0.2 once the pipeline
   runs, since the Foundry binder example that motivated those values could not
   be located.
3. **`select_fixed_atoms` per spec** — `ALL` on RGD in the campaign YAML, since
   the arginine and aspartate side chains are the binding determinants; confirm
   per motif.
4. **`af3_native` vs `openfold3`** — the Anthropic kit ships OpenFold3 (`p2`,
   Apache-2.0). Official AF3 weights are reachable via the Optimus path, whose
   own comment records that `WEIGHTS_TERMS_OF_USE.md` carries a non-commercial
   restriction and to use them only under clearance already obtained
   (`modal_common.py:172-178`). Worth deciding deliberately rather than by
   defaulting, since `model: af3` reads as either.
