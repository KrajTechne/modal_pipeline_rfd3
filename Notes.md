# Notes -- gotchas by campaign step

Traps that produce **plausible wrong numbers, silent data egress, or a blind
run** rather than a clean error. Ordered by where you hit them in a campaign.

Every item marked **[verified]** was checked against real output or upstream
source, with the date. Items marked **[unconfirmed]** are suspicions recorded so
they are not mistaken for knowledge.

---

## 0. Launching a run

**Redirect the whole stream to a file, and force UTF-8.** [verified 2026-10-07]

```sh
COLUMNS=200 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
  modal run app.py::run --campaign <campaign.yaml> --dry-run > run.log 2>&1
```

- Without `PYTHONIOENCODING=utf-8`, redirecting stdout drops Python to cp1252
  and the Modal 1.6.0 CLI's check-mark banner glyph (U+2713) kills the run on
  its first line: `'charmap' codec can't encode character '\u2713' in position
  0`. No file in this repo contains a non-ASCII character, so do not go hunting
  the glyph in our source -- it is Modal's.
- **Never pipe a launch through `tail` or `head`.** It hides both progress and
  the real error. Cost two misdiagnoses: the truncated head was where the actual
  cause was both times.

**A bare `RemoteError` from `orchestrate.remote` usually means the app was
stopped, not that the code failed.** [verified 2026-10-07] The Rich traceback
shows only the *client* call stack, so it reads like a remote crash. Check the
**head** of the log:

```
Stopping app - user stopped from dashboard.    (or: from CLI)
Runner terminated.
```

**`modal app stop` needs `-y` non-interactively.** [verified 2026-10-07] Without
it the prompt reads EOF and aborts with exit 1.

**`modal app logs <app-id>` can return empty** for a stopped or completed app.
[verified 2026-10-07] It is not a reliable post-mortem tool. The local redirected
log is the record -- which is the other reason not to pipe it through `tail`.

**Use `--detach` for real campaigns.** `orchestrate` runs remotely so the laptop
is not in the control flow, but the *client* still needs to detach for you to
close the lid.

**Environment:** venv lives at `~/venvs/rfd3-pipeline`, outside the repo -- the
project path is 97 characters and `LongPathsEnabled = 0` on this machine. Watch
free space on `C:`; it has hit 100% once, which breaks OneDrive sync and the
Modal client together.

**Clone anything destined for a Linux image with `core.autocrlf=false`.**
[verified 2026-10-07] On Windows git defaults to `true` and rewrites LF to CRLF
on checkout. Those files then go into the image as-is and shell scripts break in
ways that name the wrong culprit -- the ESMFold2 kit died on
`run.sh: line 19: set: pipefail: invalid option name`, which is bash reading
`pipefail\r`, not a bad script. Files that are *sourced* rather than executed are
worse: a `configs/*.env` with CRLF sets every variable with a trailing `\r` and
the failure surfaces much later as an unexplained `exit 3`.

    git clone -c core.autocrlf=false ...
    # repair an existing clone:
    git -C <dir> config core.autocrlf false
    git -C <dir> rm --cached -r -q . && git -C <dir> reset --hard -q

Also note the long-path limit applies to clones under the scratchpad directory:
a `git clone` there failed with `Filename too long` while the same clone
succeeded under `~/`.

---

## 1. Writing the campaign config

**RFD3 is Hydra: `key=value`, no dashes, no `--`.** [verified 2026-09-28]
The flags are `n_batches` and `diffusion_batch_size`
(`models/rfd3/docs/input.md:65`).

**The campaign YAML carries config for stages that do not exist yet.**
[verified 2026-10-07] `gates.py:36-43` declares six stages
(`rfd3 -> backbone_filters -> mpnn -> apo -> holo -> cv`); `app.py` implements
the first two. Values like "sequences per design" are currently inert. Do not
read a summary as though the later stages ran.

**Target chain count varies per campaign** but is known from
`len(chain_targets)` at config-resolution time. Output chain IDs run `A` for the
binder, then `B`, `C`, ... for targets in contig order. Only sequence *identity*
between target chains is a runtime question, read from the input structure, and
it affects MSA pairing alone.

**`--dry-run` renames the run.** It reduces to one spec / two designs / one CV
seed and suffixes the run name, so output cannot be confused with a real
campaign. A campaign that mysteriously produced two designs is almost always
this flag.

---

## 2. Spec resolution and `specs.yaml`

**RFD3 resolves a relative `input` against the directory holding the specs
file, not the working directory.** [verified 2026-10-07] This is the subtlest
trap found so far. A campaign-relative path silently becomes
`<run>/rfd3/<campaign-relative path>` and RFD3 dies with `FileNotFoundError`
having produced nothing -- which then surfaces downstream as the harmless-looking
`phase filter: 0 chunk(s)`.

Pass the absolute path the structure will have *in the container*
(`build_rfd3_specs(resolved, input_path=...)`).

**`ResolvedCampaign.structure_path` cannot be used for that.** It is resolved
against the *client's* filesystem, so on Windows it would write a
`C:\Users\...` path into a file consumed inside a Linux container.

**`length` is deliberately never emitted** into `specs.yaml`. It is optional
when a contig is present, it constrains the contig rather than the whole system,
and a derived second copy is a second thing to keep in sync.
`ResolvedSpec.length` keeps it for assertions and reporting only.

**`global_prefix=''` paired with `json_keys_subset` is the documented sharding
idiom** -- the RFD3 config annotates the pairing as "(pipelines usage)". A
consequence: design filenames carry no prefix, so the older fixture is named
`specs_rgd_core_0_model_0` while current output is `rgd_core_0_model_0`.
`_spec_of` matches spec names as substrings **longest-first**, so a spec name
that is a prefix of another cannot win by accident.

---

## 3. Images and weights

**Nothing not present in the container may be imported at module level.**
Modal imports `app.py` inside the container as well as locally, so one stray
top-level import crashes every container before its function body runs. On
`run_rfd3` that means paying for an A100 to fail on an import.

**Nothing crossing a function boundary may be a library type.** The campaign
travels as a plain dict and is re-validated on the far side. A pydantic model or
an AtomArray would fail to unpickle wherever its library is absent.

**`add_local_python_source` must come last.** No build step may follow it.

**Set `PYTHONUNBUFFERED=1` in every image.** [verified 2026-10-07] Otherwise a
child process's output sits in its pipe buffer and the log view stays empty
until the step ends.

**`populate_weights` is its own function on purpose.** The installers must run
inside a container; once the volume is populated it can be mounted read-only
everywhere else. Re-running only re-does the hash check.

### Volumes: `commit()` is only half of it

**Every step must `commit()` at the end**, and periodically inside long ones.
Writes are not durable to other containers until committed.

**Every reader must `reload()` before reading what another container wrote.**
[verified 2026-10-07] This is the half that is easy to miss, and missing it is
silent. `commit()` makes a write durable; it does **not** push it into a
container that is already running. A long-lived container therefore keeps
whatever view of the volume it had at startup.

The first full end-to-end run hit exactly this. `orchestrate` is long-lived, so
after `filter_designs` committed three passing shards it still saw an empty
directory, reported `n_passing_backbone_filters: 0`, skipped MPNN and **exited
0**. The reducer -- a fresh container -- read the same shards and reported 3.
Two numbers in one report that could not both be true.

The tell is a disagreement like that, or a step reporting zero inputs. It never
raises. Note also that it is unreachable via `--start-at`, because that
`orchestrate` container starts *after* the shards exist: only a continuous run
exposes it.

Fresh containers see everything, so `filter_designs`, `run_mpnn` and
`reduce_summaries` are correct today without it. They call `reload()` anyway,
because Modal may reuse a container across `.map()` inputs and a stale view
reads as "missing input" rather than as an error.

**The Anthropic `af3_jax` kit ships OpenFold3 *ported* weights (`--variant p2`),
not official AF3 weights** (`af3_jax/README.md:4`). [verified 2026-09-28]
Commercially relevant: the official weights carry a non-commercial restriction.

**Kit modes exit 3 rather than falling back** when a mode cannot engage, and
`--config` cards are `h100|h200|a100` only. [verified 2026-09-28]

---

## 4. RFD3 design (GPU)

### The seed is top-level `seed`, and it only goes so far

**Unseeded is upstream's default.** [verified 2026-10-08] RFD3's composed config
has `seed: None`, so passing nothing means every run samples differently -- two
runs of a byte-identical campaign produced different backbones, which is what
exposed this.

**The override is `seed=N`, not `inference_engine.seed=N`.** The nested form is
*refused*:

    ConfigCompositionException: Could not override 'inference_engine.seed'.
    To append to your config use +inference_engine.seed=0

`seed: null` is written in `configs/inference_engine/base.yaml`, but
`configs/inference.yaml` selects `inference_engine: rfdiffusion3`, and that
option is `@package _global_` inheriting `base` -- so the key merges at the
*root*, beside `out_dir`, `n_batches` and `inference_sampler`. Confirmed by
composing the config the way `rfd3/cli.py` does; the key's location in the
source tree is not where it ends up.

**Do not take Hydra's `+` suggestion.** `+inference_engine.seed=0` would be
accepted and set a key nothing reads, leaving sampling unseeded while looking
configured. Worse than the error. Where a key lives has to be read off the
*composed* config, not grepped out of the tree.

**It is a real argument, just an undocumented one.** It appears in neither the
foundry documentation nor `rfd3 design --help` -- Hydra config keys never show
up in argparse help -- so its absence from the docs is not evidence against it.
What establishes it:

* Hydra's struct mode *refuses* unknown paths, which is why
  `inference_engine.seed=0` failed. An unknown top-level key would be refused
  the same way, so `seed=0` being accepted means the key genuinely exists in
  the composed config.
* **RFD3 records the seed it used in every design's metadata JSON.** The
  checked-in fixture from an unseeded run reads `seed: None`; a run passing
  `seed=0` reads `seed: 0`. The engine reads the key rather than merely
  tolerating it.

**That output field is the cheap post-hoc check**: read `seed` from any
`rfd3/out/<design>.json` to confirm a campaign actually ran with the seed you
think it did, rather than trusting the command you believe you issued.

**A seed fixes the discrete sampling, not the diffusion trajectory.** [verified
2026-10-08] Two seeded runs gave the same binder length and the same motif
placements, against an unseeded run that differed in both:

| | binder length | motif offsets |
|---|---|---|
| unseeded | 118 | A17, A46, A63 |
| seeded A | 108 | A70, A89 |
| seeded B | 108 | A70, A89 |

But the geometry diverged -- `bb_rg` 12.63 against 12.40, `bb_loop_fraction`
0.295 against 0.324 -- and the MPNN sequences differ entirely as a consequence,
because MPNN is designing against a different backbone rather than being
unseeded itself. Over `num_timesteps: 200`, non-deterministic CUDA reductions
amplify into Angstrom-scale differences. Accepted: chasing bit-exactness would
mean forcing deterministic algorithms inside RFD3, which this pipeline does not
control and which costs speed.

**So a dry run is reproducible in shape, not in numbers.** The same code paths
get exercised -- same design count, lengths and motif placements -- while the
metrics move. For a sampler comparison (`step_scale=3` against stock), compare
*distributions* over several designs rather than diffing two runs; a two-run
diff cannot attribute the difference.

**Seeded for dry runs, unseeded at full scale.** `Rfd3Config.seed` defaults to
None and `apply_dry_run` sets it from `general.base_seed`. Once a dry run has
shown the workflow sound, diverse backbones across a campaign are the point. A
campaign that pins its own seed keeps it.

### Everything else

**Never use `capture_output=True` for the design subprocess.** [verified
2026-10-07] It holds every line in a pipe until exit, so a six-hour job is
completely silent, and a `returncode: 1` gets reported into a dict while the
pipeline carries on as though nothing happened. Use `_stream` (`app.py:77`).

**Metal ideal coordinates can fall back to NaN.** [verified 2026-10-07]

```
No suitable coordinates found for 'MN' among preferences
('ideal_pdbx', 'model', 'ideal_rdkit'). Coordinates will be 'nan'.
```

This did **not** error and did **not** break the run. Whether it perturbs any
ligand-adjacent metric is **[unconfirmed]** -- it needs a design with a real
metal contact to answer. This is the highest-priority unknown here, because NaN
propagating into a distance is exactly the "plausible wrong number" failure mode.

**Benign-looking warnings seen on every run** [verified 2026-10-07], none fatal:

- `Environment variable CCD_MIRROR_PATH not set`, same for `PDB_MIRROR_PATH`.
- `We can't fix formal charges without building from templates ... set
  add_missing_atoms = True`.
- `provider=hydra.searchpath in main, path=configs is not available`.

**`skip_existing=True` stays on**, so a preempted container regenerates only
what is missing rather than starting the spec over.

**Reading designs back:** `diffused_index_map` is **top-level**;
`sampled_contig` is at **`specification.extra.sampled_contig`**, *not* top-level
`extra`. [verified 2026-10-07] Confirmed against real output; the top-level key
set has not drifted between two real designs.

**The motif gets renumbered.** Input residues `A77-79` land at `A49-51` in the
design. A silent off-by-one here invalidates every downstream RMSD and contact
count, which is why `rfd3_spec` was built and tested first.
`DesignMetadata.binder_entries` is the mapping.

**Output chains include a chain per metal.** A run with the chain-E metal came
back `A: 105, B: 177, C: 242, E: 1`, with E holding `MN`. Budget for more chains
than the binder-plus-targets count.

---

## 5. Backbone filters

**`chunk_size` is batching, not parallelism.** [verified 2026-10-07] It is how
many designs go into *one* container, processed in a sequential loop inside it.
The parallelism is the *number of chunks*, fanned out by `.map()`. 1000 designs
at `chunk_size=32` is 32 containers each doing 32 designs in series. Batching
exists because container startup dwarfs the few hundred milliseconds of actual
filtering.

**One bad design must not lose the chunk.** The per-design `try/except` puts
failures in a `failures` list instead of killing the container's other work.
**Check that list** -- an empty `failures` is part of a green run, and a
non-empty one does not fail the step.

**Do not allocate a dense contact mask.** It was 2.6 MB for this target but
**92 MB at 3000 residues**. The neighbour-index form gives identical reductions.

**Do not use `np.ma` in an inner loop.** Measured **7.7x slower** than the
equivalent `np.where` for the same answer. Chain blocks are contiguous, so
`np.ix_` copies where a slice would view.

**Thresholds are unexercised.** 1 of 2 designs passed at smoke scale. The gate
values have never been run against a population. **[unconfirmed]**

---

## 6. MPNN

**`--chains_to_design` is not optional.** [verified 2026-10-07] Omit it and MPNN
designs *every* chain: it invents new sequences for the target chains and
optimises the binder against them. It does not error, and the binder sequences
look entirely plausible -- they are simply optimised against a target that does
not exist. Pass the binder chain. The giveaway in the log is `num_res` counting
the whole system minus only the fixed residues.

Verification that costs nothing: the target segments of every design must come
back byte-identical to the native record's.

**`--seed 0` was silently discarded until fixed locally.** [verified
2026-10-07] Upstream `run.py` tested `if args.seed:`, falsy for zero, so an
explicit seed of 0 was replaced with `np.random.randint(0, 99999)`. With
argparse's `default=0`, the documented default was in fact "random" and seed 0
could not be requested at all. This campaign sets `general.base_seed: 0`, so
sequence generation was unreproducible while reporting a seed it had invented
(observed: 87657). Fixed in the vendored copy; see
`LigandMPNN/LOCAL_CHANGES.md`. If the vendored tree is ever re-pulled from
upstream, this returns.

**The first FASTA record is the input sequence, not a design.** [verified
2026-10-07] `num_seqs: 8` yields nine records. Treating the first as a design
gives every backbone one bogus sequence identical to its input.

**Chains are joined by `:` in the FASTA**
(`--fasta_seq_separation`), binder first since it is chain A. A record holding a
target's sequence as the binder's would be refolded as though it were the
binder.

**MPNN wants PDB; RFD3 writes gzipped mmCIF.** Convert with ProDy, which is
MPNN's own parser, rather than introducing a second opinion about the structure.
RFD3 output needs no `UNK` -> `GLY` rewriting (`has_UNK: False`), so the
preprocessing in the bundled `wrapper.py` is unnecessary here.

**Batch per chunk, not per design.** MPNN is ~7.5 s for eight sequences over a
520-residue system, nearly all of it loading the model, so a container per
design is almost pure startup. `--pdb_path_multi` and `--fixed_residues_multi`
take JSON maps (`{"/path/to.pdb": ""}` -- only the keys are read for the former),
so one invocation covers a chunk with per-design fixed residues.

**Do not re-derive `fixed_residues` here.** Read the value the backbone filters
recorded in the design's shard. The numbering conversion lives in exactly one
place because getting it wrong is silent, and reading the recorded value also
makes `--start-at mpnn` select the same designs a continuous run would.

**The bundled `wrapper.py` is not usable as-is.** It passes `--return_logits`
and `--temperature_per_residue`, which this `run.py` does not define (they come
from a fork), so those calls exit 2. It also uses
`subprocess.run(capture_output=True)` -- the blindness described in section 0.

---

## 7. Reduce and summaries

**Everything the reducer writes is derived**, so it is safe to re-run at any
point, and `reduce_only` is how a restarted campaign recovers its summaries
without re-running the GPU stage. `orchestrate --start-at` likewise skips
straight to filtering over designs already on the volume.

**`reduce_only` needs `--dry-run` too.** [verified 2026-10-07] The flag renames
the run, so without it this reduces `{run_name}` while the dry run's shards sit
under `{run_name}-dryrun`. The result is a summary of nothing -- `n_records: 0`
-- rather than an error.

**A design's shards describe it at several stages, and the records must be
collapsed before summarising.** [verified 2026-10-07] Sequence records carry the
backbone's `bb_*` columns forward, so counting backbone and sequence shards
together double counts: one passing backbone with two sequences reported
`n_passing_backbone_filters: 3`. `latest_records` drops a backbone record once
sequences exist for that design and keeps it otherwise, which is the intended
set -- one row per sequence, plus the backbones that died before MPNN. The
shards themselves are left alone, so a restart still sees every verdict.

Related: tallies over a one-row-per-sequence table must count **distinct
designs**, not rows. A backbone with eight sequences is one design that passed,
not eight.

**The backbone-to-sequence link is already in the summary, twice.** Every
sequence row carries `design_id`, and `seq_id` is `{design_id}_seq{N}`; they sit
adjacent in the leading columns. Group by `design_id` for a backbone's
sequences. Do not add a third representation of the same relation.

---

## 8. Apo refolding, and the ESMFold2 kit

Apo prediction works as of 2026-10-08. Everything here was read off the kit or
observed in a run, not inferred.

### Building the kit image

**It is not a package.** `Image.from_dockerfile` against the kit's own
`esmfold2/environment/Dockerfile` is the supported route; route C (venv) wants
uv, a C++ compiler, the CUDA 13.0 toolkit and two Biohub packages at pinned
commits, and anything off-pin exits 3. The kit lives outside this repo at
`~/ubm` -- see `experiments/esmfold2_image.py` for the two-command setup, which
includes the autocrlf and `_jitcache` requirements from section 0 and below.

**`STACK=img_esmfold2_a100`** for A100 (sm80 + sm90); the default `img_ef2_fa`
is sm90 only. The stack choice doubles the flash-attn compile.

**Pin `BUILD_JOBS`.** [verified 2026-10-08] The Dockerfile computes
`JFA = MemAvailable / 9` for flash-attn, assuming ~9 GB per compile job. 17 jobs
completed on a builder reporting 328 GB and were **OOM-killed (exit 137)** on
one reporting 203 GB, so real peak is above ~12 GB. Modal allocates whichever
builder is free, so the default makes success a coin flip. `BUILD_JOBS=8` fits
either machine -- and was *faster*, 1543 s against 1932 s, because less memory
pressure beat more parallelism.

**Modal rejects Docker's optional-COPY idiom.** The last COPY is
`_jitcache/esmfold2-*-jit.ta[r]`, where the `[r]` bracket makes a glob that
matches nothing tolerable in Docker. Modal fails the build instead. Supply an
empty archive; the following `RUN` guards with `[ -f "$t" ]`, so it extracts
nothing and the image is identical:

    mkdir -p ~/ubm/_jitcache
    tar -cf ~/ubm/_jitcache/esmfold2-empty-jit.tar --files-from /dev/null

**Do not put anything new under `esmfold2/`.** It is copied at Dockerfile line
81, *before* the 32-minute wheel build at line 83, so any change there
invalidates the cache. This is why the built wheels are parked at
`~/esm_wheels/` rather than in `stock/wheels/` where `WHEELS_FROM=prebuilt`
wants them.

**`WHEELS_FROM=prebuilt` will not accept wheels you built yourself.** It checks
each sha256 against `stock/PINS.json`, and wheel builds are not
byte-reproducible, so all three mismatch. Patching `PINS.json` would fix it and
costs exactly what it saves, since `PINS.json` is also under `esmfold2/`. Worth
doing only when a rebuild is already forced for another reason.

The *runtime* pin check is a different mechanism and does pass:
`check_pins.py` validates by version string and reports all three as
`pinned (stock/PINS.json "image" layer)`.

### Running it

**`--variant` is required, no default, one per process.** Apo is always
`full_nomsa`; holo is `full_msa`. `--config` follows the GPU, not the model.

**A mode that cannot engage exits 3 and never falls back.** Exit codes: 0 ok,
1 failed, 2 usage, 3 not active. And **every config sets
`ESMFOLD2_OPT_REQUIRE_FAST_ENV=1`, so even `--mode off` exits 3 without
flash-attention and TransformerEngine** -- there is no unaccelerated escape
hatch, which is what rules out any GPU with no kit card.

**`server_mode=opt14_msa` under `full_nomsa` is not a contradiction.**
[verified 2026-10-08] It is the internal name of `fast` mode's kernel
composition (`FAST_LINE = KitMode("opt14_msa", {})`,
`opt/esmfold2_opt/modes.py:135`), constant for `--mode fast` whatever the
variant; `big` reports the same. The `msa` is the fused MSA-module kernel, which
`full_nomsa` legitimately uses -- the module exists and is fed the query alone at
depth 1.

**On A100 expect `not_for_class=t16` and `card_support=uncertified:sm80`** on
LEVER lines. The compute-capability-9.0 transition kernel stands down and the
portable kernels serve. Informational, not failure.

**`warm` is worth running once per campaign.** 285 Triton kernels, 17.1 MB, into
`MODEL_OPT_JIT_ROOT` on the jit volume. Containers are ephemeral, so without it
every prediction pays a cold compile.

**Input is upstream's prediction-input JSON**, documented in the kit at
`opt/esmfold2_opt/inputs.py:7-9` -- read that rather than the example file,
which contains protein chains only and misleads:

```json
"sequences": [
  {"type": "protein", "id": "A", "sequence": "...", "msa": null},
  {"type": "protein", "id": "B", "sequence": "...", "msa": "msas/B.a3m"},
  {"type": "ligand",  "id": "L1", "ccd": ["ATP"]}
]
```

Top level is a *list* of items, each `{"id", "sequences", "seeds",
"num_diffusion_samples"}`. `id` names the output file, so setting it to `seq_id`
makes results directly addressable. `--input` takes the list, which is what
makes `chunk_size` worth having.

**Four chain types: `protein|rna|dna|ligand`** (`inputs.py:62`). Non-ligand
chains need a `sequence`; ligands take `ccd` (a list of CCD codes) or `smiles`
instead (`inputs.py:63`). So **ESMFold2 does take ligands** -- the alphaV/beta3
campaign's chain-E metal belongs in a holo input as
`{"type": "ligand", "id": "E", "ccd": ["MN"]}`.

**`msa` is the kit's extension, not upstream's field** -- the source calls it
"the package's one extension" (`inputs.py:8`).

**There is exactly one MSA field per chain, and one MSA CLI flag.**
`--msa_max_depth` (default 1024, `report.py:622`) is the per-recycle row draw
under `msa_subsample_at_inference`, not a hard truncation. An alignment deeper
than it also makes the `mh` lever inert (`modes.py:124-125`), which is reported
on the lever line -- so a3m depth is a performance knob as well as a quality
one.

**Output** is `<out_dir>/cif_all/<id>__<fast|full>__s<seed>_x<k>.cif` plus
`..._pae.npz`, and an appended `pred_rows.jsonl`. The token is `fast`/`full`,
not the variant name, so `full_nomsa` and `full_msa` are told apart only by
`out_dir`.

**`pred_rows.jsonl` carries the metrics**: `status`, `ptm`, `iptm`, `n_tokens`,
`msa_depths`, `wall_s`. `msa_depths: [0]` is positive proof no MSA was used --
check it rather than trusting the flag. `iptm: 0.0` is correct for apo, since
ipTM scores an interface a monomer does not have.

### Cost

**Read `wall_s`, not the wall time of the step.** One 102-residue apo fold was
`wall_s: 54.7` inside a stage that took 233.8 s; the difference is one-time
startup -- checkpoint load, CCD dictionary, CUDA graph capture, kernel install.
Marginal cost is ~55 s per sequence. Weight digest off the volume is 98.7 s for
24.9 GiB, close enough to local disk not to matter.

Do not extrapolate to holo: that is 524 residues with targets, a different cost
class.

**The driver margin is thin.** Modal's A100-40GB fleet reports 580.95.05 and
the kit requires `>= 580`.

### Reading the metrics

**`b_factor` is where pLDDT lives, and biotite does not read it by default.**
[verified 2026-10-08] `load_structure(path, extra_fields=["b_factor"])` --
without it the attribute is simply absent and confidence extraction fails with
`AttributeError`. It had to be threaded through both the plain and the gzip
branch.

**`apo_rmsd_to_design` is the metric apo exists for**, and it is computed here
rather than taken from the kit: superimpose the prediction's binder on the RFD3
backbone it was designed for. A first run spread 1.31 A to 11.27 A across six
sequences from three backbones, so it discriminates.

**pLDDT, pTM and RMSD agreeing is the sanity check worth doing.** Three
independently derived numbers -- one from the kit, one from the structure's
B-factors, one from a superposition -- ranked the same six sequences the same
way. If they disagree, suspect the wiring before the designs.

**Every metric is guarded individually.** A format surprise should leave one
null in the summary rather than lose a whole chunk. This earned itself
immediately: the `b_factor` failure above cost one column and nothing else.

### Re-running a stage

**A stage must drop its own columns before rewriting them.** [verified
2026-10-08] `_refold` starts from the shard as it finds it, and a shard already
holds the previous attempt's columns. An apo re-run that computed pLDDT fine
still carried the earlier run's `apo_plddt_error`, so the row asserted both a
value and a failure:

    apo_plddt_binder = 58.114
    apo_plddt_error  = "AttributeError: ... no attribute 'b_factor'"

`REFOLD_SUPERSEDES` drops a stage's own prefix plus every downstream one.
Diagnostic fields are the dangerous ones: a stale number reads as a measurement
and a stale error reads as a live problem. **Holo deliberately keeps apo's
columns** -- the apo/holo RMSD gate needs them, and apo is a completed upstream
stage rather than a previous attempt.

**Every phase in `orchestrate` must be gated on `start_at`.** An ungated phase
is invisible until a later stage exists beyond it: the MPNN phase was
unconditional, which was correct while MPNN was last, and became a silent
re-run of the whole GPU stage once `--start-at apo` was possible.

### Gates

`threshold_plddt_binder` is **70, settled from the user's prior practice**
(2026-10-08) rather than derived here.

**Apo rejecting binders that holo scores above 70 is the intent, not a
miscalibration.** Observed on a full run: two sequences failed apo at 63.9 and
67.9 yet scored 72.5 and 71.6 at holo with ipTM 0.86. Apo is deliberately the
harsher test -- a binder should fold correctly on its own, not only when the
target holds it in place -- so filtering there is what spends holo's budget
well. Do not "fix" this by lowering the gate.

The dry run relaxes it to 40 via `DRY_RUN_OVERRIDES`, because at smoke scale 70
rejects everything and holo folds nothing; that is about exercising the
pipeline's shape, and the production value stays honest. `apo_pass` is the only
gate apo can evaluate; `threshold_rmsd_apo_holo` and `threshold_ipTM` need
holo.

---

## 9. Stages not built yet (holo refolding, cross-validation)

Recorded now because the traps are already known and all are the silent kind.
Verified against upstream source rather than docs, dated per item -- most on
2026-09-28, the `msa_server.py` mechanics on 2026-10-07.

### Which stage each trap belongs to

**Corrected 2026-10-08.** Everything below about `--use_msa_server`,
`fill_missing_msas`, the two independent suppression gates, the `len > 1`
pairing threshold and the discarded-paired-results bug is **AF3-fork
mechanics**, so it applies to **cross-validation** (`cross_validation.model`
is `af3_jax`) and **not** to holo.

**The ESMFold2 kit contacts no MSA server at all** -- no `use_msa_server`,
`msa_server`, ColabFold or MMseqs reference anywhere in its Python, shell or
docs. It reads an a3m you hand it, or folds without one. So for holo on
`esmfold2` there is no automatic egress to disarm.

What holo needs instead:

1. **Target a3m files, built by us.** The kit will not fetch them. The egress
   is our own client's, and only over **target** sequences -- public proteins
   for this campaign. The binder never needs an alignment here, so the
   sensitive sequences stay in-house by construction rather than by
   suppression.
2. **The chain-E ligand** -- done, but with a limit to know about. `_refold`
   now derives the entry from `rfd3.ligand` plus the input structure, because
   neither the chain letter nor the CCD code is guessable:

   ```
   input structure:  chain D  res 1004  MN
                     chain E  res 703, 704, 705  MN
   rfd3.ligand: "E703"  ->  {"type": "ligand", "id": "E", "ccd": ["MN"]}
   ```

   "The chain after the protein chains" would be **D** -- a different ion the
   campaign does not use, at an unrelated residue number. And the chain letter
   has to match RFD3's output anyway, since the structural comparisons pair on
   `(chain, res_id)`.

   **ESMFold2 takes one ligand component, not a chain of them.** More than one
   configured ligand residue raises rather than folding against a subset. This
   is a *refolding* limit, not a config error -- RFD3 can legitimately be given
   the whole chain-E triad, so such a campaign validates and runs through RFD3
   and MPNN before failing at holo. Anyone splitting the metals for RFD3 needs
   to know that; it is not enforced at config time on purpose.
3. **What a3m shape `full_msa` expects** -- query first? -- and whether it
   tolerates a mix of chains with and without an `msa` path. Both unverified.

**`sokrypton/alphafold3 --use_msa_server` does not cache, despite its help text
saying results are cached in `--cache_dir`.** `run_alphafold.py:1728` calls
`fill_missing_msas` on every fold input, every invocation; `--cache_dir` is only
the JAX compile and tokamax autotune cache. Leaving it on would **POST every
designed binder sequence to api.colabfold.com**, once per prediction. This is
the data-egress trap, not a performance one.

### Which models need an MSA at all

Before any of the MSA machinery matters, check the model. The table is
`MODEL_CAPABILITIES` in `config.py` -- one row per model, no scattered
conditionals, and a missing field is a `KeyError` at import rather than a silent
fall-through. As of 2026-10-07:

| model | unpaired | paired | templates | seeds | weights | variant (no-msa / msa) |
|---|---|---|---|---|---|---|
| `esmfold2` | yes | **no** | no | `--seeds a,b,c` | -- | `full_nomsa` / `full_msa` |
| `esmfold2-fast` | no | no | no | `--seeds a,b,c` | -- | `fast` / `fast` |
| `boltz2` | yes | **no** *[unverified]* | no *[unverified]* | `--seeds a,b,c` | -- | none / none |
| `af3_jax` | yes | **yes** | yes | `--num_seeds N` | `openfold3` | `p2` / `p2` |

`consumes_msa` is **derived** from the two MSA flags (`msa_unpaired or
msa_paired`), not stored, so it cannot contradict them. `msa_paired` requires
`msa_unpaired`, checked at import: pairing refines per-chain alignments rather
than replacing them.

**`msa_paired` is the column that decides whether the cache needs a paired
path at all.** It says the *caller* must do the pairing. ESMFold2 takes a single
a3m per chain and pairs internally, so for an esmfold2 refolding stage the
answer is no -- which is why only unpaired alignments are worth caching. The AF3
fork takes `unpaired_msa` and `paired_msa` separately, so cross-validation is
the stage that needs paired alignments built.

The practical trap: applying the fork's two-field mechanics to an ESMFold2 input
writes a key its schema has no place for. Silently ignored, while looking
configured.

**boltz2's `msa_paired=False` is unverified, and note the asymmetry** with its
`supports_templates=False` above. There, `False` prevents a campaign silently
requesting something that would be dropped. Here, `False` risks silently
*under*-supplying: if boltz2 does want paired alignments we would never build
them and the predictions would be quietly weaker rather than refused. Same
conservative-looking value, opposite failure direction.

**On `esmfold2-fast` there is no MSA step and no egress risk** -- nothing below
is reachable. Apo and holo then differ only by which chains are in the input.
Everything below goes live the moment the model changes.

**The kit `--config` card is a function of the GPU, not the model.** `GPU_TO_CARD`
maps `A100`/`A100-40GB`/`A100-80GB` -> `a100`, `H100` -> `h100`, `H200` -> `h200`.
`Mode` (`off|exact|fast|big`) is configured per step. Three independent axes --
model capabilities, GPU card, acceleration mode -- so do not merge them into one
table.

`boltz2` is the least-specified row: `consumes_msa` is true but it has neither a
derived nor a fixed variant, and its template support is held false
deliberately. Verify that row against source before trusting it.

### Suppressing the binder's MSA takes *two* fields, not one

[verified 2026-10-07 against `msa_server.py`, read at `main`]

`fill_missing_msas` runs two independent queries, gated on **different** fields:

```python
# :195  unpaired query -- gated on unpaired_msa
if chain.unpaired_msa is None and chain.sequence not in protein_seqs_needing_msa:

# :212  paired query -- gated on paired_msa, separately
c.sequence for c in protein_chains if c.paired_msa is None
# :215
if len(unique_protein_seqs) > 1:
    _query_server(unique_protein_seqs, use_pairing=True, ...)
```

| set on the binder | unpaired query | paired query | binder sequence leaves? |
|---|---|---|---|
| nothing | included | included | **yes, twice** |
| `unpaired_msa` only | excluded | **still included** | **yes, via pairing** |
| `unpaired_msa` + `paired_msa=""` | excluded | excluded | no |

**The middle row is the trap.** Suppressing the unpaired MSA does not keep the
binder out of the paired query; that is a separate gate on a separate field.
`""` counts as set, since the check is `is None`, so empty string is a valid
suppression value.

**A paired MSA is generated per chain, for every chain in the query** -- the
binder included, if not suppressed. The query goes out once with all
unsuppressed chains and the results are zipped back per sequence and assigned
individually (`:246`).

That matters beyond egress. Pairing keeps only species present across *all*
queried chains, so including a homolog-free de novo binder should collapse the
paired MSA for the **targets** as well, not merely yield an empty one for the
binder. The per-chain generation is verified in the source; the
intersection-by-taxonomy behaviour is standard ColabFold pairing semantics,
server-side, and is **[unconfirmed]** here. Either way, suppressing the binder
protects target pairing depth as well as keeping the sequence in-house.

### Worked example: binder + one target, binder suppressed

Both binder fields set, target left at `None`:

| step | line | outcome |
|---|---|---|
| unpaired list | `:195` | `[target]` -- binder excluded, its field is set |
| unpaired query | `:199` | fires, target gets a real a3m |
| paired list | `:212` | `[target]` -- binder excluded |
| paired query | `:215` | `len == 1`, not `> 1`, so **never fires** |
| early return | `:235` | unpaired list non-empty, so does *not* return |
| applied to target | `:244`, `:246` | unpaired <- real MSA; paired <- `''` |

So the target ends with a real unpaired MSA and `paired_msa = ''` -- empty
string, not `None`. That is not a loss: with one target there is nothing to pair
*with*. Pairing only means anything with two or more unsuppressed chains, i.e.
two or more target chains. The alphaV/beta3 campaign has two distinct targets,
so pairing does fire between them, and suppressing the binder is what keeps it
out of that pair.

**The outcome flips once the target is pre-populated**, which is what a cache
creates. Set the target's `unpaired_msa` from cache and leave its `paired_msa`
as `None`:

- the unpaired list is now empty, every chain having one set
- `:235` checks **only** the unpaired list, so the early return fires
- `return fold_input` unchanged -- the target's `paired_msa` stays `None` and is
  never defaulted to `''`

And with two or more targets it is worse: the paired query at `:215` runs
*before* that early return, so the sequences go out and the answer is then
discarded.

**So do not rely on `fill_missing_msas` for anything once you pre-populate.**
Set every field on every chain explicitly and treat that function as a no-op
being deliberately bypassed. The plan's own `msa_client.py` is what does the
querying; this function is upstream's convenience path, and it is only safe when
nothing has been filled in already.

### What is worth caching

**Cache the unpaired MSAs, not the paired ones.** An unpaired MSA is a property
of a single sequence, so it is context-free and cacheable by sequence. A paired
MSA is a property of the *set* of chains queried together, so it is not
cacheable in general -- the same chain paired against a different partner gives
a different answer.

One consequence specific to this pipeline, worth noticing: because the binder is
always suppressed, the pairing context is exactly the target set, and the
targets do not change across a campaign. So for a given campaign the paired MSA
for the targets is also constant, and could be computed once rather than per
prediction -- 640 queries saved at this campaign's full scale (80 designs x 8
sequences). That holds *only* while the
binder is excluded from pairing; include it and the context becomes per-design
and nothing about pairing is cacheable. **[unconfirmed]** -- the reasoning
follows from the code above, but no paired query has been run yet.

### Two further quirks in that function

**The paired query needs more than one unsuppressed chain** (`:215`). With a
single target, pairing never fires and that chain's `paired_msa` falls back to
`''` (`:246`) -- no paired MSA at all, by omission rather than by decision.

**Identical target sequences collapse to one.** `dict.fromkeys` dedupes by
sequence, so a homodimer target gives `len == 1` and pairing is skipped
entirely. This is the mechanism behind the standing note that sequence identity
between target chains is a runtime question affecting MSA pairing alone.

**The early return discards paired results it has already paid for.** `:235`
checks only `protein_seqs_needing_msa`, the *unpaired* list:

```python
if not protein_seqs_needing_msa and not rna_needing_stub:
    return fold_input
```

The paired query at `:215` has already run by then. So setting `unpaired_msa`
everywhere while leaving `paired_msa` as `None` fires the paired query, sends
the sequences, then returns early and **throws the answer away**. The egress is
paid and nothing is gained -- the strongest reason to set both fields on every
chain rather than relying on the unpaired gate.

**Optimus `mmseqs2.py` has a colliding cache key.** It keys on
`f"{prefix}_{mode}"` (`:179`) and `generate_msa` passes the caller's `msa_dir`
as the prefix (`:293`), so two sequences sharing a directory resolve to the same
cached tarball. Its download timeout is also 6.02 s (`:140`), too short for a
large tarball.

**Seed flags differ per model, and not only in name.** esmfold2 and boltz2 take
an explicit `--seeds a,b,c`; the AF3 fork takes `--num_seeds N` and *generates*
the seeds, so the ones actually used are readable only from its output JSON.

**A refolding model that does not read MSAs is rejected at config time.**
[settled 2026-10-08] Holo folds the binder against its targets, and a target
folded from sequence alone is a worse model of the thing being bound -- the
interface metrics would still be produced, describing a complex nobody asked
for. This was a warning in `resolve_campaign` until the user decided the choice
should be deliberate: naming a refolding model means confirming it supports
alignments.

Rejected in the schema rather than at the stage, so it fails on the client in
seconds instead of after RFD3 and MPNN have spent GPU time.

The practical effect is that **`esmfold2-fast` cannot be a refolding model.**
It stays in the enum and the capability table so the schema can name what it is
and refuse it for this role, rather than the choice silently degrading holo.

**`supports_templates=False` for boltz2 is a deliberate false, not a measured
one** -- held that way so a campaign cannot silently request something that may
be ignored. See the table above.

---

## 10. Working on `app.py` itself

The two module rules in section 3 are the whole of it, and both were learned the
expensive way. The module docstring (`app.py:10-19`) states them.

One more: **the `experiments/` probe scripts still use `capture_output=True` and
`stderr_tail`.** They will go blind exactly as `app.py` did. Left as-is on
purpose, but do not reach for them expecting live output.
