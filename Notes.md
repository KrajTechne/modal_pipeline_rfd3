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

## 8. Stages not built yet (apo/holo refolding, cross-validation)

Recorded now because the traps are already known and all are the silent kind.
Verified against upstream source rather than docs, dated per item -- most on
2026-09-28, the `msa_server.py` mechanics on 2026-10-07. Re-check if those repos
move. Worth reading *before* the apo stage is written rather than after: the
egress trap below is the reason this section exists.

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

| model | consumes_msa | templates | seeds | weights | variant (no-msa / msa) |
|---|---|---|---|---|---|
| `esmfold2` | **yes** | no | `--seeds a,b,c` | -- | `full_nomsa` / `full_msa` |
| `esmfold2-fast` | **no** | no | `--seeds a,b,c` | -- | `fast` / `fast` |
| `boltz2` | **yes** | no *[unconfirmed]* | `--seeds a,b,c` | -- | none / none |
| `af3_jax` | **yes** | yes | `--num_seeds N` | `openfold3` | `p2` / `p2` |

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

**Refolding models that do not read MSAs** still get the cached target MSA
built; resolution emits a warning rather than skipping the work. On
`esmfold2-fast` that warning is the signal that the whole cache step is dead
weight for the campaign, not a problem to fix.

**`supports_templates=False` for boltz2 is a deliberate false, not a measured
one** -- held that way so a campaign cannot silently request something that may
be ignored. See the table above.

---

## 9. Working on `app.py` itself

The two module rules in section 3 are the whole of it, and both were learned the
expensive way. The module docstring (`app.py:10-19`) states them.

One more: **the `experiments/` probe scripts still use `capture_output=True` and
`stderr_tail`.** They will go blind exactly as `app.py` did. Left as-is on
purpose, but do not reach for them expecting live output.
