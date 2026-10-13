# Adding Boltz-2 as a refolding **or** cross-validation model

Plan only — no code written yet. Every claim below was checked against the kit
at `~/ubm/boltz2` and the stock wheel on **2026-10-13**, with the source line
cited. Where something is still unverified it says so.

Goal: `refolding.model: boltz2` and `cross_validation.model: boltz2` both work.
Not both in one campaign — `CampaignConfig` already enforces
`cross_validation.model != refolding.model`, and that stays.

## What the kit is

`boltz2` in `anthropics/uplifting-biomolecular-modeling`: Boltz-2 (`boltz`
2.2.1, commit `cb04aecc`), MIT, cofolding, modes `off · exact · fast · big`
plus `--n_gpu P`. Materialise with `git sparse-checkout add boltz2` in `~/ubm`
(4.2 MB under the blob filter; already done).

Stack: Debian 12, Python 3.11.12, CUDA 13.0, torch 2.12.0+cu130, triton 3.7.0,
cuequivariance 0.10.0, numpy 1.26.4. **Host driver 580+.**

That driver line looked like the blocker and is not: the ESMFold2 kit we run
every campaign is CUDA 13.0 / torch 2.13.0+cu130, and CUDA 13 requires 580+
anyway. The requirement is already met in production.

## Verified against our assumptions

| thing | status |
|---|---|
| `bash run.sh pred` | same verb |
| `--config` / `--mode` / `--input` / `--out_dir` | same flag names |
| cards `a100 \| h100 \| h200` | `GPU_TO_KIT_CARD` already maps exactly these |
| modes, default `fast` | matches `RefoldingMode` |
| `--n_gpu 2\|4\|8` under `big` | matches `n_gpu_shard` |
| `--seeds a,b` | `seed_style: LIST` is correct — "every input at every seed in one worker" |
| exit 3 = mode refused, never silent fallback | `_refold` already special-cases it |
| `MODEL_OPT_JIT_ROOT` | we already have a jit volume |
| pLDDT in the CIF B-factor | `to_mmcif(..., plddts=plddts)` — existing extraction works unchanged |
| GPU floor | compute capability 8.0; A100 qualifies |

Cheaper than ESMFold2 per call: **≈25 s start-up** in every mode against
ESMFold2's ~180 s. Chunking still pays, but less.

## What does not match — the actual work

1. **No `--variant` flag exists.** `folding_variant()` returns `None` and
   `_refold` *raises* on that (`app.py:1019`). The variant is an ESMFold2
   concept conflating model choice with MSA presence; Boltz has nothing like
   it. Code change, not a config row.
2. **`--det 1` is `off`-only.** We pass it unconditionally; on `fast` that is a
   usage error (exit 2).
3. **Input is Boltz's native YAML**, not ESMFold2's JSON list.
4. **Output layout is entirely different** — `<out_dir>/by_seed/<name>/s<seed>/`
   (`opt/boltz2_opt/stack.py:1368`) plus `<out_dir>/pred_worker.log`. No
   `pred_rows.jsonl`, no `cif_all/`.
5. **The PAE npz holds only `pae`** (`stock/src/boltz/data/write/writer.py:236`:
   `np.savez_compressed(path, pae=...)`). ESMFold2's carries `asym_id`, which is
   what `ipsae_from_tokens` uses for per-token chain labels.
6. **No `msa_depths`** — compute from the a3m we supply.

## The MSA convention is the inverse of AF3's

`schema.py:1109-1139` and `main.py:580`:

| `msa:` value | `msa_id` | behaviour |
|---|---|---|
| `empty` | −1 | single-sequence mode |
| `""` / `None` / absent | 0 | **auto-generate** — `RuntimeError` unless `--use_msa_server` |
| `path.a3m` | custom | reads that alignment |

AF3, which we already drive for `af3_fork`, is the other way round: `""`
suppresses and absent fetches. `_refold` writes `"msa": None` for the binder
with the comment *"leaving `msa` unset would send it to the MSA server"* —
correct for ESMFold2, and for Boltz `None` **is** that case. A de novo binder
has no homologs, so it must get `msa: empty`.

Two consequences:

* **`msa_paired=False` is correct** and the row's UNVERIFIED flag can go —
  but for a stronger reason than first written here. There is no paired-MSA
  field, *and* **a custom a3m gets no cross-chain pairing at all**:
  `main.py:618` calls `parse_a3m(msa_path, taxonomy=None, …)` on every
  prediction, so the `if taxonomy and header.startswith(">UniRef100")` branch
  at `parse/a3m.py:48` is dead and every sequence is read with
  `taxonomy_id = -1`. Pairing happens only on the `--use_msa_server` path,
  where `msa_pairing_strategy` is handed to the server and the server returns
  the pairing.

  Corrected 2026-10-13: an earlier draft of this plan said Boltz pairs
  internally from our `>UniRef100` headers and that the cache therefore had to
  preserve them. The cache *does* preserve them — 2,658 of 4,481 headers in
  `msa_cache/a4766…a3m` are `>UniRef100_…`, the rest being `SRR`/`ERR`/`MGYP`
  environmental hits with no taxonomy — but it makes no difference, because
  nothing reads them.

  The real consequence: **feeding the cache gives Boltz unpaired information
  only.** That is a known limitation, not a bug, and it may not matter — the
  one measurement we have says 1,771 paired rows per target chain changed AF3's
  ipTM, pTM and ranking_score not at all (`Notes.md`, 2026-10-09, one
  sequence). Whether Boltz is likewise indifferent is unmeasured. The
  alternative is `--use_msa_server`, which abandons the cache and re-fetches
  per prediction, so it is not obviously worth it.
* `empty` sets neither the custom nor the auto flag, so binder-`empty` plus
  targets-with-paths avoids `"Cannot mix custom and auto-generated MSAs in the
  same input!"` (`schema.py:1311`). The combination we need is legal.

## Capability row, as verified

```python
FoldingModel.BOLTZ2: ModelCapabilities(
    uses_kit=True,
    # `templates: - cif: path/to/template.cif` (schema.py:980). Whole-CIF, not
    # per-chain, so `target_template` needs a decision -- see Open questions.
    supports_templates=True,
    msa_unpaired=True,
    # Verified 2026-10-13: no paired field; Boltz pairs by taxonomy from
    # UniRef100 headers. Was flagged UNVERIFIED.
    msa_paired=False,
    seed_style=SeedStyle.LIST,
    # No variant concept at all -- neither `derived_variants` nor
    # `fixed_variant`, which is what forces change (1) below.
)
```

`default_weights` stays unset: one checkpoint set, so `weights:` in the YAML is
correctly rejected today (`weights is meaningful only for models that ship
several checkpoints, not boltz2`).

## Work, in the order it should happen

Cheap checks first, GPU last — the habit that found the `af3_jax` kernel
refusal for pennies and saved two campaign relaunches.

### Phase 1 — image and weights, no pipeline code (~1 h)

`experiments/boltz2_image.py`, following `experiments/af3_fork_image.py`: build
the image, run `run.sh check`, and in a second stage fold the kit's own bundled
`inputs/1BRS_barnase_barstar.yaml`. Stages, each independently runnable:

* `build` — image from `boltz2/environment/Dockerfile` + `requirements.lock`.
* `weights` — `run.sh install --weights /weights/boltz2` into a volume. The
  cache holds `boltz2_conf.ckpt`, `boltz2_aff.ckpt`, `ccd.pkl`, `mols.tar` and
  `mols/`, must be **writable until complete**, and `BOLTZ_CACHE` must name it
  **in every shell that runs a command** (STOCK.md, Pin).
* `check` — `run.sh check --config a100 --mode fast`, expecting `DRY-RUN ok:`.
* `fold` — the bundled input; **record the real output tree**, since phase 3
  depends on the exact file names.

Exit 5 (`KERNELS-REFUSED`, cuEquivariance) is the risk to watch here — the same
failure class as `af3_jax`, though with its own pins and a documented exit code
rather than a silent dead end. Phase 1 is where we find out, for pennies.

**Gate: do not start phase 2 until `fold` has written a prediction.**

### Phase 2 — config (~30 min)

* Fill the capability row above; drop the UNVERIFIED comment.
* `_refold`'s variant check becomes conditional: a model whose row has neither
  `derived_variants` nor `fixed_variant` passes no `--variant`, and only a
  model that *should* have one and does not still raises.
* `--det 1` only when `mode == off`.
* Optional guard: reject `mode: exact` on an A100 when the token count lands in
  a refused band (below), at config time rather than mid-campaign.

### Phase 3 — the input builder and output adapter (~3 h, the real work)

**Input** — `_boltz_fold_inputs`, parallel to `_cv_fold_inputs`:

```yaml
version: 1
sequences:
  - protein: {id: A, sequence: <designed>, msa: empty}      # suppressed
  - protein: {id: B, sequence: <target>, msa: <rel>.a3m}    # cached
  - protein: {id: C, sequence: <target>, msa: <rel>.a3m}
  - ligand:  {id: E, ccd: MN}
```

One YAML per sequence, since `--input` takes a file or a directory of them;
a directory is the natural batch unit and keeps `id`/`name` addressable.

`ligand` takes `ccd` **xor** `smiles`, asserted mutually exclusive
(`schema.py:1029-1030`), and `ccd` is a **single string** where AF3 wants
`ccdCodes: ["MN"]`. So this is a *third* ligand builder — Notes.md's "Two
schemas, not one, for ligands" becomes three. Do not try to unify them; the
same reasoning applies (ESMFold2 caps at one component, AF3 does not, Boltz
takes one code per entity).

**Output** — `_boltz_metrics`, replacing the `pred_rows.jsonl` read:

| we need | Boltz writes |
|---|---|
| `ptm`, `iptm` | `confidence_<id>_model_<rank>.json` — also `protein_iptm`, `ligand_iptm`, `complex_plddt`, `pair_chains_iptm` |
| the structure | `<id>_model_<rank>.cif`, pLDDT already in B-factor |
| PAE for ipSAE | `pae_<id>_model_<rank>.npz`, key `pae` only |
| per-token chain labels | **not in the PAE npz** |
| `msa_depth` | nothing — count from the a3m we wrote |
| `n_tokens`, `wall_s` | nothing — derive / time it ourselves |

For the chain labels, take them from the structure npz written beside the PAE,
whose `chains` array carries `asym_id` (`boltz/data/types.py:136,169-175`).
**Do not** reconstruct them by assuming one token per residue: the MN ligand
tokenises separately, so a residue-count guess would mis-slice every holo PAE
and return plausible wrong numbers. `ipsae_from_tokens` already refuses a label
list that does not match the matrix, and that check is the safety net — it
turns a bad construction into an error instead of a quiet wrong ipSAE.

`protein_iptm` is worth recording: it is ipTM over protein chains only, i.e.
without the one-token metal, which is the same distortion `ipsae_from_tokens`
avoids with `keep`.

### Phase 4 — the two wrappers (~2 h)

Both registered, so either slot can name it:

* `refold_boltz2` → `REFOLD_FUNCTIONS["boltz2"]` — one seed, gated, apo and
  holo, reusing `_refold`'s staging and `REFOLD_SUPERSEDES`.
* `cross_validate_boltz2` → `CV_FUNCTIONS["boltz2"]` — `num_seeds` seeds via
  `--seeds a,b,c` in **one** worker (cheaper than the fork, which needs a
  process per seed), keeping the best by `confidence_score` and reporting the
  spread as `cv_*_best/_mean/_worst`.

Separate functions because a Modal function's image is fixed at decoration
time; one image, two wrappers.

As a refolding model note it is a **cofolding** model, so holo is its natural
shape and apo (binder alone) is the odd one — fine, but `msa: empty` plus a
single chain is exactly the single-sequence mode upstream warns is suboptimal.
Apo is a pLDDT gate, so that is acceptable; worth stating rather than
discovering.

### Phase 5 — prove it on real shards, then calibrate (~1 h + GPU)

`refold_only --stage holo --limit 4` and `cross_validate_only --limit 4`
against an existing run's shards, exactly as the fork was proven. Then a dry
run, then a campaign.

**Its numbers are not comparable to ESMFold2's or the fork's until calibrated.**
ipTM, ipSAE and pLDDT distributions are model-specific — run 02/03 showed the
fork's ipTM running a median 0.22 below holo's on identical sequences. A new
model means new distributions and therefore new thresholds, on its own run.

## Which mode, and the MSA A/B — both to be decided, not defaulted

### Mode for cross-validation

**Today CV uses no mode at all.** `mode` is a `_FoldingBase` field defaulting to
`Mode.FAST` (`config.py:527`), so `CrossValidationConfig` inherits it, but as
`kit_config`'s docstring says it "is defaulted on this base, so a campaign
cannot be stopped from setting it, but nothing reads it for a non-kit model."
`af3_fork` is called directly, so no `--mode` exists; the campaign YAML omits
it deliberately. Refolding, a kit, sets `mode: fast` outright.

The moment Boltz-2 is the CV model, `uses_kit=True` and **`mode` becomes live
and would silently take the `FAST` default.** State it in the YAML instead, so
the setting is visible where the stage is configured.

**Recommend `fast`**, and for a reason rather than because it is the kit's
default: cross-validation already reports a five-seed spread
(`cv_iptm_best/_mean/_worst`), and `fast` is documented as differing from stock
only "within the tool's own seed-to-seed variation". `exact`'s bitwise
guarantee therefore buys nothing this stage can use, while costing throughput
and pulling in the A100 refused-band risk below. `big` only if `fast` runs out
of memory — at ~531 tokens it will not.

For refolding the same argument applies and more strongly: apo and holo are a
one-seed *filter*, and a gate wants a verdict, not bitwise reproducibility.

### Does our unpaired cache cost anything? (phase 5, 4 sequences)

Specified now because it is the experiment that decides whether the pairing
limitation above matters, and because it mirrors what was already done for the
fork on 2026-10-09 (one sequence, cached vs `--use_msa_server`: 1,771 paired
rows per target chain changed ipTM, pTM and ranking_score not at all).

Four sequences, three arms, same seeds throughout:

| arm | `msa:` per chain | flags |
|---|---|---|
| A — cache | binder `empty`, targets `<hash>.a3m` | none |
| B — server | binder `empty`, targets `""` | `--use_msa_server` |
| C — control | every chain `empty` | none |

Arm B is the point: the server generates **paired and unpaired** alignments and
returns the pairing, which is the only route by which Boltz receives pairing at
all. Arm C bounds the comparison from below — if A ≈ C, the cache is buying
nothing and the MSA is not being read the way we think.

Compare `iptm`, `protein_iptm`, `ptm`, `confidence_score`, `ipsae_min_<target>`
and `pae_floor_<target>`, plus placement and motif RMSD. Four sequences settles
nothing statistically; it is a check for a *large* effect, which is what would
change the design. If B clearly beats A, the choice becomes paying the server
per prediction against caching unpaired only — and that is a cost decision to
take with numbers, not now.

Note arm B needs `msa: ""` on the targets, **not** absent-and-hoping: `""` and
absent both mean auto-generate, and auto-generate without `--use_msa_server` is
a `RuntimeError`, so the flag and the field have to move together. Mixing arm A
and arm B chains in one input is refused outright ("Cannot mix custom and
auto-generated MSAs in the same input!").

## The `exact`-on-A100 caveat

`STOCK.md:42`: on A100 (cc 8.0) `exact`'s fused pair transition has **no
per-call fallback**. For a token count whose pair-row count the shared core has
not established as bitwise with this card's cuBLAS, the prediction *completes*
and then exits 3 with `NOT ACTIVE: fused transition gate refused: …
exact_rows_…_on_cc_8.0`.

Refused bands, N tokens: **< 64**, 1,024–1,099, 1,446–1,502, 1,772–1,818,
2,046–2,087, 2,288–2,324, 2,507–2,540, 2,708–2,738, 2,895–2,924.

It is a refusal to *certify*, not a numerical failure: `exact` promises output
bit-identical to stock, and the kit will not claim that for a result it cannot
guarantee. `fast` and `big` serve every size; H100/H200 (cc 9.0) serve every
size in every mode.

Our measured counts — apo ~111, holo 521–538, CV ~531 — are clear of every
band, and we run `fast`. Two ways it could bite later: a binder under 64
residues would hit the `N < 64` band in apo (our contig is `100-120`), and
`mode: exact` on A100 with a 1,024–1,099-token complex would raise in
`_refold` *after* paying for the prediction. Both are cheap to guard at config
time.

Also A100-specific: `msa_pwa_exact` (`exact`) and `atom_fused` (`big`) are
switched off on that card, reported as `card_off=<lever>:<reason>` on the
`ACTIVE` line.

## Open questions

1. ~~Does our MSA cache keep UniRef100 headers?~~ **Answered 2026-10-13: yes
   (2,658 of 4,481), and it turns out not to matter** — `taxonomy=None` makes
   the lookup dead code, so a custom a3m is never paired. See the MSA section.
   What remains open is whether unpaired-only input measurably hurts Boltz on
   this campaign's two targets; that is a question for phase 5, answerable by
   folding a handful of sequences both ways.
2. **How does `target_template` map?** Boltz takes `templates: - cif: <path>`
   for the whole complex, not per chain. `supports_templates=True` is honest
   about the model, but our config models templating as a per-target flag. A
   campaign setting `target_template: true` needs to mean something specific
   before it is allowed.
3. **Refolding or CV first?** Both get wired; only one can be exercised per
   campaign. CV is the cheaper first target — the fork already works there, so
   a disagreement between Boltz-2 and the fork on the same sequences is
   immediately interpretable, and `af3_fork` stays available as the control.
4. **Affinity.** `boltz2_aff.ckpt` ships and `properties: - affinity: {binder}`
   exists in the schema. Out of scope here, worth knowing it is free with the
   weights.

## Estimate

Phases 1–5 ≈ **8 h of work** plus GPU time, assuming phase 1 clears. If exit 5
turns up, stop and treat it as the `af3_jax` case: diagnose the pin conflict
before writing pipeline code, since no amount of wrapper work fixes a kernel
the mode will not engage.
