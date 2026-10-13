"""Probe: drive the sokrypton/alphafold3 fork directly, without the af3_jax kit.

    modal run experiments/af3_fork_image.py --stage env
    modal run experiments/af3_fork_image.py --stage weights
    modal run experiments/af3_fork_image.py --stage predict

**Why not the kit.** `af3_jax` completes its predictions but exits `5
kernels_refused` in *every* mode -- `fast`, `exact` and `off` alike -- because
only 3 of its 16 advertised levers engage. The cause is in its own
`environment/requirements.lock`: `tokamax==0.0.12` against `jaxlib==0.10.2`,
whose bundled MLIR bindings no longer expose the `is_a_*` accessors tokamax
calls, so autotuning dies with

    AttributeError: 'jaxlib...ir.NameLoc' object has no attribute 'is_a_name'

Verified in the kit's running image 2026-10-09: jax, jaxlib and
jax-cuda12-plugin are all 0.10.2 (so *not* a jax/jaxlib mismatch), tokamax
imports fine at 0.0.12, and `dir(ir.NameLoc)` holds **no** `is_a` attribute at
all. Both versions come from the kit, so this is an upstream pin conflict, and
unpinning it means fighting the kit's own `check_pins`.

**Why this is expected to work.** Prior art: the user's own Optimus repo
(`KrajTechne/Optimus`, `modal_common.py` + `RunAlphaFold3.py`) already drives
this fork on Modal, pinning **jax 0.10.1 and tokamax 0.0.11** -- one minor
version below the kit's broken pair on each. Those pins are replicated here
deliberately rather than floated to latest.

**No Dockerfile, no kit checkout, no CUDA build.** `debian_slim` plus the
fork's published wheel. The af3_jax kit needed 435 s and a sparse checkout with
two Windows-specific workarounds; this needs neither. The fork ships
`run_alphafold.py` and `convert_of3_weights.py` *outside* the wheel, as
standalone scripts, so both are fetched separately -- pinned to the wheel's tag
rather than Optimus's `refs/heads/main`, so the scripts cannot drift away from
the wheel they drive.

Deliberate departures from Optimus, both for the same reason -- this is a
cross-validation stage inside a campaign, not a one-off prediction:

* **No `--use_msa_server`.** Optimus lets the fork query api.colabfold.com per
  run. The pipeline's whole MSA design is a content-addressed cache
  (`rfd3_pipeline.msa`) precisely so alignments are fetched once and the binder
  sequence never leaves. Targets here carry `unpairedMsaPath` pointing at that
  cache; the binder carries a single-sequence stub. Nothing reaches the
  network.
* **Weights on the volume, not baked into the image.** Optimus bakes both sets
  into image layers. The repo's convention is a `populate_*_weights` function
  writing under `/weights/<model>`, which also means a weight swap costs no
  rebuild.

What is still missing after this probe, and known to be: **paired** alignments.
`cache_key` already takes `paired=`, but the cache only holds unpaired, so
`pairedMsa` is `""` throughout here. af3_jax is the only model in
`MODEL_CAPABILITIES` with `msa_paired=True`, so that work lands on this stage.
One shortcut applies -- the binder is always suppressed, so the pairing context
is exactly the target set, which is constant across a campaign.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("af3-fork-probe")

data = modal.Volume.from_name("rfd3-pipeline", create_if_missing=True)
weights = modal.Volume.from_name("rfd3-weights", create_if_missing=True)

DATA = "/data"
WEIGHTS = "/weights"
VOLUMES = {DATA: data, WEIGHTS: weights}

# Under /weights/af3_fork, kept apart from /weights/af3 (the kit's own, in the
# kit's own `p2/of3_ported_weights.bin.zst` layout) so neither can be mistaken
# for the other.
FORK_ROOT = f"{WEIGHTS}/af3_fork"
OF3_CHECKPOINT = f"{FORK_ROOT}/of3-p2-155k.pt"
OF3_DIR = f"{FORK_ROOT}/of3"
NATIVE_DIR = f"{FORK_ROOT}/native"

# The JIT/compilation cache. The kit taught us the key embeds GPU model and JAX
# version, so this is scoped by neither -- the fork manages its own subkeys.
AF3_CACHE = f"{WEIGHTS}/af3_fork/jit"

GPU = "A100-40GB"
HOUR = 60 * 60

# Pinned together: the wheel's tag also serves the two standalone scripts.
FORK_TAG = "v3.1.5"
WHEEL_URL = (
    f"https://github.com/sokrypton/alphafold3/releases/download/{FORK_TAG}/"
    "alphafold3_open-3.1.5-cp313-cp313-manylinux_2_27_x86_64"
    ".manylinux_2_28_x86_64.whl"
)
RAW = f"https://raw.githubusercontent.com/sokrypton/alphafold3/refs/tags/{FORK_TAG}"

# v3.1.7 exists and is tempting -- it bundles libcifpp's `components.cif`,
# which would drop the `build_data` step below, and publishes to PyPI as
# `alphafold3_colabfold`. Not taken: v3.1.5's own notes say v3.1.4 -> v3.1.5 was
# "packaging only, no functional changes to inference", and this is the version
# Optimus proved. Upgrade once cross-validation produces numbers, not before.

PROJECT_DEPS = (
    "numpy>=1.26",
    "biotite>=1.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "requests>=2.31",
)

af3_fork_image = (
    # 3.13 because the fork's cp313 wheel is what Optimus proved; a cp312 wheel
    # also exists at this tag if some dependency ever forces 3.12.
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("wget")
    .pip_install(
        # Optimus's pins. See the module docstring: the kit's 0.10.2/0.0.12 is
        # the combination that refuses kernels.
        "jax[cuda12]==0.10.1",
        "tokamax==0.0.11",
        "dm-haiku==0.0.17",
        "rdkit==2025.9.4",
        "zstandard",
        "awscli",
        *PROJECT_DEPS,
    )
    # CPU-only torch, used solely to convert the OpenFold3 checkpoint. The GPU
    # belongs to JAX; a CUDA torch here would just contend for memory.
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .run_commands(f"pip install --no-deps '{WHEEL_URL}'")
    .run_commands(
        f"cd /root && wget -q -O run_alphafold.py {RAW}/run_alphafold.py",
        f"cd /root && wget -q -O convert_of3_weights.py {RAW}/convert_of3_weights.py",
    )
    # The chemical-components database. Weight-independent and one-time, so it
    # belongs in a layer -- the official AF3 Dockerfile runs it at build too.
    .run_commands("cd /root && build_data")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("rfd3_pipeline")
)


def _stream(command: list[str], *, timeout: int, cwd: str | None = None,
            env: dict | None = None, tail_lines: int = 60) -> tuple[int, str]:
    """Run a command, echoing output so the Modal log view shows progress.

    Same reason as `app.py`'s: `capture_output=True` makes a long job silent,
    and this one compiles kernels per token bucket before it predicts anything.
    """
    import os
    import subprocess
    import threading

    merged = {**os.environ, **(env or {})}
    process = subprocess.Popen(
        command, cwd=cwd, env=merged, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    watchdog = threading.Timer(timeout, process.kill)
    watchdog.start()
    lines: list[str] = []
    try:
        for line in process.stdout:
            line = line.rstrip()
            print(line, flush=True)
            lines.append(line)
    finally:
        watchdog.cancel()
        process.wait()
    return process.returncode, "\n".join(lines[-tail_lines:])




# ---------------------------------------------------------------------------
# env -- cheapest first: does the image build, and do the pins behave?
# ---------------------------------------------------------------------------
@app.function(image=af3_fork_image, gpu=GPU, volumes=VOLUMES, timeout=HOUR)
def env() -> dict:
    """Prove the image, the GPU, the scripts, and the tokamax pin.

    The last of those is the point of the whole exercise, so it is asserted
    here rather than discovered during a prediction: the same `NameLoc` check
    that fails in the kit is run against these pins.
    """
    import subprocess

    report: dict[str, object] = {}

    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    report["gpu"] = smi.stdout.strip() or smi.stderr.strip()

    probe_lines = [
        "import importlib.metadata as md",
        "for n in ('jax', 'jaxlib', 'tokamax', 'alphafold3_open', 'dm-haiku', 'rdkit'):",
        "    try: print('  ', n, md.version(n))",
        "    except Exception as e: print('  ', n, 'ABSENT', type(e).__name__)",
        "import jax",
        "print('  devices', jax.devices())",
        "from jaxlib.mlir import ir",
        "attrs = [a for a in dir(ir.NameLoc) if 'is_a' in a]",
        "print('  NameLoc is_a attrs:', attrs or 'NONE')",
        "import tokamax",
        "print('  tokamax imported OK')",
    ]
    versions = subprocess.run(
        ["python", "-c", chr(10).join(probe_lines)], capture_output=True, text=True,
    )
    report["versions"] = (versions.stdout or "").strip()
    if versions.returncode != 0:
        report["versions_stderr"] = versions.stderr[-1200:]

    report["scripts"] = {
        name: Path("/root", name).is_file()
        for name in ("run_alphafold.py", "convert_of3_weights.py")
    }
    from rfd3_pipeline.gpu import attention_tuning

    report["attention_tuning"] = attention_tuning()

    # `build_data` ran at build time; confirm it landed rather than trusting it.
    ccd = subprocess.run(
        ["python", "-c",
         "import alphafold3.constants.converters as c, pathlib, alphafold3;"
         "root = pathlib.Path(alphafold3.__file__).parent;"
         "hits = sorted(p.name for p in root.rglob('*.pickle'))"
         " + sorted(p.name for p in root.rglob('*.bin'));"
         "print(hits[:10] or 'NO DATA FILES')"],
        capture_output=True, text=True,
    )
    report["ccd_data"] = (ccd.stdout or "").strip() or ccd.stderr[-400:]

    helped = subprocess.run(
        ["python", "run_alphafold.py", "--help"], cwd="/root",
        capture_output=True, text=True,
    )
    flags = sorted(
        line.strip().split()[0]
        for line in helped.stdout.splitlines()
        if line.strip().startswith("--")
    )
    report["n_flags"] = len(flags)
    report["key_flags"] = [
        f for f in flags
        if any(k in f for k in ("input_dir", "json_path", "output_dir", "model_dir",
                                "of3", "msa_server", "num_seeds", "diffusion",
                                "force_output", "cache_dir", "flash"))
    ]
    return report


# ---------------------------------------------------------------------------
# weights -- both sets onto the volume
# ---------------------------------------------------------------------------
# Memory and CPU for the conversion: the checkpoint is thousands of tensors
# read into RAM at once, and the two downloads are the rest of the wall time.
@app.function(
    image=af3_fork_image, volumes=VOLUMES, timeout=2 * HOUR,
    cpu=4.0, memory=32768,
)
def populate(native: bool = True) -> dict:
    """Land OpenFold3 (converted) and, optionally, official AF3 weights.

    Idempotent: each step is skipped when its output already exists, so a
    re-run after a partial failure costs only the missing piece.

    `--of3_checkpoint` would let `run_alphafold.py` convert on first use and
    cache into `--model_dir`, but converting here keeps it off the prediction
    path, where it would be paid once per cold container rather than once ever.
    """
    import hashlib
    import subprocess

    report: dict[str, object] = {}
    Path(FORK_ROOT).mkdir(parents=True, exist_ok=True)

    # --- OpenFold3: public, unauthenticated, Apache-2.0 ---
    if not Path(OF3_CHECKPOINT).is_file():
        code, tail = _stream(
            ["aws", "s3", "cp", "s3://openfold/staging/of3-p2-155k.pt",
             OF3_CHECKPOINT, "--no-sign-request"],
            # aws prints a progress line per tick; a long tail of those buries
            # every other field in the report.
            timeout=40 * 60, tail_lines=3,
        )
        report["of3_fetch"] = {"returncode": code, "tail": tail}
        if code != 0:
            return report
    report["of3_checkpoint_bytes"] = Path(OF3_CHECKPOINT).stat().st_size

    converted = list(Path(OF3_DIR).glob("*")) if Path(OF3_DIR).is_dir() else []
    if not converted:
        code, tail = _stream(
            ["python", "convert_of3_weights.py",
             f"--of3_checkpoint={OF3_CHECKPOINT}", f"--output_dir={OF3_DIR}"],
            cwd="/root", timeout=60 * 60,
        )
        report["of3_convert"] = {"returncode": code, "tail": tail}
        if code != 0:
            weights.commit()
            return report
    report["of3_dir"] = sorted(
        f"{p.name} ({p.stat().st_size} B)" for p in Path(OF3_DIR).iterdir()
    )

    # --- Official AlphaFold3: non-commercial terms, used under the clearance
    # the user already holds. Same af3.bin.zst URL they supplied.
    if native:
        Path(NATIVE_DIR).mkdir(parents=True, exist_ok=True)
        target = Path(NATIVE_DIR, "af3.bin.zst")
        if not target.is_file():
            code, tail = _stream(
                ["wget", "-q", "-O", str(target),
                 "https://storage.googleapis.com/alphafold3/af3.bin.zst"],
                timeout=60 * 60,
            )
            report["native_fetch"] = {"returncode": code, "tail": tail}
        if target.is_file():
            digest = hashlib.sha256()
            with target.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 22), b""):
                    digest.update(block)
            report["native"] = {
                "bytes": target.stat().st_size,
                "sha256": digest.hexdigest()[:16],
            }

    Path(AF3_CACHE).mkdir(parents=True, exist_ok=True)
    weights.commit()
    return report


# ---------------------------------------------------------------------------
# predict -- the real thing, on the pipeline's own shards
# ---------------------------------------------------------------------------
@app.function(image=af3_fork_image, gpu=GPU, volumes=VOLUMES, timeout=4 * HOUR)
def predict(
    campaign_payload: dict,
    run_root: str,
    n_seeds: int = 5,
    limit: int = 1,
    native: bool = False,
    msa_source: str = "cache",
) -> dict:
    """One cross-validation-shaped run, assembled the way the real stage will be.

    Five seeds at **one** diffusion sample each, which is the cheap half of a
    decision worth stating: `--num_diffusion_samples` defaults to **5**, so
    leaving it alone would turn five seeds into twenty-five predictions per
    sequence. Seeds vary the whole trajectory, samples only the diffusion
    branch, so seeds are the better spend for a cross-check.

    Chains are mapped to their cached alignment through `cache_key`, not by
    sorting the cache directory -- positional pairing is the mistake that made
    `SAME DESIGN SET: True` meaningless earlier in this project.

    `msa_source="server"` is a **comparison mode only**, not a pipeline option:
    it leaves the targets' MSA fields unset so `--use_msa_server` fetches them
    from api.colabfold.com, to measure what the cache is worth. The binder keeps
    its single-sequence stub in both modes, so the only sequences that leave the
    container are the two public target chains -- the same ones the cache was
    built from. It also fetches *paired* alignments, which the cache does not
    hold, so the two modes differ by pairing as well as by provenance.
    """
    import time

    from rfd3_pipeline.config import CampaignConfig
    from rfd3_pipeline.gpu import attention_tuning
    from rfd3_pipeline.msa import a3m_cache_path, cache_key
    from rfd3_pipeline.spec import resolve_campaign

    data.reload()
    weights.reload()

    campaign = CampaignConfig.model_validate(campaign_payload)
    root = Path(DATA) / run_root
    shards = root / "shards"
    cache_dir = Path(DATA) / campaign.general.name_design_campaign / "msa_cache"

    records = [json.loads(p.read_text()) for p in sorted(shards.glob("*.json"))]
    # Cross-validation exists to second-guess holo, so it has no business
    # running on sequences holo never folded.
    candidates = [
        r for r in records
        if r.get("seq_id") and r.get("sequence") and r.get("holo_status") == "ok"
    ]
    report: dict[str, object] = {
        "shards": len(records),
        "holo_ok_sequences": len(candidates),
        "weights": "af3_native" if native else "openfold3",
    }
    if not candidates:
        report["error"] = f"no holo-folded sequences under {shards}"
        return report

    resolved = resolve_campaign(campaign, structure_root=DATA)
    target_sequences = dict(resolved.target_sequences)
    msa = campaign.msa

    # Target chain -> its cached unpaired alignment, keyed by content.
    alignments: dict[str, str] = {}
    missing: list[str] = []
    for chain in campaign.general.chain_targets:
        sequence = target_sequences.get(chain)
        if sequence is None:
            missing.append(f"{chain} (no target sequence)")
            continue
        key = cache_key(sequence, use_env=msa.use_env, use_filter=msa.use_filter)
        path = a3m_cache_path(cache_dir, key)
        # Probed by stat, not by `read_cached_a3m`: these alignments run to
        # thousands of sequences, and reading one in full only to discard it
        # and let AF3 read it again costs megabytes per chain per container.
        # A truncated write is still worse than a miss, hence the size test.
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(f"{chain} ({key})")
            continue
        # Absolute: the fork resolves a relative path against the *JSON's*
        # directory (`folding_input._read_file`), the same trap RFD3's `input`
        # had. It also decompresses gzip/xz/zstd by magic number, so a
        # compressed cache would work unchanged.
        #
        # A path rather than the text itself, deliberately. The dict is built
        # once per container and shared by every design, so inlining would copy
        # the same ~14k alignment rows into each design's JSON -- megabytes
        # times the campaign, for bytes AF3 would read either way.
        alignments[chain] = str(path)
    if missing and msa_source == "cache":
        report["error"] = (
            f"no cached MSA for target chain(s) {', '.join(missing)} under "
            f"{cache_dir}; run `modal run app.py::cache_msas` first"
        )
        return report
    report["msa_source"] = msa_source
    report["alignments"] = alignments if msa_source == "cache" else "(server)" 

    # Ions are ligands to AF3 (`docs/input.md:93-96`), named by CCD code rather
    # than SMILES so the chemistry comes from the components database.
    ligands: list[dict] = []
    if campaign.rfdiffusion3.ligand:
        from rfd3_pipeline.contig import parse_residue_selection
        from rfd3_pipeline.structure import load_structure

        selection = parse_residue_selection(campaign.rfdiffusion3.ligand)
        structure = load_structure(resolved.structure_path)
        for chain, res_id in selection:
            chosen = structure[
                (structure.chain_id == chain) & (structure.res_id == res_id)
            ]
            names = sorted({str(name) for name in chosen.res_name})
            if not names:
                report["error"] = f"no residue {chain}{res_id} in the input structure"
                return report
            ligands.append({"ligand": {"id": chain, "ccdCodes": names}})
    report["ligands"] = ligands

    work = Path("/tmp/cv")
    in_dir, out_dir = work / "in", work / "out"
    in_dir.mkdir(parents=True, exist_ok=True)

    binder = campaign.general.chain_binder
    seeds = list(range(campaign.general.base_seed, campaign.general.base_seed + n_seeds))
    written = []
    for record in candidates[:limit]:
        entities: list[dict] = [
            {
                "protein": {
                    "id": binder,
                    "sequence": record["sequence"],
                    # A single-sequence stub, not "" and not unset. Unset would
                    # be fetched if --use_msa_server were ever added; the stub
                    # reads as "already set" to `fill_missing_msas` either way,
                    # and a de novo binder has no homologs to find.
                    "unpairedMsa": ">query" + chr(10) + record["sequence"] + chr(10),
                    "pairedMsa": "",
                    "templates": [],
                }
            }
        ]
        for chain in campaign.general.chain_targets:
            target: dict = {
                "id": chain,
                "sequence": target_sequences[chain],
                "templates": [],
            }
            if msa_source == "cache":
                target["unpairedMsaPath"] = alignments[chain]
                # Unpaired only -- see the module docstring. "" counts as set,
                # so this suppresses pairing rather than searching for it.
                target["pairedMsa"] = ""
            # msa_source == "server": every MSA field left absent, which is the
            # only state `fill_missing_msas` treats as "go and fetch".
            entities.append({"protein": target})
        entities.extend(ligands)

        payload = {
            "dialect": "alphafold3",
            # 1, 2, 3 and 4 are all accepted (`folding_input.JSON_VERSIONS`).
            "version": 1,
            "name": record["seq_id"],
            # Explicit, never `--num_seeds`: the fork uses the input's seeds
            # when the flag is absent, so these stay ours and the run repeats.
            "modelSeeds": seeds,
            "sequences": entities,
        }
        (in_dir / f"{record['seq_id']}.json").write_text(json.dumps(payload, indent=2))
        written.append(record["seq_id"])

    report["seq_ids"] = written
    report["model_seeds"] = seeds
    report["chains"] = [
        f"{list(e)[0]}:{e[list(e)[0]]['id']}" for e in entities
    ]

    implementation, xla_flags, nojit = attention_tuning()
    command = [
        "python", "run_alphafold.py",
        f"--input_dir={in_dir}",
        f"--output_dir={out_dir}",
        f"--model_dir={NATIVE_DIR if native else OF3_DIR}",
        f"--cache_dir={AF3_CACHE}",
        # Never AF3's own genetic search -- that needs the local databases the
        # fork exists to avoid. Kept in both modes, as Optimus does: the MSA
        # server path is independent of the data pipeline.
        "--norun_data_pipeline",
        # Without this a re-run appends a timestamp to the job directory, so
        # the output stops being addressable by seq_id.
        "--force_output_dir",
        f"--flash_attention_implementation={implementation}",
        "--num_recycles=10",
        # One sample per seed. See the docstring: the default is 5.
        "--num_diffusion_samples=1",
    ]
    if msa_source == "server":
        command += ["--use_msa_server", "--msa_server_url=https://api.colabfold.com"]
    if nojit:
        command.append("--nojit")
    if not native:
        # Required even with pre-converted weights -- it selects OF3-compatible
        # model settings, not just the conversion.
        command.append("--of3_weights")

    started = time.monotonic()
    code, tail = _stream(
        command, cwd="/root", timeout=3 * HOUR,
        env={"XLA_FLAGS": " ".join(xla_flags)} if xla_flags else None,
    )
    report["seconds"] = round(time.monotonic() - started, 1)
    report["returncode"] = code
    report["output_tail"] = tail

    # Reported whatever the exit code: the kit's `rc=5` came with ten complete
    # predictions, and returning early there would have discarded them.
    produced = sorted(
        str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()
    ) if out_dir.is_dir() else []
    report["n_files"] = len(produced)
    report["produced"] = produced[:40]

    # The one failure worth being loud about: AF3 folding the targets from
    # sequence alone would still produce a full set of interface numbers. It
    # echoes its *resolved* input to `<name>_data.json`, so the alignment depth
    # it actually used is readable rather than assumed.
    depths: dict[str, int] = {}
    for echoed in out_dir.rglob("*_data.json"):
        for entity in json.loads(echoed.read_text()).get("sequences", []):
            chain = entity.get("protein")
            if not chain:
                continue
            for label, field in (("", "unpairedMsa"), ("_paired", "pairedMsa")):
                text = chain.get(field) or ""
                depths[chain["id"] + label] = sum(
                    1 for line in text.splitlines() if line.startswith(">")
                )
    report["msa_depth"] = depths
    expected = set(campaign.general.chain_targets)
    # In server mode the echoed input is the *fetched* result, so a thin chain
    # there means the server returned nothing -- still worth the warning.
    thin = sorted(c for c in expected if depths.get(c, 0) < 2)
    if thin:
        report["msa_warning"] = (
            f"target chain(s) {thin} reached the model with no alignment; the "
            f"interface numbers below were computed from sequence alone"
        )

    # Prefixed with the job name -- `rglob("ranking_scores.csv")` finds
    # nothing, which is why the first run reported no ranking table.
    ranking = next((p for p in out_dir.rglob("*ranking_scores.csv")), None)
    if ranking is not None:
        report["ranking_scores_csv"] = ranking.read_text()[:800]
    summaries = sorted(out_dir.rglob("*_summary_confidences.json"))
    report["n_summaries"] = len(summaries)
    if summaries:
        first = json.loads(summaries[0].read_text())
        report["summary_keys"] = sorted(first)
        report["summary_sample"] = {
            k: first[k] for k in ("iptm", "ptm", "ranking_score", "has_clash",
                                  "fraction_disordered", "chain_iptm")
            if k in first
        }
    return report


@app.local_entrypoint()
def main(
    stage: str = "env",
    campaign: str = "inputs_motif_scaffold_alphav_beta3/alphav_rgd_unindex_claude.yaml",
    run_root: str = "motif_scaffold_alphav_beta3/alphav_beta3_01-dryrun",
    n_seeds: int = 5,
    limit: int = 1,
    native: bool = False,
    msa_source: str = "cache",
):
    """`--stage env|weights|predict`.

    `--native` picks the official AF3 weights over OpenFold3, on both `weights`
    (fetch them) and `predict` (use them).
    """
    import yaml

    stage = stage.replace("-", "_")
    if stage == "env":
        report = env.remote()
    elif stage == "weights":
        report = populate.remote(native=native)
    elif stage == "predict":
        from rfd3_pipeline.config import CampaignConfig

        payload = yaml.safe_load(Path(campaign).read_text())
        CampaignConfig.model_validate(payload)  # fail on the client, in seconds
        if msa_source not in ("cache", "server"):
            raise SystemExit(f"--msa-source must be cache|server, got {msa_source!r}")
        report = predict.remote(
            payload, run_root=run_root, n_seeds=n_seeds, limit=limit,
            native=native, msa_source=msa_source,
        )
    else:
        raise SystemExit(f"unknown stage {stage!r}: env|weights|predict")

    for key, value in report.items():
        print(f"### {key}")
        print(value)
        print()
