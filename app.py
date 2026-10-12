"""The RFD3 pipeline on Modal.

One app, one image per step, orchestration in a remote function so a campaign
survives the laptop closing:

    modal run --detach app.py::run --campaign inputs_.../alphav_rgd_unindex_claude.yaml
    modal run app.py::install_weights
    modal run app.py::reduce_only --run-name alphav_beta3_01

Two rules the whole file follows, both learned the hard way:

* **Nothing that is not in the container may be imported at module level.**
  Modal imports this file inside the container as well as locally, so a stray
  top-level import crashes every container before its function body runs -- on
  a GPU function, that means paying for an A100 to fail on an import.
* **Nothing crossing a function boundary may be a library type.** The campaign
  travels as a plain dict and is re-validated on the far side; a pydantic model
  or an AtomArray would fail to unpickle wherever its library is absent.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "rfd3-pipeline"
app = modal.App(APP_NAME)

# ---------------------------------------------------------------------------
# Volumes. One subfolder per model under weights, created when first needed.
# ---------------------------------------------------------------------------
data_volume = modal.Volume.from_name("rfd3-pipeline", create_if_missing=True)
weights_volume = modal.Volume.from_name("rfd3-weights", create_if_missing=True)
jit_volume = modal.Volume.from_name("rfd3-jit", create_if_missing=True)

DATA = "/data"
WEIGHTS = "/weights"
JIT = "/jit"
RFD3_CHECKPOINTS = f"{WEIGHTS}/rfd3"
MPNN_CHECKPOINTS = f"{WEIGHTS}/ligandmpnn"
ESM_WEIGHTS = f"{WEIGHTS}/esmfold2"
ESM_JIT = f"{JIT}/esmfold2"
ESM_KIT = "/kit/esmfold2"

# Cross-validation on the sokrypton/alphafold3 fork, driven directly. Kept
# apart from `{WEIGHTS}/af3`, which holds the Anthropic kit's own layout, so
# neither can be mistaken for the other.
AF3_FORK_ROOT = f"{WEIGHTS}/af3_fork"
AF3_FORK_OF3 = f"{AF3_FORK_ROOT}/of3"
AF3_FORK_NATIVE = f"{AF3_FORK_ROOT}/native"
AF3_FORK_CHECKPOINT = f"{AF3_FORK_ROOT}/of3-p2-155k.pt"
AF3_FORK_JIT = f"{JIT}/af3_fork"

# The kit's `--config` card follows the GPU, not the model. `GPU_TO_CARD` in
# config.py is the validated mapping; this is its string form for the CLI.
GPU_TO_KIT_CARD = {
    "A100": "a100", "A100-40GB": "a100", "A100-80GB": "a100",
    "H100": "h100", "H200": "h200",
}

VOLUMES = {DATA: data_volume, WEIGHTS: weights_volume, JIT: jit_volume}

# ---------------------------------------------------------------------------
# Images. `add_local_python_source` must come last: no build step may follow it.
# ---------------------------------------------------------------------------
PROJECT_DEPS = (
    "numpy>=1.26",
    "biotite>=1.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    # The MSA client's only dependency. The ESMFold2 kit image already has it
    # via its own lock, so this is for `cpu_image`, where the cache is built.
    "requests>=2.31",
)

# CPU work: spec resolution, backbone filters, the reducer. No model, no GPU.
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*PROJECT_DEPS)
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("rfd3_pipeline")
)

# RFD3. rc-foundry pins >=3.12,<3.13 and brings torch with the default CUDA
# wheel, so no index-url juggling is needed.
rfd3_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("rc-foundry[rfd3]==0.2.0")
    .pip_install(*PROJECT_DEPS)
    .env(
        {
            "FOUNDRY_CHECKPOINT_DIRS": RFD3_CHECKPOINTS,
            "MODEL_OPT_JIT_ROOT": JIT,
            # Without this, a child process's output sits in its pipe buffer and
            # the log view stays empty until the step ends.
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_python_source("rfd3_pipeline")
)

# LigandMPNN. Vendored at `LigandMPNN/` rather than cloned, because its bundled
# openfold used numpy aliases removed in 1.24 and upstream therefore pins
# numpy==1.23.5 -- which has no cp312 wheels. With those six occurrences fixed
# in the vendored copy it runs on 3.12 with numpy 2.x, so it shares one image
# with the project's own code instead of needing a container of its own.
# `requirements.txt` is deliberately not installed: it would drag numpy 1.23.5
# back and pin torch to 2.2.1.
MPNN_CHECKOUT = "/opt/LigandMPNN"
MPNN_DEPS = ("prody", "ml-collections", "dm-tree", "scipy", "biopython")

mpnn_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget")
    .pip_install("torch")
    .pip_install(*MPNN_DEPS)
    .pip_install(*PROJECT_DEPS)
    .env({"PYTHONUNBUFFERED": "1"})
    # copy=True: a copy=False local source may not be followed by further build
    # steps, and `add_local_python_source` has to come last.
    .add_local_dir(
        Path(__file__).resolve().parent / "LigandMPNN",
        MPNN_CHECKOUT,
        copy=True,
        ignore=["__pycache__", "*.pyc"],
    )
    .add_local_python_source("rfd3_pipeline")
)

# `mpnn_type` -> (run.py --model_type, its checkpoint flag, the checkpoint file).
# The flag differs per model type, so it cannot be derived from the name.
MPNN_MODELS = {
    "soluble": ("soluble_mpnn", "--checkpoint_soluble_mpnn", "solublempnn_v_48_020.pt"),
    "protein": ("protein_mpnn", "--checkpoint_protein_mpnn", "proteinmpnn_v_48_020.pt"),
    "ligand": ("ligand_mpnn", "--checkpoint_ligand_mpnn", "ligandmpnn_v_32_010_25.pt"),
}

# ESMFold2. Built from the kit's own Dockerfile rather than assembled here: its
# route C wants uv, a C++ compiler, the CUDA 13.0 toolkit and two Biohub
# packages at pinned commits, and anything off-pin exits 3. The checkout lives
# outside this repo -- `experiments/esmfold2_image.py` documents the two-command
# setup, including the `core.autocrlf=false` and empty-jitcache requirements
# that are not optional. `BUILD_JOBS` is pinned because the Dockerfile's own
# heuristic over-subscribes memory and gets the build OOM-killed on a smaller
# builder. The kit's requirements.lock already carries biotite, numpy, pydantic
# and pyyaml, so `rfd3_pipeline` needs no extra installs here.
ESM_KIT_CONTEXT = Path.home() / "ubm"

esmfold2_image = (
    modal.Image.from_dockerfile(
        ESM_KIT_CONTEXT / "esmfold2" / "environment" / "Dockerfile",
        context_dir=ESM_KIT_CONTEXT,
        build_args={"STACK": "img_esmfold2_a100", "BUILD_JOBS": "8"},
    )
    .env(
        {
            # The kit reads the weights root from HF_HOME, not from a flag.
            "HF_HOME": ESM_WEIGHTS,
            # Without a persistent root, every container pays a cold Triton
            # compile. `install_esm_jit` fills it once.
            "MODEL_OPT_JIT_ROOT": ESM_JIT,
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_python_source("rfd3_pipeline")
)

# The fork, driven directly rather than through the af3_jax kit, which exits
# `5 kernels_refused` in every mode -- see "The af3_jax kit refuses its own
# kernels" in Notes.md for why, and why it is not fixable from here.
#
# No Dockerfile, no checkout, no CUDA build: a published wheel on debian_slim,
# which builds in about a minute against the kit's 435 s.
# Pins copied from the user's Optimus repo, which already drives this fork on
# Modal: a known-working combination, not a floor. jax 0.10.1 and tokamax
# 0.0.11 are each one minor version below the pair the kit cannot run.
AF3_FORK_TAG = "v3.1.5"
AF3_FORK_WHEEL = (
    f"https://github.com/sokrypton/alphafold3/releases/download/{AF3_FORK_TAG}/"
    "alphafold3_open-3.1.5-cp313-cp313-manylinux_2_27_x86_64"
    ".manylinux_2_28_x86_64.whl"
)
# `run_alphafold.py` and `convert_of3_weights.py` are not in the wheel -- the
# fork ships them as standalone scripts. Pinned to the wheel's tag rather than
# `main`, so a script cannot drift away from the wheel it drives.
_AF3_FORK_RAW = (
    f"https://raw.githubusercontent.com/sokrypton/alphafold3/refs/tags/{AF3_FORK_TAG}"
)

af3_fork_image = (
    # 3.13 for the cp313 wheel; a cp312 one exists at this tag if some
    # dependency ever forces 3.12.
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("wget")
    .pip_install(
        "jax[cuda12]==0.10.1",
        "tokamax==0.0.11",
        "dm-haiku==0.0.17",
        "rdkit==2025.9.4",
        "zstandard",
        "awscli",
        *PROJECT_DEPS,
    )
    # CPU-only torch, used solely to convert the OpenFold3 checkpoint. The GPU
    # belongs to JAX; a CUDA torch here would only contend for its memory.
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .run_commands(f"pip install --no-deps '{AF3_FORK_WHEEL}'")
    .run_commands(
        f"cd /root && wget -q -O run_alphafold.py {_AF3_FORK_RAW}/run_alphafold.py",
        "cd /root && wget -q -O convert_of3_weights.py "
        f"{_AF3_FORK_RAW}/convert_of3_weights.py",
    )
    # The chemical-components database: weight-independent and one-time, so it
    # belongs in a layer. The official AF3 Dockerfile also runs it at build.
    .run_commands("cd /root && build_data")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("rfd3_pipeline")
)

HOUR = 60 * 60


def _stream(
    command: list[str],
    *,
    timeout: int,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    tail_lines: int = 60,
) -> tuple[int, str]:
    """Run a command, echoing its output so the Modal log view shows progress.

    `capture_output=True` holds every line in a pipe until the process exits,
    which on a six-hour design job leaves no way to tell a slow run from a hung
    one. Lines are printed as they arrive; a bounded tail is still returned for
    the report. stderr is merged into stdout to keep the interleaving the
    subprocess intended.

    `env` is merged over the inherited environment rather than replacing it --
    the fork needs one `XLA_FLAGS` entry added, not a bare environment.
    """
    import os
    import subprocess
    import threading
    from collections import deque

    tail: deque[str] = deque(maxlen=tail_lines)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=cwd,
        env={**os.environ, **env} if env else None,
    )
    # A watchdog rather than a deadline checked per line, so a process that
    # hangs while printing nothing is still killed.
    watchdog = threading.Timer(timeout, process.kill)
    watchdog.start()
    try:
        for line in process.stdout or ():
            line = line.rstrip()
            tail.append(line)
            print(line, flush=True)
    finally:
        watchdog.cancel()
        if process.stdout is not None:
            process.stdout.close()
    return process.wait(), "\n".join(tail)


def _sequence_records(shards) -> dict[str, dict]:
    """Every MPNN sequence on the volume, keyed by `seq_id`.

    Shared by the refolding and cross-validation stages, which both need the
    whole set to look up the handful in their chunk. Records without a
    `sequence` are dropped: those are backbone shards, which describe the same
    design at an earlier stage and have nothing to fold.
    """
    from rfd3_pipeline.report import read_shards

    return {
        record["seq_id"]: record
        for record in read_shards(shards)
        if record.get("seq_id") and record.get("sequence")
    }


def _campaign_from(payload: dict):
    """Re-validate the campaign on this side of the boundary."""
    from rfd3_pipeline.config import CampaignConfig

    return CampaignConfig.model_validate(payload)


def _run_root(campaign) -> str:
    general = campaign.general
    return f"{DATA}/{general.name_design_campaign}/{general.run_name}"


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
@app.function(image=rfd3_image, volumes=VOLUMES, timeout=2 * HOUR)
def populate_weights() -> dict:
    """Download the RFD3 checkpoint. Idempotent: re-runs hash-check only.

    Its own function rather than a side effect of a step, because the installers
    run inside containers and the volume is then mountable read-only.
    """
    from pathlib import Path

    target = Path(RFD3_CHECKPOINTS)
    target.mkdir(parents=True, exist_ok=True)
    returncode, output_tail = _stream(
        ["foundry", "install", "rfd3", "--checkpoint-dir", RFD3_CHECKPOINTS],
        timeout=2 * HOUR,
    )
    weights_volume.commit()

    files = sorted(
        (str(p.relative_to(target)), int(p.stat().st_size))
        for p in target.rglob("*")
        if p.is_file()
    )
    return {
        "returncode": returncode,
        "files": files,
        "output_tail": output_tail,
    }


# ---------------------------------------------------------------------------
# RFD3
# ---------------------------------------------------------------------------
@app.function(
    image=rfd3_image, gpu="A100-80GB", volumes=VOLUMES, timeout=6 * HOUR
)
def run_rfd3(spec_name: str, campaign_payload: dict) -> dict:
    """Generate every design for one spec key.

    Sharded by `json_keys_subset`, which runs a subset of the input file's keys,
    with `global_prefix` naming the outputs -- the RFD3 config annotates that
    pairing as "(pipelines usage)". `skip_existing` stays on, so a preempted
    container regenerates only what is missing.
    """
    from pathlib import Path

    campaign = _campaign_from(campaign_payload)
    rfd3 = campaign.rfdiffusion3
    root = Path(_run_root(campaign))
    out_dir = root / "rfd3" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "rfd3",
        "design",
        f"out_dir={out_dir}",
        f"inputs={root / 'rfd3' / 'specs.yaml'}",
        f"json_keys_subset=[{spec_name}]",
        "global_prefix=''",
        f"n_batches={rfd3.n_batches}",
        f"diffusion_batch_size={rfd3.diffusion_batch_size}",
        f"inference_sampler.num_timesteps={rfd3.num_timesteps}",
        f"inference_sampler.step_scale={rfd3.step_scale}",
        f"inference_sampler.gamma_0={rfd3.gamma_0}",
        "prevalidate_inputs=True",
        "skip_existing=True",
        "dump_prediction_metadata_json=True",
    ]
    if rfd3.seed is not None:
        # Top-level `seed`, not `inference_engine.seed`: `inference.yaml`
        # selects `inference_engine: rfdiffusion3`, and that option is
        # `@package _global_` inheriting `base`, so base's `seed` merges at the
        # root beside `out_dir` and `n_batches`. Verified by composing the
        # config as `rfd3/cli.py` does -- the nested form is refused with
        # "Could not override", and Hydra's suggested `+inference_engine.seed=`
        # would be worse: it sets a key nothing reads, leaving sampling
        # unseeded while looking configured.
        #
        # Upstream's default is `seed: null`, so passing nothing means unseeded
        # -- which is why two runs of an identical campaign produced different
        # backbones. Appended only when set, so a full campaign keeps that.
        command.append(f"seed={rfd3.seed}")
    print(f"rfd3 design: spec {spec_name} in {root}", flush=True)
    returncode, output_tail = _stream(command, timeout=6 * HOUR, cwd=str(root))
    data_volume.commit()

    designs = sorted(p.stem for p in out_dir.glob(f"*{spec_name}*.json"))
    print(f"rfd3 design: spec {spec_name} exit {returncode}, {len(designs)} designs", flush=True)
    return {
        "spec_name": spec_name,
        "returncode": returncode,
        "n_designs": len(designs),
        "designs": designs,
        "output_tail": output_tail,
    }


# ---------------------------------------------------------------------------
# MPNN
# ---------------------------------------------------------------------------
@app.function(image=mpnn_image, volumes=VOLUMES, timeout=HOUR)
def populate_mpnn_weights() -> dict:
    """Download the MPNN checkpoints into the weights volume.

    Separate from `populate_weights` because it is a different image and a
    different installer, and because a campaign that never runs MPNN should not
    pay for it. All 15 checkpoints are 123 MB together, so re-fetching is cheap
    enough that no hash-check dance is warranted.
    """
    target = Path(MPNN_CHECKPOINTS)
    target.mkdir(parents=True, exist_ok=True)

    # The script takes its destination as $1.
    returncode, output_tail = _stream(
        ["bash", f"{MPNN_CHECKOUT}/get_model_params.sh", str(target)], timeout=HOUR
    )
    weights_volume.commit()

    landed = sorted(
        (str(f.relative_to(target)), int(f.stat().st_size))
        for f in target.rglob("*")
        if f.is_file()
    )
    return {
        "returncode": returncode,
        "files": landed,
        # `wget -q` leaves a zero-byte file behind on failure rather than
        # erroring, so an empty checkpoint would otherwise surface only as a
        # torch.load failure inside the GPU step.
        "empty_files": [name for name, size in landed if size == 0],
        "output_tail": output_tail,
    }


@app.function(image=mpnn_image, gpu="L40S", volumes=VOLUMES, timeout=2 * HOUR)
def run_mpnn(design_ids: list[str], campaign_payload: dict) -> dict:
    """Design sequences for a chunk of backbones, writing one shard per sequence.

    Chunked like `filter_designs`, and for a stronger reason: MPNN is ~7.5 s for
    eight sequences over a 520-residue system, most of it loading the model, so
    a container per design would be almost all startup. `--pdb_path_multi` and
    `--fixed_residues_multi` take JSON maps, so a whole chunk runs in one
    invocation with per-design fixed residues.
    """
    import gzip
    import json
    import shutil

    import prody

    from rfd3_pipeline.report import read_shards, write_shard

    # Same reason as in `filter_designs`: the shards and the designs this reads
    # were written by other containers.
    data_volume.reload()

    campaign = _campaign_from(campaign_payload)
    mpnn = campaign.mpnn
    binder = campaign.general.chain_binder
    root = Path(_run_root(campaign))
    designs_dir = root / "rfd3" / "out"
    shards = root / "shards"
    work = Path("/tmp/mpnn")
    work.mkdir(parents=True, exist_ok=True)

    model_type, checkpoint_flag, checkpoint_file = MPNN_MODELS[mpnn.mpnn_type.value]
    checkpoint = Path(MPNN_CHECKPOINTS) / checkpoint_file
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"{checkpoint} missing; run `modal run app.py::install_mpnn_weights` first"
        )

    # Each design's fixed residues come from its own shard, which is where the
    # backbone filters recorded them. Re-deriving them here would be a second
    # implementation of the numbering, and the reason it lives in exactly one
    # place is that input-vs-output numbering fails silently rather than loudly.
    by_design = {
        record["design_id"]: record
        for record in read_shards(shards)
        if record.get("design_id") and not record.get("seq_id")
    }

    # run.py wants PDB; RFD3 writes gzipped mmCIF. ProDy is MPNN's own parser,
    # so converting with it keeps one opinion about the structure rather than
    # introducing biotite's as a second.
    pdb_paths: dict[str, str] = {}
    fixed: dict[str, str] = {}
    prepared: list[str] = []
    skipped: list[dict] = []
    for design_id in design_ids:
        record = by_design.get(design_id)
        if record is None:
            skipped.append({"design_id": design_id, "reason": "no backbone shard"})
            continue
        residues = record.get("mpnn_fixed_residues")
        if not residues:
            skipped.append({"design_id": design_id, "reason": "no mpnn_fixed_residues"})
            continue
        source = designs_dir / f"{design_id}.cif.gz"
        if not source.exists():
            skipped.append({"design_id": design_id, "reason": "missing cif.gz"})
            continue

        plain = work / f"{design_id}.cif"
        with gzip.open(source, "rb") as src, plain.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        parsed = prody.parseMMCIF(str(plain))
        if parsed is None:
            skipped.append({"design_id": design_id, "reason": "prody parse returned None"})
            continue

        as_pdb = work / f"{design_id}.pdb"
        prody.writePDB(str(as_pdb), parsed)
        # Only the keys are read from the pdb map; the values are ignored.
        pdb_paths[str(as_pdb)] = ""
        fixed[str(as_pdb)] = str(residues)
        prepared.append(design_id)

    if not prepared:
        return {"returncode": 0, "n_sequences": 0, "designs": [], "skipped": skipped}

    multi_pdb = work / "pdbs.json"
    multi_fixed = work / "fixed.json"
    multi_pdb.write_text(json.dumps(pdb_paths))
    multi_fixed.write_text(json.dumps(fixed))

    out_folder = work / "out"
    command = [
        "python", "run.py",
        "--model_type", model_type,
        checkpoint_flag, str(checkpoint),
        "--pdb_path_multi", str(multi_pdb),
        "--fixed_residues_multi", str(multi_fixed),
        "--out_folder", str(out_folder),
        # Without this, MPNN designs EVERY chain: it invents new target
        # sequences and optimises the binder against them. It does not error --
        # it returns plausible sequences for a target that does not exist.
        # Verified 2026-10-07 by diffing the target segments against the native.
        "--chains_to_design", binder,
        "--temperature", str(mpnn.mpnn_sampling_temp),
        "--batch_size", str(mpnn.num_seqs),
        "--number_of_batches", "1",
        "--omit_AA", mpnn.rm_aa,
        # One base seed for the campaign, so a re-run is comparable to its
        # original. MPNN takes a single `--seed`, not a list.
        "--seed", str(campaign.general.base_seed),
    ]
    print(f"mpnn: {len(prepared)} design(s), model {model_type}", flush=True)
    returncode, output_tail = _stream(command, timeout=2 * HOUR, cwd=MPNN_CHECKOUT)

    written: list[str] = []
    if returncode == 0:
        for design_id in prepared:
            fasta = out_folder / "seqs" / f"{design_id}.fa"
            if not fasta.exists():
                skipped.append({"design_id": design_id, "reason": "no fasta produced"})
                continue
            for record in _mpnn_records(
                fasta.read_text(), design_id, by_design[design_id], mpnn
            ):
                write_shard(record, shards)
                written.append(record["seq_id"])

    data_volume.commit()
    print(f"mpnn: exit {returncode}, {len(written)} sequence(s)", flush=True)
    return {
        "returncode": returncode,
        "n_sequences": len(written),
        "designs": prepared,
        "skipped": skipped,
        "output_tail": output_tail,
    }


def _mpnn_records(text: str, design_id: str, backbone: dict, mpnn) -> list[dict]:
    """One record per designed sequence, parsed out of MPNN's FASTA.

    The format, verified against real output on 2026-10-07: the **first** record
    is the input sequence rather than a design, so designs start at index 1.
    Chains are joined by `--fasta_seq_separation` (`:` by default) and the
    binder is the first segment, since it is chain A. Each design's header
    carries `id`, `T`, `seed`, `overall_confidence`, `ligand_confidence` and
    `seq_rec`.

    The backbone record is carried forward so a sequence row in `summary.csv`
    keeps its design's `bb_*` columns; later stages are stripped in case this is
    a re-run over shards that already have them.
    """
    from rfd3_pipeline.gates import Stage

    entries: list[tuple[str, str]] = []
    header: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            header = line[1:]
        elif header is not None:
            entries.append((header, line))
            header = None

    designs = entries[1:]
    records: list[dict] = []
    for index, (head, sequence) in enumerate(designs, start=1):
        meta: dict[str, str] = {}
        for field in head.split(","):
            key, sep, value = field.partition("=")
            if sep:
                meta[key.strip()] = value.strip()

        binder_sequence = sequence.split(":")[0]
        record = {
            key: value
            for key, value in backbone.items()
            if not key.startswith(("apo_", "holo_", "cv_", "rank_", "cluster_"))
        }
        record.update(
            {
                "seq_id": f"{design_id}_seq{index}",
                "design_id": design_id,
                "mpnn_index": index,
                # Not a link to the backbone -- `design_id` and `seq_id` already
                # carry that -- but an invariant. Every design should yield
                # `num_seqs`, and one that quietly yielded fewer is otherwise
                # visible only by counting rows, which nobody does at scale.
                "mpnn_n_seqs": len(designs),
                "stage_reached": Stage.MPNN,
                "sequence": binder_sequence,
                "seq_length": len(binder_sequence),
                "mpnn_temp": _as_float(meta.get("T")) or mpnn.mpnn_sampling_temp,
                "mpnn_seed": _as_int(meta.get("seed")),
                "mpnn_overall_confidence": _as_float(meta.get("overall_confidence")),
                "mpnn_ligand_confidence": _as_float(meta.get("ligand_confidence")),
                "mpnn_seq_rec": _as_float(meta.get("seq_rec")),
            }
        )
        records.append(record)
    return records


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Backbone filters
# ---------------------------------------------------------------------------
@app.function(image=cpu_image, cpu=1.0, memory=4096, volumes=VOLUMES, timeout=2 * HOUR)
def filter_designs(design_ids: list[str], campaign_payload: dict) -> dict:
    """Filter a chunk of designs, writing one shard each.

    Chunked rather than one design per container: the per-call overhead of
    starting a container dwarfs the few hundred milliseconds of filtering.

    One CPU, not four: the loop below is sequential, and the arrays are a few
    thousand atoms, far too small for BLAS threading to earn a second core.
    4 GiB rather than 16: the old figure predated replacing the dense contact
    mask with neighbour indices, and even that mask was 92 MB at 3000 residues.
    """
    from pathlib import Path

    from rfd3_pipeline.filters import backbone_record
    from rfd3_pipeline.report import write_shard
    from rfd3_pipeline.spec import (
        assert_output_chains,
        load_design_metadata,
        resolve_campaign,
    )
    from rfd3_pipeline.structure import load_structure

    campaign = _campaign_from(campaign_payload)
    root = Path(_run_root(campaign))
    out_dir = root / "rfd3" / "out"
    shards = root / "shards"

    # The designs were written by other containers. A fresh container sees them,
    # but Modal may reuse one across map inputs, and a stale view reads as
    # "missing design" rather than as an error.
    data_volume.reload()

    resolved = resolve_campaign(campaign, structure_root=DATA)
    by_name = {spec.name: spec for spec in resolved.specs}
    binder = campaign.general.chain_binder

    written, failed = [], []
    for design_id in design_ids:
        try:
            metadata = load_design_metadata(out_dir / f"{design_id}.json", binder)
            structure = load_structure(out_dir / f"{design_id}.cif.gz")
            spec = by_name[_spec_of(design_id, by_name)]

            # Checked on every design, not once per spec as its docstring
            # suggests: the test is a set comparison on chain ids of a
            # structure already loaded, so per-design costs nothing and also
            # catches one design that deviates. It guards exactly the failure
            # this pipeline now risks -- "binder first, then targets in contig
            # order, lettered from A" is derived from how RFD3 renumbers rather
            # than guaranteed by it, and every ipSAE and placement column is
            # attributed by chain id. A contig ordered differently would
            # otherwise yield a whole campaign of plausible numbers against the
            # wrong chains.
            _check_hotspot_numbering(metadata, spec, structure, binder)
            record = backbone_record(structure, metadata, spec, campaign)
            record["bb_non_polymer_chains"] = ",".join(
                assert_output_chains(
                    structure, [binder, *campaign.general.chain_targets]
                )
            )
            record["design_id"] = design_id
            record["campaign"] = campaign.general.name_design_campaign
            record["run_name"] = campaign.general.run_name
            write_shard(record, shards)
            written.append(design_id)
        except Exception as error:  # one bad design must not lose the chunk
            failed.append({"design_id": design_id, "error": f"{type(error).__name__}: {error}"})

    data_volume.commit()
    return {"n_written": len(written), "failures": failed}


def _check_hotspot_numbering(metadata, spec, structure, binder: str) -> None:
    """Verify the hotspots the filter scores are the ones this design placed.

    `bb_hotspot_fraction` is scored against `spec.hotspots_by_output_chain()`,
    which is computed once per campaign from the **declared** contig.
    `renumber_hotspots_for_design` instead uses `metadata.sampled_contig` --
    the contig RFD3 actually realised -- and cross-checks the result against
    RFD3's own target-chain entries.

    The two agree whenever every hotspot sits in a chain whose numbering cannot
    shift. On this campaign they do: the contig is
    `100-120,/0,B1-177,/0,C1-242`, so the variable-length segment is the binder
    and the hotspots are on B and C, which follow chain breaks and are numbered
    from 1 independently. They would diverge for a contig with a
    variable-length segment *before* a hotspot in the same chain, and then the
    hotspot gate would be scoring the wrong residues while looking perfectly
    healthy.

    Raised rather than recorded, because a disagreement means the gate's
    verdict is wrong. `filter_designs` catches per design, so this fails one
    design loudly instead of losing the chunk.
    """
    # Its own import. `app.py` keeps imports inside function bodies, and a
    # helper called from such a function does **not** inherit them -- this read
    # `renumber_hotspots_for_design` from `filter_designs`'s namespace and
    # raised `NameError` on all 80 designs of the first full campaign. Local
    # testing missed it because a test script imports at module scope.
    from rfd3_pipeline.spec import renumber_hotspots_for_design

    declared = spec.hotspots_by_output_chain()
    if not declared:
        return

    # `spec.hotspots` is already the resolve-time input->output mapping, so its
    # keys are the input residue keys this wants.
    mapping = renumber_hotspots_for_design(
        list(spec.hotspots), spec.contig, metadata, structure=structure
    )
    realised: dict[str, list[int]] = {}
    for chain, res_id in mapping.values():
        realised.setdefault(chain, []).append(res_id)
    realised = {chain: sorted(ids) for chain, ids in realised.items()}

    if realised != declared:
        raise ValueError(
            f"hotspot numbering from the sampled contig {realised} differs "
            f"from the campaign's {declared}; `bb_hotspot_fraction` would be "
            f"scored against residues this design did not place"
        )


def _spec_of(design_id: str, by_name: dict) -> str:
    """Which spec a design belongs to, from its file name.

    RFD3 names outputs `{prefix}{json_key}_{batch}_{model}`, so the spec name is
    a substring. Matched longest-first so one spec name that is a prefix of
    another cannot win by accident.
    """
    for name in sorted(by_name, key=len, reverse=True):
        if name in design_id:
            return name
    raise KeyError(f"cannot tell which spec {design_id!r} came from")


# ---------------------------------------------------------------------------
# Target MSA cache
# ---------------------------------------------------------------------------
@app.function(image=cpu_image, volumes=VOLUMES, timeout=4 * HOUR)
def build_msa_cache(campaign_payload: dict, force: bool = False) -> dict:
    """Fetch an unpaired alignment per target sequence, once per campaign.

    Its own step, and run once rather than per prediction, because the target
    sequences are fixed for a campaign: they come from the input structure and
    do not change across 640 folds. The alignments are cached by a digest of
    the sequence and the query options, so a re-run is free and a changed option
    cannot be served a stale answer.

    **Only target sequences leave this network.** The binder is never submitted:
    a de novo binder has no homologs to find, and holo on esmfold2 does not
    need one. The targets are the campaign's known input proteins.

    Unpaired only. An unpaired MSA is a property of one sequence and so
    cacheable; a paired MSA is a property of the set of chains queried together
    and is not. ESMFold2 takes one a3m per chain and pairs internally, which is
    why `msa_paired` is False for it and nothing paired is built here.
    """
    data_volume.reload()

    from rfd3_pipeline.config import MODEL_CAPABILITIES
    from rfd3_pipeline.msa import cache_key, query_colabfold, read_cached_a3m, write_cached_a3m
    from rfd3_pipeline.spec import resolve_campaign

    campaign = _campaign_from(campaign_payload)
    capabilities = MODEL_CAPABILITIES[campaign.refolding.model]
    root = Path(_run_root(campaign))
    # Keyed per campaign rather than per run: the targets are the same for
    # every run of a campaign, so a second run reuses the first one's work.
    cache_dir = Path(DATA) / campaign.general.name_design_campaign / "msa_cache"

    if not capabilities.consumes_msa:
        return {
            "skipped": (
                f"{campaign.refolding.model.value} does not read MSAs "
                f"(consumes_msa is False), so there is nothing to build"
            ),
            "cache_dir": str(cache_dir),
        }

    msa = campaign.msa
    resolved = resolve_campaign(campaign, structure_root=DATA)
    sequences = dict(resolved.target_sequences)
    if not sequences:
        return {"skipped": "no target sequences", "cache_dir": str(cache_dir)}

    # Identical target chains collapse to one submission.
    keys = {
        chain: cache_key(sequence, use_env=msa.use_env, use_filter=msa.use_filter)
        for chain, sequence in sequences.items()
    }
    wanted: dict[str, str] = {}
    already: list[str] = []
    for chain, sequence in sequences.items():
        key = keys[chain]
        if not force and read_cached_a3m(cache_dir, key) is not None:
            already.append(chain)
            continue
        wanted[key] = sequence

    fetched: dict[str, int] = {}
    if wanted:
        order = list(wanted)
        print(
            f"msa: querying {len(order)} target sequence(s) at {msa.host_url} "
            f"(env={msa.use_env}, filter={msa.use_filter})",
            flush=True,
        )
        results = query_colabfold(
            [wanted[key] for key in order],
            paired=False,
            use_env=msa.use_env,
            host_url=msa.host_url,
            username=msa.username,
            password=msa.password,
            auth_header=msa.auth_header,
        )
        for key, a3m in zip(order, results):
            if not a3m.strip():
                raise RuntimeError(
                    f"MSA server returned an empty alignment for {key}; "
                    f"folding against it would be worse than failing here"
                )
            write_cached_a3m(cache_dir, key, a3m)
            fetched[key] = a3m.count(">")
    data_volume.commit()

    return {
        "cache_dir": str(cache_dir),
        "chains": {chain: keys[chain] for chain in sequences},
        "already_cached": sorted(already),
        "fetched_depths": fetched,
        "host_url": msa.host_url,
        "authenticated": msa.needs_secret,
        "run_root": str(root),
    }


def _target_alignments(campaign, sequences: dict[str, str], *, stage_into=None) -> dict[str, str]:
    """Each target chain's cached a3m, keyed by chain.

    Shared by holo and cross-validation, which want the same alignments by two
    different conventions -- hence `stage_into` rather than two functions:

    * `stage_into=<dir>` copies each a3m there and returns paths *relative* to
      it. The ESMFold2 kit reads `"msa"` relative to the input JSON
      (`esmfold2/README.md`) and its own example writes `"msas/B.a3m"`.
    * `stage_into=None` returns absolute paths on the volume, which is what the
      AF3 fork wants. It resolves a relative path against the JSON's directory
      (`folding_input._read_file`) -- the trap RFD3's `input` had -- and reads
      gzip/xz/zstd by magic number, so nothing needs copying or decompressing.

    Raises when one is missing. Folding without a target's alignment would
    build it from sequence alone and still produce interface numbers, which is
    the failure worth being loud about.
    """
    from rfd3_pipeline.msa import a3m_cache_path, cache_key, read_cached_a3m

    msa = campaign.msa
    cache_dir = Path(DATA) / campaign.general.name_design_campaign / "msa_cache"
    if stage_into is not None:
        stage_into.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    missing: list[str] = []
    for chain, sequence in sequences.items():
        key = cache_key(sequence, use_env=msa.use_env, use_filter=msa.use_filter)
        if stage_into is None:
            # Probed by stat rather than read: these alignments run to
            # thousands of sequences, and reading one in full only to discard
            # it and let the model read it again costs megabytes per chain per
            # container. An empty file is a truncated write, worse than a miss.
            path = a3m_cache_path(cache_dir, key)
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(f"{chain} ({key})")
                continue
            paths[chain] = str(path)
            continue

        a3m = read_cached_a3m(cache_dir, key)
        if a3m is None:
            missing.append(f"{chain} ({key})")
            continue
        (stage_into / f"{chain}.a3m").write_text(a3m)
        paths[chain] = f"{stage_into.name}/{chain}.a3m"

    if missing:
        raise RuntimeError(
            f"no cached MSA for target chain(s) {', '.join(missing)} under "
            f"{cache_dir}. Run `modal run app.py::cache_msas --campaign <file>` "
            f"first; folding without them would score the binder against "
            f"targets built from sequence alone."
        )
    return paths


# ---------------------------------------------------------------------------
# Refolding (apo and holo)
# ---------------------------------------------------------------------------
# Stage name -> (include_targets, supplies_msa). The two stages differ only in
# what goes into the input JSON, which is why they share one function. The
# variant follows from `supplies_msa` via `folding_variant`, so no field can
# contradict the step's MSA policy.
REFOLD_STAGES = {"apo": (False, False), "holo": (True, True)}

# Column prefixes a stage owns or supersedes, dropped from the shard before its
# own are written. Re-running a stage otherwise inherits its previous attempt's
# columns, and the diagnostic fields are the ones that mislead: an apo re-run
# that computed pLDDT fine still carried the earlier run's
# `apo_plddt_error`, so the row claimed both a value and a failure.
REFOLD_SUPERSEDES = {
    "apo": ("apo_", "holo_", "cv_", "rank_", "cluster_"),
    "holo": ("holo_", "cv_", "rank_", "cluster_"),
}

# One function per folding model, because a Modal function's image is fixed at
# decoration time and `with_options` cannot override it -- only cpu/memory/gpu
# and friends. Keyed by `FoldingModel`'s value as a plain string, the same way
# `MPNN_MODELS` is, so nothing library-typed crosses a function boundary.
# Add a model by adding an image, a three-line wrapper and an entry here; only
# the models a campaign actually uses need defining, since every image declared
# here is built on the next `modal run`.
REFOLD_FUNCTIONS: dict[str, object] = {}


def _refold(seq_ids: list[str], campaign_payload: dict, stage: str) -> dict:
    """Refold a chunk of MPNN sequences, writing `{stage}_*` columns per shard.

    Shared by every per-model wrapper below. The kit is invoked once for the
    whole chunk: `--input` takes a list, and each process pays ~180 s of
    start-up (checkpoint load, CCD dictionary, CUDA graph capture) against
    ~55 s per 100-residue fold, so a process per sequence would be mostly
    start-up.
    """
    import json

    from rfd3_pipeline.config import MODEL_CAPABILITIES, folding_variant
    from rfd3_pipeline.gates import Stage
    from rfd3_pipeline.report import write_shard

    data_volume.reload()

    include_targets, supplies_msa = REFOLD_STAGES[stage]
    campaign = _campaign_from(campaign_payload)
    refolding = campaign.refolding
    binder = campaign.general.chain_binder
    root = Path(_run_root(campaign))
    shards = root / "shards"
    out_root = root / stage
    work = Path("/tmp") / stage
    work.mkdir(parents=True, exist_ok=True)

    msa_paths: dict[str, str] = {}
    variant = folding_variant(refolding.model, supplies_msa)
    if variant is None:
        raise ValueError(
            f"{refolding.model.value} has no --variant for msa_present="
            f"{supplies_msa}; its MODEL_CAPABILITIES row needs filling in"
        )

    by_seq = _sequence_records(shards)

    # Target sequences come from the input structure once per campaign, not from
    # the shards -- they are identical for every prediction.
    target_sequences: dict[str, str] = {}
    ligands: list[dict] = []
    if include_targets:
        from rfd3_pipeline.spec import resolve_campaign
        from rfd3_pipeline.structure import load_structure

        resolved = resolve_campaign(campaign, structure_root=DATA)
        target_sequences = dict(resolved.target_sequences)
        # The ligand belongs in holo: RFD3 designed against it, and the
        # backbone filters already measure `bb_ligand_clashes` and
        # `bb_ligand_min_distance` against it. Folding holo without it would
        # score the binder on a complex it was not designed for.
        ligands = _ligand_entries(campaign, load_structure(resolved.structure_path))

        # Staged only when the model actually reads them, so an esmfold2-fast
        # campaign neither needs a cache nor trips the missing-cache error.
        if MODEL_CAPABILITIES[refolding.model].msa_unpaired:
            msa_paths = _target_alignments(
                campaign, target_sequences, stage_into=work / "msas"
            )

    # Chain order per sequence, kept because the PAE comes back labelled only
    # by position (`asym_id` 0..n) and this is the list those indices refer to.
    # Reconstructing it later from the campaign would be a second source of
    # truth for something already decided here.
    layouts: dict[str, list[dict]] = {}

    items, prepared, skipped = [], [], []
    for seq_id in seq_ids:
        record = by_seq.get(seq_id)
        if record is None:
            skipped.append({"seq_id": seq_id, "reason": "no sequence shard"})
            continue

        # The binder is always suppressed: a de novo binder has no homologs, and
        # leaving `msa` unset would send it to the MSA server. `full_nomsa`
        # would not read one anyway, but holo's `full_msa` would.
        chains = [
            {"type": "protein", "id": binder, "sequence": record["sequence"], "msa": None}
        ]
        if include_targets:
            missing = [c for c in campaign.general.chain_targets if c not in target_sequences]
            if missing:
                skipped.append({"seq_id": seq_id, "reason": f"no target sequence for {missing}"})
                continue
            for chain in campaign.general.chain_targets:
                chains.append(
                    {
                        "type": "protein",
                        "id": chain,
                        "sequence": target_sequences[chain],
                        # A path relative to this JSON, or None when the model
                        # reads no alignments. The binder above stays None
                        # whatever the model: a de novo binder has no homologs,
                        # and this is what keeps its sequence off the network.
                        "msa": msa_paths.get(chain),
                    }
                )

        chains.extend(ligands)

        layouts[seq_id] = [
            {
                "id": chain["id"],
                "type": chain["type"],
                # Residue count for a protein; None for a ligand, whose token
                # count depends on the component and is not worth asserting.
                "n": len(chain["sequence"]) if chain.get("sequence") else None,
            }
            for chain in chains
        ]

        # `id` names the output file, so the result is directly addressable.
        items.append(
            {
                "id": seq_id,
                "sequences": chains,
                "seeds": [campaign.general.base_seed],
                "num_diffusion_samples": 1,
            }
        )
        prepared.append(seq_id)

    if not prepared:
        return {"returncode": 0, "stage": stage, "n_written": 0, "skipped": skipped}

    input_json = work / f"{stage}_input.json"
    input_json.write_text(json.dumps(items, indent=2))
    out_dir = work / "out"

    command = [
        "bash", "run.sh", "pred",
        "--config", GPU_TO_KIT_CARD[refolding.gpu],
        "--mode", refolding.mode.value,
        # Required, no default, one per process.
        "--variant", variant,
        "--input", str(input_json),
        "--out_dir", str(out_dir),
        # One seed, deliberately, and not an omission to be "fixed" with a
        # `refolding.num_seeds`: refolding is a *filter*, and a gate wants one
        # verdict per sequence rather than a distribution to average. Seeds are
        # spent where they buy something instead -- cross-validation, which
        # exists to be compared against and reports its own spread. esmfold2's
        # capability row is `seed_style: LIST`, so several would be accepted;
        # the decision is ours, not the model's.
        "--seeds", str(campaign.general.base_seed),
        "--det", "1",
    ]
    print(
        f"{stage}: {len(prepared)} sequence(s), variant {variant}"
        + (f", ligand chains {[entry['id'] for entry in ligands]}" if ligands else ""),
        flush=True,
    )
    returncode, output_tail = _stream(command, timeout=4 * HOUR, cwd=ESM_KIT)

    # Exit 3 is the kit refusing to engage a mode, by name, rather than falling
    # back to stock. It is a configuration error, not a failed fold.
    if returncode == 3:
        raise RuntimeError(
            f"{stage}: kit mode '{refolding.mode.value}' could not engage "
            f"(exit 3). Its reason is on the NOT ACTIVE line:\n{output_tail}"
        )

    written = []
    if returncode == 0:
        rows = _pred_rows(out_dir / "pred_rows.jsonl")
        out_root.mkdir(parents=True, exist_ok=True)
        superseded = REFOLD_SUPERSEDES[stage]
        for seq_id in prepared:
            record = {
                key: value
                for key, value in by_seq[seq_id].items()
                if not key.startswith(superseded)
            }
            metrics = _refold_metrics(
                seq_id, stage, rows.get(seq_id), out_dir, root, binder, campaign,
                layout=layouts.get(seq_id),
            )
            record.update(metrics)
            record["stage_reached"] = getattr(Stage, stage.upper())
            write_shard(record, shards)
            written.append(seq_id)

            # Keep the structure: holo needs the apo one for the apo/holo RMSD.
            produced = sorted((out_dir / "cif_all").glob(f"{seq_id}__*"))
            for path in produced:
                (out_root / seq_id).mkdir(parents=True, exist_ok=True)
                (out_root / seq_id / path.name).write_bytes(path.read_bytes())

    data_volume.commit()
    print(f"{stage}: exit {returncode}, {len(written)} record(s)", flush=True)
    return {
        "returncode": returncode,
        "stage": stage,
        "variant": variant,
        "n_written": len(written),
        "skipped": skipped,
        "output_tail": output_tail,
    }


def _ligand_components(campaign, structure) -> list[tuple[str, int, str]]:
    """The configured ligand residues, resolved against the input structure.

    `rfd3.ligand` is a reference *into* that structure -- `"E703"` means chain
    E, residue 703 -- so both the chain id and the CCD code come from there
    rather than from code. Hardcoding either would be wrong in a way that is
    hard to see: this campaign's input holds MN in *two* chains, one in D at
    1004 and three in E at 703-705, and names only E703. Lettering the ligand
    "after the protein chains" would land on D, a different ion at an unrelated
    residue number.

    The chain id must also match what RFD3 emitted, since the structural
    comparisons pair on (chain, res_id). RFD3 preserves the input letter --
    observed 2026-10-07, output chains `A/B/C/E` -- which is what this relies
    on.

    Returns (chain, res_id, ccd_code) per configured residue. The two callers
    differ in what they *allow*, not in how they look it up, so the limits stay
    with them: ESMFold2 represents one ligand component and the AF3 fork has no
    such cap.
    """
    from rfd3_pipeline.contig import parse_residue_selection

    if not campaign.rfdiffusion3.ligand:
        return []

    components = []
    for chain, res_id in parse_residue_selection(campaign.rfdiffusion3.ligand):
        selected = structure[
            (structure.chain_id == chain) & (structure.res_id == res_id)
        ]
        names = sorted({str(name) for name in selected.res_name})
        if not names:
            raise ValueError(
                f"ligand {chain}{res_id} is not in the input structure; "
                f"rfd3.ligand names a residue that does not exist"
            )
        if len(names) > 1:
            raise ValueError(
                f"ligand {chain}{res_id} resolves to several residue names "
                f"{names}; one position cannot be more than one component"
            )
        components.append((chain, res_id, names[0]))
    return components


def _ligand_entries(campaign, structure) -> list[dict]:
    """ESMFold2's ligand chains. Shape from `opt/esmfold2_opt/inputs.py:7-9`:
    `{"type": "ligand", "id": ..., "ccd": [...]}`, with no `sequence`.

    **One ion only.** The kit takes a single ligand component here, not a chain
    of them, so more than one configured residue raises rather than folding
    against a subset. Note this is a *refolding* limit, not a config error:
    RFD3 can legitimately be given the whole chain-E triad, and such a campaign
    is valid right up to holo. That is why the check lives here rather than in
    the schema -- but it does mean the failure arrives after RFD3 and MPNN have
    already run. See `_af3_ligand_entries`, which has no such cap.
    """
    components = _ligand_components(campaign, structure)
    if len(components) > 1:
        named = ",".join(f"{chain}{res_id}" for chain, res_id, _ in components)
        raise ValueError(
            f"rfd3.ligand names {len(components)} residues ({named}), and "
            f"refolding can represent one ligand component. Folding against a "
            f"subset would score the binder on a complex it was not designed "
            f"for, so name a single residue or split the stage."
        )
    return [
        {"type": "ligand", "id": chain, "ccd": [ccd]}
        for chain, _, ccd in components
    ]


def _pred_rows(path) -> dict:
    """`pred_rows.jsonl` keyed by complex id.

    The kit appends to this file, so a re-run in the same out_dir leaves older
    rows behind; the last row for an id wins.
    """
    import json

    rows: dict[str, dict] = {}
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("complex_id"):
            rows[row["complex_id"]] = row
    return rows


def _holo_ipsae(cif, layout: list[dict], binder: str, campaign) -> dict:
    """ipSAE for a holo prediction, from the PAE written beside the CIF.

    The kit's `*_pae.npz` carries `asym_id`, one 0-based chain index per token,
    in the order the input JSON listed the chains -- which is what `layout` is.
    The indices are therefore resolved against that list rather than against
    the campaign, so a reordering would have to happen in one place to go
    unnoticed, and the protein block lengths are checked against the sequences
    that were submitted to make sure it did not.

    Ligands are excluded. A metal ion is a single token, and a one-residue
    chain yields a defined but meaningless ipSAE -- measured on this campaign,
    including MN pulled `ipsae_min` from 0.598 to 0.199, because the binder's
    worst "interface" became the one with the ion.
    """
    import numpy as np

    from rfd3_pipeline.metrics.ipsae import cross_chain_pae_floor, ipsae_from_tokens

    pae_path = cif.with_name(f"{cif.stem}_pae.npz")
    if not pae_path.is_file():
        return {"holo_ipsae_error": f"no PAE beside {cif.name}"}

    try:
        with np.load(pae_path) as data:
            # float16 on disk; the score divides by d0 and squares, so it is
            # promoted before any arithmetic happens.
            pae = data["pae"].astype(np.float32)
            asym = np.asarray(data["asym_id"])

        if int(asym.max()) + 1 != len(layout):
            raise ValueError(
                f"PAE holds {int(asym.max()) + 1} chains but the input listed "
                f"{len(layout)} ({[c['id'] for c in layout]})"
            )

        # The check that would catch a reordering: every protein block must be
        # as long as the sequence submitted for it.
        for index, chain in enumerate(layout):
            if chain["n"] is None:
                continue
            found = int(np.count_nonzero(asym == index))
            if found != chain["n"]:
                raise ValueError(
                    f"chain {chain['id']} was submitted with {chain['n']} "
                    f"residues but its PAE block holds {found}; the asym_id "
                    "order does not match the input order"
                )

        labels = [layout[index]["id"] for index in asym]
        scores = ipsae_from_tokens(
            pae,
            labels,
            binder_chain=binder,
            pae_cutoff=campaign.ipsae.pae_cutoff,
            keep=[chain["id"] for chain in layout if chain["type"] == "protein"],
            # Binder-centric only. Target-to-target pairs do not involve the
            # designed chain and grow as (N+1)N, doubled across the two stages.
            pairs="binder",
        )
        # Recorded beside the score so a zero can be read: ipSAE is 0 when no
        # residue pair falls under the cutoff, which is a real result and not
        # distinguishable from a mis-slice without this.
        scores.update(
            cross_chain_pae_floor(
                pae, labels, binder,
                [c["id"] for c in layout
                 if c["type"] == "protein" and c["id"] != binder],
            )
        )
    except Exception as error:
        return {"holo_ipsae_error": f"{type(error).__name__}: {error}"}
    return {f"holo_{name}": value for name, value in scores.items()}


def _target_aligned_rmsds(
    reference, predicted, design_json, binder: str, targets, prefix: str
) -> dict:
    """Placement RMSDs: fit on the targets, measure the binder.

    The distinction that matters, and the one the other RMSD columns do not
    make. `{prefix}rmsd_to_design` superimposes the binder on the binder, so
    translation and rotation are fitted away and it reports how well the
    *fold* was reproduced. Aligning on the targets instead leaves the binder's
    displacement in the measurement, which is what says whether it docked
    where it was designed to -- `metrics/rmsd.py` puts it plainly: "what makes
    binder displacement visible instead of being absorbed into the fit".

    Two measurements, because they answer different questions:

    * `rmsd_binder_on_target` over the whole binder chain.
    * `rmsd_motif_on_target` over the scaffolded motif residues only, which for
      a motif-scaffolding campaign is the design objective rather than a
      diagnostic -- the rest of the binder is scaffold and is allowed to move.

    The motif residues come from the design's own metadata
    (`binder_entries`, RFD3's input-to-output map), so they are the residues
    that design actually placed rather than the ones the contig asked for.
    Returns the columns it could compute; a missing metadata file costs the
    motif column alone.
    """
    from rfd3_pipeline.metrics.rmsd import rmsd_binder_on_target, rmsd_motif_on_target

    metrics: dict[str, object] = {}
    if not targets:
        return metrics

    try:
        metrics[f"{prefix}rmsd_binder_on_target"] = rmsd_binder_on_target(
            reference, predicted, binder, list(targets)
        )
    except Exception as error:
        metrics[f"{prefix}rmsd_binder_on_target"] = None
        metrics[f"{prefix}rmsd_on_target_error"] = f"{type(error).__name__}: {error}"

    if design_json is None or not design_json.is_file():
        return metrics
    try:
        from rfd3_pipeline.spec import load_design_metadata

        metadata = load_design_metadata(design_json, binder)
        motif = sorted({res_id for _, res_id in metadata.binder_entries.values()})
        if motif:
            metrics[f"{prefix}rmsd_motif_on_target"] = rmsd_motif_on_target(
                reference, predicted, binder, list(targets), motif
            )
            metrics[f"{prefix}n_motif_residues"] = len(motif)
    except Exception as error:
        metrics[f"{prefix}rmsd_motif_on_target"] = None
        metrics[f"{prefix}rmsd_motif_error"] = f"{type(error).__name__}: {error}"
    return metrics


def _refold_metrics(
    seq_id, stage, row, out_dir, root, binder, campaign, layout=None
) -> dict:
    """The `{stage}_*` columns for one sequence.

    `pred_rows.jsonl` supplies what the kit measured; the rest is computed from
    the predicted structure. Every field is optional on purpose -- a missing
    metric should leave a null in the summary rather than lose the chunk.
    """
    prefix = f"{stage}_"
    metrics: dict[str, object] = {
        f"{prefix}status": (row or {}).get("status"),
        f"{prefix}ptm": _as_float((row or {}).get("ptm")),
        f"{prefix}n_tokens": _as_int((row or {}).get("n_tokens")),
        f"{prefix}wall_s": _as_float((row or {}).get("wall_s")),
        # Positive evidence about the MSA rather than trust in the flag: 0 means
        # the fold genuinely saw no alignment.
        f"{prefix}msa_depth": max((row or {}).get("msa_depths") or [0]),
    }

    # ipTM scores an *interface*. Apo folds the binder alone, so the kit reports
    # 0.000 for want of a second chain -- a number that reads as a catastrophic
    # score in any table rather than as "not applicable". Only holo gets it.
    if stage != "apo":
        metrics[f"{prefix}iptm"] = _as_float((row or {}).get("iptm"))

    cifs = sorted((out_dir / "cif_all").glob(f"{seq_id}__*.cif"))
    if not cifs:
        metrics[f"{prefix}error"] = "no cif produced"
        return metrics

    from rfd3_pipeline.structure import ca_only, chain_subset, load_structure

    # `b_factor` is where the kit writes pLDDT, and biotite does not read it
    # unless asked -- without this the attribute is absent entirely.
    predicted = load_structure(cifs[0], extra_fields=["b_factor"])

    # Mean pLDDT over the binder's alpha carbons. ESMFold2 writes it into the
    # B-factor column; guarded because a format change should not lose the fold.
    try:
        binder_ca = ca_only(chain_subset(predicted, [binder]))
        metrics[f"{prefix}plddt_binder"] = float(binder_ca.b_factor.mean())
    except Exception as error:
        metrics[f"{prefix}plddt_binder"] = None
        metrics[f"{prefix}plddt_error"] = f"{type(error).__name__}: {error}"

    # Did the sequence fold into the backbone it was designed for? This is the
    # question apo exists to answer, so it is computed against the RFD3 design
    # rather than taken from the kit.
    design_id = (seq_id.rsplit("_seq", 1)[0]) if "_seq" in seq_id else None
    design_path = root / "rfd3" / "out" / f"{design_id}.cif.gz" if design_id else None
    if design_path is not None and design_path.is_file():
        try:
            from rfd3_pipeline.metrics.rmsd import rmsd_binder_fold

            metrics[f"{prefix}rmsd_to_design"] = rmsd_binder_fold(
                load_structure(design_path), predicted, binder
            )
        except Exception as error:
            metrics[f"{prefix}rmsd_to_design"] = None
            metrics[f"{prefix}rmsd_error"] = f"{type(error).__name__}: {error}"

    if stage == "holo" and layout is not None:
        metrics.update(_holo_ipsae(cifs[0], layout, binder, campaign))

    if stage == "holo":
        # Did binding change the fold? A binder whose apo and holo backbones
        # disagree was not folding into one shape, so its interface numbers
        # describe a conformation the apo gate never vetted. Apo's structure is
        # kept on the volume by the caller for exactly this comparison.
        apo_cifs = sorted((root / "apo" / seq_id).glob("*.cif")) if seq_id else []
        if apo_cifs:
            try:
                # Diagnostic since 2026-10-09: recorded because a binder
                # that changes shape on binding is worth seeing, but no longer
                # gated -- "does it fold without the target" is the apo gate's
                # question, asked earlier and more cheaply.
                from rfd3_pipeline.metrics.rmsd import rmsd_binder_fold

                apo = load_structure(apo_cifs[0])
                metrics["holo_rmsd_apo_holo"] = rmsd_binder_fold(
                    predicted, apo, binder
                )
            except Exception as error:
                metrics["holo_rmsd_apo_holo"] = None
                metrics["holo_rmsd_apo_holo_error"] = f"{type(error).__name__}: {error}"
        else:
            metrics["holo_rmsd_apo_holo"] = None

    # The verdict is domain logic, not orchestration: it lives beside the gate
    # declarations it applies, where it can be tested without Modal.
    from rfd3_pipeline.gates import refold_verdict

    # Placement, for the stage that has targets to align on. Apo folds the
    # binder alone, so there is nothing to fit it against.
    if stage == "holo" and design_path is not None and design_path.is_file():
        metrics.update(
            _target_aligned_rmsds(
                load_structure(design_path),
                predicted,
                design_path.with_suffix("").with_suffix(".json"),
                binder,
                campaign.general.chain_targets,
                prefix,
            )
        )

    metrics.update(refold_verdict(metrics, stage, campaign.refolding))
    return metrics






@app.function(image=esmfold2_image, gpu="A100-40GB", volumes=VOLUMES, timeout=4 * HOUR)
def refold_esmfold2(seq_ids: list[str], campaign_payload: dict, stage: str) -> dict:
    """ESMFold2's wrapper over `_refold`. One per model, because the image is
    fixed at decoration time."""
    return _refold(seq_ids, campaign_payload, stage)


@app.function(image=esmfold2_image, gpu="A100-40GB", volumes=VOLUMES, timeout=HOUR)
def install_esm_jit() -> dict:
    """Precompile the kit's Triton kernels into the jit volume, once.

    Needs a GPU to compile for, so it is a GPU function despite doing no
    folding. Without it each of ~640 predictions pays a cold compile.
    """
    Path(ESM_JIT).mkdir(parents=True, exist_ok=True)
    returncode, output_tail = _stream(
        ["bash", "run.sh", "warm", "--config", "a100",
         "--variant", "full_nomsa", "--mode", "fast"],
        timeout=HOUR,
        cwd=ESM_KIT,
    )
    jit_volume.commit()
    cached = [p for p in Path(ESM_JIT).rglob("*") if p.is_file()]
    return {
        "returncode": returncode,
        "jit_files": len(cached),
        "jit_mb": round(sum(p.stat().st_size for p in cached) / 1e6, 1),
        "output_tail": output_tail,
    }


REFOLD_FUNCTIONS["esmfold2"] = refold_esmfold2


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
# One function per model, for the same reason as `REFOLD_FUNCTIONS`: a Modal
# function's image is fixed at decoration time. Cross-validation is guaranteed
# to need a different image from refolding, because `CampaignConfig` rejects a
# campaign whose two stages name the same model.
CV_FUNCTIONS: dict[str, object] = {}

# What cross-validation owns, dropped from a shard before its own are written.
# Narrower than the refolding stages' lists because nothing runs after it, so
# there is nothing downstream to supersede.
CV_SUPERSEDES = ("cv_", "rank_", "cluster_")


def _af3_ligand_entries(campaign, structure) -> list[dict]:
    """The fork's ligand chains: `{"ligand": {"id": ..., "ccdCodes": [...]}}`.

    AF3 wraps entities by type and names components by CCD code
    (`docs/input.md:93-96`), and unlike ESMFold2 it accepts several -- so no
    cap here, and a campaign folding the whole chain-E triad reaches this stage
    intact.
    """
    return [
        {"ligand": {"id": chain, "ccdCodes": [ccd]}}
        for chain, _, ccd in _ligand_components(campaign, structure)
    ]


def _cv_fold_inputs(seq_ids, by_seq, campaign, resolved, alignments, ligands, in_dir):
    """Write one AF3 fold-input JSON per sequence.

    Returns (prepared, skipped, seeds). The fork's schema, which is not
    ESMFold2's: entities wrapped by type, MSAs in camelCase, and the seeds
    carried in the input rather than on the command line.
    """
    import json

    from rfd3_pipeline.config import seed_sequence

    binder = campaign.general.chain_binder
    # Explicit, never `--num_seeds`. The fork uses the input's seeds when the
    # flag is absent, and when the flag is set it just builds
    # `range(base, base + n)` -- the same list `seed_sequence` computes. Passing
    # them keeps cross-validation's seeds ours, which is the point of a stage
    # whose whole purpose is to be compared against.
    seeds = seed_sequence(
        campaign.general.base_seed, campaign.cross_validation.num_seeds
    )

    prepared, skipped = [], []
    for seq_id in seq_ids:
        record = by_seq.get(seq_id)
        if record is None:
            skipped.append({"seq_id": seq_id, "reason": "no sequence shard"})
            continue

        entities: list[dict] = [
            {
                "protein": {
                    "id": binder,
                    "sequence": record["sequence"],
                    # A single-sequence stub, not "" and not absent. Absent is
                    # the one state `fill_missing_msas` reads as "fetch this",
                    # and a de novo binder has no homologs to find anyway. This
                    # is what keeps the binder's sequence off the network.
                    "unpairedMsa": f">query\n{record['sequence']}\n",
                    "pairedMsa": "",
                    "templates": [],
                }
            }
        ]
        for chain in campaign.general.chain_targets:
            entities.append(
                {
                    "protein": {
                        "id": chain,
                        "sequence": resolved.target_sequences[chain],
                        "unpairedMsaPath": alignments[chain],
                        # Unpaired only: the cache does not hold paired
                        # alignments. "" counts as *set*, so this suppresses
                        # pairing rather than triggering a search for it.
                        "pairedMsa": "",
                        "templates": [],
                    }
                }
            )
        entities.extend(ligands)

        payload = {
            "dialect": "alphafold3",
            # 1 through 4 are all accepted (`folding_input.JSON_VERSIONS`).
            "version": 1,
            # Names the output directory, so results stay addressable by seq_id.
            "name": seq_id,
            "modelSeeds": seeds,
            "sequences": entities,
        }
        (in_dir / f"{seq_id}.json").write_text(json.dumps(payload, indent=2))
        prepared.append(seq_id)

    # Two sequences whose names strip to one directory would overwrite each
    # other's results silently, so this is checked rather than trusted. It
    # should be unreachable while `SpecConfig.name` holds.
    by_job: dict[str, list[str]] = {}
    for seq_id in prepared:
        by_job.setdefault(_af3_job_name(seq_id), []).append(seq_id)
    collisions = {job: ids for job, ids in by_job.items() if len(ids) > 1}
    if collisions:
        raise ValueError(
            f"sequence ids collide once the fork sanitises them: {collisions}. "
            f"Their results would overwrite each other in one output directory."
        )
    return prepared, skipped, seeds


def _cv_metrics(seq_id, job_dir, root, binder, seeds, campaign=None) -> dict:
    """The `cv_*` columns for one sequence.

    Reported across all seeds rather than from the best one alone. A single
    best score cannot distinguish a model that is confident from one that is
    confident once in five, and cross-validation exists to answer exactly that.
    Every field is optional on purpose: a missing metric should leave a null in
    the summary rather than lose the chunk.
    """
    import json

    metrics: dict[str, object] = {"cv_n_seeds_requested": len(seeds)}
    if not job_dir.is_dir():
        metrics["cv_error"] = "no output directory"
        return metrics

    per_seed = []
    for seed in seeds:
        for summary in sorted(
            job_dir.glob(f"seed-{seed}_sample-*/*_summary_confidences.json")
        ):
            try:
                per_seed.append((seed, json.loads(summary.read_text()), summary.parent))
            except Exception:
                continue
    metrics["cv_n_seeds_ok"] = len(per_seed)
    if not per_seed:
        metrics["cv_error"] = "no summary_confidences.json produced"
        return metrics

    def spread(field: str) -> None:
        # `_as_float` rather than `float()`, matching mpnn and refolding: a
        # field that arrives as a string or a null costs one column here
        # instead of raising and losing the sequence's whole record.
        values = [
            value
            for value in (_as_float(payload.get(field)) for _, payload, _ in per_seed)
            if value is not None
        ]
        if not values:
            return
        metrics[f"cv_{field}_mean"] = sum(values) / len(values)
        metrics[f"cv_{field}_best"] = max(values)
        metrics[f"cv_{field}_worst"] = min(values)

    for field in ("iptm", "ptm", "ranking_score"):
        spread(field)
    metrics["cv_has_clash_any"] = any(
        bool(payload.get("has_clash")) for _, payload, _ in per_seed
    )

    # The best-ranked seed carries the structure worth comparing.
    best_seed, _, best_dir = max(
        per_seed, key=lambda item: item[1].get("ranking_score", float("-inf"))
    )
    metrics["cv_best_seed"] = best_seed

    cifs = sorted(best_dir.glob("*_model.cif"))
    if not cifs:
        metrics["cv_structure_error"] = "no model.cif for the best seed"
        return metrics
    metrics.update(_cv_structure_metrics(cifs[0], seq_id, root, binder, campaign))
    return metrics


def _cv_ipsae(cif, binder: str, campaign) -> dict:
    """ipSAE for one cross-validation sample, from the fork's confidences.

    Simpler than holo's: the fork writes `token_chain_ids` -- a chain id per
    token, by name -- in the per-sample `*_confidences.json`, so there is no
    index-to-chain mapping to get wrong and nothing to validate against the
    submitted lengths.

    Read from the container's scratch output rather than from the volume,
    because `_af3_keep` deliberately does not retain this file: it holds the
    full PAE and contact-probability matrices, which are large and are needed
    only to compute these numbers once.

    The same `ipsae.pae_cutoff` as holo, which is the point -- a
    cross-validation score computed at a different cutoff would not be
    comparable with the holo one it exists to check.
    """
    import json

    import numpy as np

    from rfd3_pipeline.metrics.ipsae import cross_chain_pae_floor, ipsae_from_tokens

    confidences = [
        path
        for path in sorted(cif.parent.glob("*_confidences.json"))
        if "summary" not in path.name
    ]
    if not confidences:
        return {"cv_ipsae_error": f"no confidences.json in {cif.parent.name}"}

    try:
        payload = json.loads(confidences[0].read_text())
        pae = np.asarray(payload["pae"], dtype=np.float32)
        labels = [str(label) for label in payload["token_chain_ids"]]
        proteins = [binder, *campaign.general.chain_targets]
        scores = ipsae_from_tokens(
            pae,
            labels,
            binder_chain=binder,
            pae_cutoff=campaign.ipsae.pae_cutoff,
            # Ligands excluded, as in holo. Named from the campaign here rather
            # than from a layout, because the fork labels its tokens by chain
            # id and the protein chains are exactly these.
            keep=proteins,
            # As holo: binder-centric, so the two stages' column sets match and
            # neither grows quadratically in the number of targets.
            pairs="binder",
        )
        scores.update(
            cross_chain_pae_floor(
                pae, labels, binder, list(campaign.general.chain_targets)
            )
        )
    except Exception as error:
        return {"cv_ipsae_error": f"{type(error).__name__}: {error}"}
    return {f"cv_{name}": value for name, value in scores.items()}


def _cv_structure_metrics(
    cif, seq_id: str, root, binder: str, campaign=None
) -> dict:
    """pLDDT and the comparison against holo, from one predicted structure.

    Model-independent: any cross-validation model produces a CIF and wants
    these two numbers from it, so this is the half a second model reuses rather
    than copies. Only locating the CIF is model-specific, and that stays with
    the caller.

    Both measurements are guarded individually. A format change should cost one
    column, not the whole record -- and the two failures are recorded as
    `cv_plddt_error` / `cv_rmsd_error` rather than left as bare nulls, because
    a null that means "biotite raised" and a null that means "no holo
    structure" need telling apart.
    """
    from rfd3_pipeline.structure import ca_only, chain_subset, load_structure

    metrics: dict[str, object] = {}
    if campaign is not None:
        metrics.update(_cv_ipsae(cif, binder, campaign))
    # pLDDT lives in the B-factor column, which biotite does not read unless
    # asked -- without `extra_fields` the attribute is absent entirely.
    predicted = load_structure(cif, extra_fields=["b_factor"])
    try:
        binder_ca = ca_only(chain_subset(predicted, [binder]))
        metrics["cv_plddt_binder"] = float(binder_ca.b_factor.mean())
    except Exception as error:
        metrics["cv_plddt_binder"] = None
        metrics["cv_plddt_error"] = f"{type(error).__name__}: {error}"

    # Did the independent model reproduce the *designed* backbone? This is the
    # stronger of the two comparisons: agreeing with holo could mean both
    # models share a bias, whereas the RFD3 design is the thing actually asked
    # for. Completes the triple with `apo_rmsd_to_design` and
    # `holo_rmsd_to_design`, which `_refold_metrics` already writes.
    design_id = seq_id.rsplit("_seq", 1)[0] if "_seq" in seq_id else None
    design = root / "rfd3" / "out" / f"{design_id}.cif.gz" if design_id else None
    if design is not None and design.is_file():
        try:
            from rfd3_pipeline.metrics.rmsd import rmsd_binder_fold

            designed = load_structure(design)
            metrics["cv_rmsd_to_design"] = rmsd_binder_fold(designed, predicted, binder)
            if campaign is not None:
                metrics.update(
                    _target_aligned_rmsds(
                        designed,
                        predicted,
                        design.with_suffix("").with_suffix(".json"),
                        binder,
                        campaign.general.chain_targets,
                        "cv_",
                    )
                )
        except Exception as error:
            metrics["cv_rmsd_to_design"] = None
            metrics["cv_rmsd_to_design_error"] = f"{type(error).__name__}: {error}"

    # Do the two models agree with each other? Weaker than the above, but it is
    # what says whether holo's interface was one model's opinion.
    holo_cifs = sorted((root / "holo" / seq_id).glob("*.cif"))
    if not holo_cifs:
        metrics["cv_rmsd_to_holo"] = None
        return metrics
    try:
        from rfd3_pipeline.metrics.rmsd import rmsd_binder_fold

        holo = load_structure(holo_cifs[0])
        metrics["cv_rmsd_to_holo"] = rmsd_binder_fold(holo, predicted, binder)
        # Whether the two models dock the binder in the same place, as opposed
        # to folding it the same way. The holo structure carries the targets,
        # so this is the one comparison that can distinguish them.
        if campaign is not None:
            from rfd3_pipeline.metrics.rmsd import rmsd_binder_on_target

            metrics["cv_rmsd_placement_vs_holo"] = rmsd_binder_on_target(
                holo, predicted, binder, list(campaign.general.chain_targets)
            )
    except Exception as error:
        metrics["cv_rmsd_to_holo"] = None
        metrics["cv_rmsd_error"] = f"{type(error).__name__}: {error}"
    return metrics


def _cv_write_records(
    prepared: list[str],
    by_seq: dict,
    *,
    shards,
    root,
    out_dir,
    out_root,
    binder: str,
    seeds: list[int],
    metrics,
    keep,
    job_dir_of,
    campaign=None,
) -> list[str]:
    """Write one shard per cross-validated sequence and keep its structure.

    Model-independent, and the part a second cross-validation model gets for
    free: superseded columns are dropped, `stage_reached` is set, and the
    structures named by `keep` are copied off the container's scratch disk onto
    the volume.

    Three callables, because each model owns its own output layout and none of
    it need match AF3's `seed-N_sample-M/`:

    * `job_dir_of(out_dir, seq_id)` -- where that sequence's results landed.
      Not `out_dir / seq_id`: a model is free to rename the job, and the fork
      does (see `_af3_job_name`).
    * `metrics(seq_id, job_dir, root, binder, seeds) -> dict` -- read them.
    * `keep(record) -> list[str]` -- globs worth retaining, relative to
      `job_dir`.
    """
    from rfd3_pipeline.gates import Stage
    from rfd3_pipeline.report import write_shard

    written = []
    out_root.mkdir(parents=True, exist_ok=True)
    for seq_id in prepared:
        job_dir = job_dir_of(out_dir, seq_id)
        record = {
            key: value
            for key, value in by_seq[seq_id].items()
            if not key.startswith(CV_SUPERSEDES)
        }
        record.update(metrics(seq_id, job_dir, root, binder, seeds, campaign))
        record["stage_reached"] = Stage.CV
        write_shard(record, shards)
        written.append(seq_id)

        for pattern in keep(record):
            for path in job_dir.glob(pattern):
                (out_root / seq_id).mkdir(parents=True, exist_ok=True)
                (out_root / seq_id / path.name).write_bytes(path.read_bytes())
    return written


def _af3_job_name(seq_id: str) -> str:
    """Where the fork actually writes this job's output.

    `run_alphafold.py` names the directory after `sanitised_name()`, not after
    the name it was given (`folding_input.py:1031-1035`): spaces become
    underscores and anything outside `[A-Za-z0-9_-.]` is dropped. Those
    characters are rejected by `SpecConfig.name` so this should be the identity
    function -- it exists because the consequence of being wrong is silent.
    A mismatch produces `cv_error: no output directory` for every sequence
    while the run exits 0, which looks like a model failure rather than a
    naming one.
    """
    import string

    allowed = set(string.ascii_letters + string.digits + "_-.")
    return "".join(c for c in seq_id.replace(" ", "_") if c in allowed)


def _af3_keep(record: dict, *, full_confidences: bool = False) -> list[str]:
    """The best seed's structure, and nothing else by default.

    Five CIFs per sequence across a campaign is a lot of volume for a
    comparison already reduced to `cv_rmsd_to_holo`.

    `full_confidences` also keeps the per-sample `*_confidences.json`, which
    holds the whole token-by-token PAE and contact-probability matrices. That
    is megabytes per sequence, so it is off at campaign scale -- but it is the
    only way to interrogate a score after the fact, and a dry run exists to be
    interrogated. `run.dry_run` turns it on for exactly that reason: the first
    cross-validation ipSAE came back 0.0 and there was no way to tell a real
    "no confident contact" from a mis-sliced matrix without the matrix.

    Note the glob also matches `*_summary_confidences.json`, which is already
    kept above; the duplicate copy is harmless and cheaper than a tighter
    pattern.
    """
    best = record.get("cv_best_seed")
    if best is None:
        return []
    patterns = [
        f"seed-{best}_sample-*/*_model.cif",
        f"seed-{best}_sample-*/*_summary_confidences.json",
        "*_ranking_scores.csv",
    ]
    if full_confidences:
        patterns.append(f"seed-{best}_sample-*/*_confidences.json")
    return patterns


def _cross_validate_af3(seq_ids: list[str], campaign_payload: dict) -> dict:
    """Refold holo's survivors with the AF3 fork, on seeds we chose.

    **AF3-specific, despite being the only cross-validation implementation.**
    Named for the model rather than for the stage so that is not mistaken: the
    fold-input schema, the `run_alphafold.py` command line and the output
    layout all live here or in the helpers above.

    To add another cross-validation model -- as an *alternative* to this one,
    since `cross_validation.model` names exactly one -- you need all of:

    * an image, weight paths and a `populate_*_weights` function, next to
      `af3_fork_image`;
    * four model-specific pieces: an input writer (`_cv_fold_inputs`), a
      ligand builder (`_af3_ligand_entries`), a metrics reader (`_cv_metrics`)
      and a retention policy (`_af3_keep`);
    * this function, an `@app.function` wrapper, and a `CV_FUNCTIONS` entry
      keyed by the model's `FoldingModel` value;
    * a `MODEL_CAPABILITIES` row, including `uses_kit` -- which decides whether
      `--config`, `--mode` and `n_gpu_shard` mean anything for it.

    What you get for free is the model-independent half: `_target_alignments`
    for the cached MSAs, `_ligand_components` for the residue lookup,
    `_cv_write_records` for shards and structure retention, `_cv_structure_metrics`
    for pLDDT and the holo comparison, and the sequence selection in
    `orchestrate`.

    Two things to know before copying this function. `_cv_write_records` reads
    `cv_best_seed` off the record, which is an AF3 notion -- a model without
    seeds should write its own key and its own `keep`. And `CV_SUPERSEDES` is
    keyed on the *stage*, not the model, so two models cannot write `cv_*`
    columns side by side; swapping models supersedes the previous one's
    results, which is the intent.
    """
    import shutil
    import tempfile

    from rfd3_pipeline.config import Weights
    from rfd3_pipeline.gpu import attention_tuning
    from rfd3_pipeline.spec import resolve_campaign
    from rfd3_pipeline.structure import load_structure

    # The volume is read by a long-lived container, and `commit()` makes a
    # write durable without pushing it into a reader already running.
    data_volume.reload()
    weights_volume.reload()

    campaign = _campaign_from(campaign_payload)
    cv = campaign.cross_validation
    root = Path(_run_root(campaign))
    shards = root / "shards"
    # A fresh directory per call, not a fixed `/tmp/cv`. `--input_dir` folds
    # every JSON it finds, and Modal reuses one container across `.map()`
    # inputs -- so a directory that survives between chunks means chunk k
    # re-folds chunks 1..k. That is quadratic GPU spend, and silent, because
    # every extra prediction succeeds and simply overwrites its own output.
    work = Path(tempfile.mkdtemp(prefix="cv-", dir="/tmp"))
    in_dir, out_dir = work / "in", work / "out"
    in_dir.mkdir(parents=True)

    by_seq = _sequence_records(shards)
    resolved = resolve_campaign(campaign, structure_root=DATA)

    # Absolute paths on the volume, resolved once per container and shared by
    # every sequence in the chunk. See `_target_alignments` for why absolute,
    # and why a path rather than the alignment text.
    alignments = _target_alignments(campaign, dict(resolved.target_sequences))
    ligands = _af3_ligand_entries(campaign, load_structure(resolved.structure_path))

    prepared, skipped, seeds = _cv_fold_inputs(
        seq_ids, by_seq, campaign, resolved, alignments, ligands, in_dir
    )
    if not prepared:
        return {"returncode": 0, "stage": "cv", "n_written": 0, "skipped": skipped}

    native = cv.weights is Weights.AF3_NATIVE
    implementation, xla_flags, nojit = attention_tuning()
    Path(AF3_FORK_JIT).mkdir(parents=True, exist_ok=True)
    command = [
        "python", "run_alphafold.py",
        # One process for the whole chunk, so the model loads and compiles
        # once. At ~100 s of start-up that is what chunking exists for.
        f"--input_dir={in_dir}",
        f"--output_dir={out_dir}",
        f"--model_dir={AF3_FORK_NATIVE if native else AF3_FORK_OF3}",
        f"--cache_dir={AF3_FORK_JIT}",
        # Never AF3's own genetic search, which wants the local databases this
        # fork exists to avoid -- and never `--use_msa_server`, since every
        # chain's alignment is already decided above.
        "--norun_data_pipeline",
        # Without this a re-run appends a timestamp to the job directory, and
        # the output stops being addressable by seq_id.
        "--force_output_dir",
        f"--flash_attention_implementation={implementation}",
        # AF3's own default, and the reference notebook's. Not configurable
        # because nothing has yet suggested a reason to vary it; if that
        # changes it belongs on `CrossValidationConfig`, not here.
        "--num_recycles=10",
        # One sample per seed. The flag's own default is 5, so leaving it alone
        # would turn five seeds into twenty-five predictions per sequence.
        "--num_diffusion_samples=1",
    ]
    if nojit:
        command.append("--nojit")
    if not native:
        # Needed even with pre-converted weights: it selects OF3-compatible
        # model settings, not merely the conversion.
        command.append("--of3_weights")

    print(
        f"cv: {len(prepared)} sequence(s), {len(seeds)} seed(s) {seeds}, "
        f"weights {cv.weights.value}, attention {implementation}"
        + (f", ligand chains {[e['ligand']['id'] for e in ligands]}" if ligands else ""),
        flush=True,
    )
    returncode, output_tail = _stream(
        command,
        timeout=6 * HOUR,
        cwd="/root",
        env={"XLA_FLAGS": " ".join(xla_flags)} if xla_flags else None,
    )

    # Written whatever the exit code. The af3_jax probe learned this the other
    # way round -- the kit's `rc=5` came with ten complete predictions, and
    # returning early would have thrown them away. One malformed input in a
    # chunk of sixteen should not discard the fifteen that folded, so
    # `_cv_metrics` records a per-sequence `cv_error` instead.
    written = _cv_write_records(
        prepared,
        by_seq,
        shards=shards,
        root=root,
        out_dir=out_dir,
        out_root=root / "cv",
        binder=campaign.general.chain_binder,
        seeds=seeds,
        metrics=_cv_metrics,
        # Full matrices at smoke scale only -- see `_af3_keep`.
        keep=lambda record: _af3_keep(
            record, full_confidences=campaign.run.dry_run
        ),
        job_dir_of=lambda out, seq_id: out / _af3_job_name(seq_id),
        campaign=campaign,
    )

    data_volume.commit()
    # The compiled kernels live on the jit volume and are worth ~100 s of
    # start-up per container, but only if they are pushed. `install_esm_jit`
    # commits this volume; without the same call here every container
    # recompiled from cold.
    jit_volume.commit()
    shutil.rmtree(work, ignore_errors=True)
    print(f"cv: exit {returncode}, {len(written)} record(s)", flush=True)
    return {
        "returncode": returncode,
        "stage": "cv",
        "weights": cv.weights.value,
        "seeds": seeds,
        "n_written": len(written),
        "skipped": skipped,
        "output_tail": output_tail,
    }


@app.function(image=af3_fork_image, gpu="A100-80GB", volumes=VOLUMES, timeout=6 * HOUR)
def cross_validate_af3_fork(seq_ids: list[str], campaign_payload: dict) -> dict:
    """The fork's wrapper. One per model, because a function's image is fixed
    at decoration time."""
    return _cross_validate_af3(seq_ids, campaign_payload)


# Memory and CPU for the conversion: the checkpoint is thousands of tensors
# read into RAM at once, and the two downloads are the rest of the wall time.
@app.function(
    image=af3_fork_image, volumes=VOLUMES, timeout=2 * HOUR, cpu=4.0, memory=32768
)
def populate_af3_fork_weights(native: bool = False) -> dict:
    """Land the fork's weights on the volume. Idempotent.

    Each step is skipped when its output already exists, so a re-run after a
    partial failure costs only the missing piece.

    Two sets, and unlike the af3_jax kit this fork loads either -- which is why
    `weights: af3_native` is a one-line campaign change here rather than a
    weights-conversion project. `--native` also fetches the official AlphaFold3
    parameters; their terms are non-commercial, so they are opt-in and used
    under the clearance already held for them.

    `--of3_checkpoint` would let `run_alphafold.py` convert on first use and
    cache into `--model_dir`, but converting here keeps it off the prediction
    path, where it would be paid once per cold container instead of once ever.
    """
    import hashlib

    report: dict[str, object] = {}
    Path(AF3_FORK_ROOT).mkdir(parents=True, exist_ok=True)

    # OpenFold3: public, unauthenticated, Apache-2.0.
    if not Path(AF3_FORK_CHECKPOINT).is_file():
        returncode, tail = _stream(
            ["aws", "s3", "cp", "s3://openfold/staging/of3-p2-155k.pt",
             AF3_FORK_CHECKPOINT, "--no-sign-request"],
            # aws prints a progress line per tick, and a long tail of those
            # buries every other field in the report.
            timeout=40 * 60, tail_lines=3,
        )
        report["of3_fetch"] = {"returncode": returncode, "tail": tail}
        if returncode != 0:
            return report
    report["of3_checkpoint_bytes"] = Path(AF3_FORK_CHECKPOINT).stat().st_size

    converted = Path(AF3_FORK_OF3).is_dir() and any(Path(AF3_FORK_OF3).glob("*"))
    if not converted:
        returncode, tail = _stream(
            ["python", "convert_of3_weights.py",
             f"--of3_checkpoint={AF3_FORK_CHECKPOINT}",
             f"--output_dir={AF3_FORK_OF3}"],
            cwd="/root", timeout=HOUR,
        )
        report["of3_convert"] = {"returncode": returncode, "tail": tail}
        if returncode != 0:
            weights_volume.commit()
            return report
    report["of3_dir"] = sorted(p.name for p in Path(AF3_FORK_OF3).iterdir())

    if native:
        Path(AF3_FORK_NATIVE).mkdir(parents=True, exist_ok=True)
        target = Path(AF3_FORK_NATIVE, "af3.bin.zst")
        if not target.is_file():
            returncode, tail = _stream(
                ["wget", "-q", "-O", str(target),
                 "https://storage.googleapis.com/alphafold3/af3.bin.zst"],
                timeout=HOUR,
            )
            report["native_fetch"] = {"returncode": returncode, "tail": tail}
        if target.is_file():
            digest = hashlib.sha256()
            with target.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 22), b""):
                    digest.update(block)
            report["native"] = {
                "bytes": target.stat().st_size,
                "sha256": digest.hexdigest()[:16],
            }

    weights_volume.commit()
    return report


CV_FUNCTIONS["af3_fork"] = cross_validate_af3_fork


# ---------------------------------------------------------------------------
# Reduce
# ---------------------------------------------------------------------------
@app.function(image=cpu_image, volumes=VOLUMES, timeout=HOUR)
def reduce_summaries(campaign_payload: dict) -> dict:
    """Rebuild summary.csv and summary.json from the shards.

    Derived artifacts, so this is safe to re-run at any point and is how a
    restarted campaign recovers its summaries.
    """
    from pathlib import Path

    from rfd3_pipeline.gates import Stage, backbone_gates, refold_gates
    from rfd3_pipeline.report import latest_records, read_shards, write_summaries

    # Every shard was written by some other container.
    data_volume.reload()

    campaign = _campaign_from(campaign_payload)
    rfd3 = campaign.rfdiffusion3
    root = Path(_run_root(campaign))
    shards = read_shards(root / "shards")
    # One row per sequence, plus the backbones that died before MPNN. Without
    # this a passing backbone is counted once as itself and again per sequence.
    records = latest_records(shards)

    paths = write_summaries(
        root,
        records,
        campaign=campaign.general.name_design_campaign,
        run_name=campaign.general.run_name,
        n_specs=len(campaign.rfdiffusion3.specs),
        # Every gated stage, not just the backbone filters. A stage missing
        # from here is missing from the funnel, which is how the summary came
        # to describe two stages out of six.
        stage_gates={
            Stage.BACKBONE_FILTERS: backbone_gates(campaign.rfdiffusion3),
            Stage.APO: refold_gates(campaign.refolding, Stage.APO),
            # Campaign-level here, where the per-design record is not in view:
            # a spec that sets `unindex` is scaffolding a motif.
            Stage.HOLO: refold_gates(
                campaign.refolding,
                Stage.HOLO,
                has_motif=any(spec.unindex for spec in rfd3.specs),
            ),
        },
        dry_run=campaign.run.dry_run,
        # What the campaign asked for. The records cannot say whether three
        # designs means "three were configured" or "the rest failed", and at
        # dry-run scale those look identical.
        requested={
            "n_batches": rfd3.n_batches,
            "diffusion_batch_size": rfd3.diffusion_batch_size,
            "designs_per_spec": rfd3.designs_per_spec,
            "total_designs": rfd3.total_designs,
            "n_mpnn_seqs_per_design": campaign.mpnn.num_seqs,
            "max_sequences": rfd3.total_designs * campaign.mpnn.num_seqs,
            "n_cv_seeds": campaign.cross_validation.num_seeds,
            # Recorded beside the scores, because ipSAE at one cutoff is not
            # comparable with ipSAE at another and every other implementation
            # defaults to 15.
            "ipsae_pae_cutoff": campaign.ipsae.pae_cutoff,
        },
    )
    data_volume.commit()

    # Distinct designs, not rows: a passing backbone with eight sequences is
    # one design that passed, not eight.
    passing = len(
        {
            r.get("design_id")
            for r in records
            if r.get("bb_pass") is True and r.get("design_id")
        }
    )
    return {
        "n_records": len(records),
        "n_passing_backbone_filters": passing,
        # Mirrors the funnel so the console says as much as the file does.
        "n_passing": {
            stage: sum(1 for r in records if r.get(f"{prefix}pass") is True)
            for stage, prefix in (("apo", "apo_"), ("holo", "holo_"))
        },
        "n_cross_validated": sum(1 for r in records if r.get("cv_n_seeds_ok")),
        "dry_run": campaign.run.dry_run,
        "written": {name: str(path) for name, path in paths.items()},
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@app.function(image=cpu_image, volumes=VOLUMES, timeout=24 * HOUR)
def orchestrate(campaign_payload: dict, start_at: str = "rfd3", chunk_size: int = 32) -> dict:
    """Run the campaign. Remote, so the client may disconnect.

    A local entrypoint doing this work would put the laptop in the control flow;
    `modal run --detach` only keeps already-dispatched functions alive, not the
    coordination between them.
    """
    from rfd3_pipeline.gates import Stage

    campaign = _campaign_from(campaign_payload)
    rfd3 = campaign.rfdiffusion3
    report: dict[str, object] = {"start_at": start_at}

    print(f"orchestrate: start_at={start_at}, {len(rfd3.specs)} spec(s)", flush=True)

    root = Path(_run_root(campaign))
    # The whole order now that cross-validation exists. This was truncated at
    # HOLO while CV was unbuilt, which also made `--start-at cv` unreachable.
    stages = Stage.ORDER
    if start_at not in stages:
        raise ValueError(f"start_at must be one of {list(stages)}, got {start_at!r}")
    reached = stages.index(start_at)

    def chunked(items: list[str]) -> list[list[str]]:
        return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    if reached <= stages.index(Stage.RFD3):
        spec_names = [spec.name for spec in rfd3.specs]
        print(
            f"phase rfd3: fanning out to {spec_names}, "
            f"max_concurrent={rfd3.max_concurrent}",
            flush=True,
        )
        # `max_concurrent` is campaign config, so it cannot be a decorator
        # argument: decorators run at import, before any campaign is read.
        # `with_options` is the runtime equivalent.
        rfd3_results = list(
            # `gpu=` as well as `max_containers=`: without it the
            # decorator's default wins and `rfdiffusion3.gpu` is silently
            # ignored, which reads as working only because the two agree.
            run_rfd3.with_options(
                gpu=rfd3.gpu, max_containers=rfd3.max_concurrent
            ).map(
                spec_names, kwargs={"campaign_payload": campaign_payload}
            )
        )
        report["rfd3"] = rfd3_results
        designs = [d for result in rfd3_results for d in result["designs"]]
    else:
        data_volume.reload()
        designs = sorted(p.stem for p in (root / "rfd3" / "out").glob("*.json"))
    report["n_designs"] = len(designs)
    print(f"phase rfd3 done: {len(designs)} designs", flush=True)

    if reached <= stages.index(Stage.BACKBONE_FILTERS):
        chunks = chunked(designs)
        print(f"phase filter: {len(chunks)} chunk(s) of up to {chunk_size}", flush=True)
        report["backbone_filters"] = list(
            filter_designs.map(chunks, kwargs={"campaign_payload": campaign_payload})
        )

    # Only backbones that passed go to MPNN. The verdict lives in the shards
    # rather than in the filter step's return value, so that restarting at
    # `mpnn` selects exactly the same set as a continuous run would.
    #
    # `reload()` is not optional, and its absence is silent. This container is
    # long-lived, so its view of the volume is whatever it was when the
    # container started; the filter containers' `commit()` makes their shards
    # durable but does not push them into an already-running reader. Without
    # this, the first end-to-end run found zero passing designs, skipped MPNN
    # and exited 0, while the reducer -- a fresh container -- saw all three.
    data_volume.reload()
    passing = _passing_designs(root)
    report["n_passing_backbone_filters"] = len(passing)

    # Gated like every other phase. Without this, `--start-at apo` re-ran MPNN
    # and regenerated sequences that already existed -- harmless at dry-run
    # scale because MPNN is deterministic under a fixed seed, but a wasted GPU
    # pass over every passing design at campaign scale, and it defeats the
    # point of starting late.
    if reached <= stages.index(Stage.MPNN):
        chunks = chunked(passing)
        print(
            f"phase mpnn: {len(passing)} of {len(designs)} passed, "
            f"{len(chunks)} chunk(s) of up to {chunk_size}, "
            f"max_concurrent={campaign.mpnn.max_concurrent}",
            flush=True,
        )
        if chunks:
            report["mpnn"] = list(
                run_mpnn.with_options(
                    gpu=campaign.mpnn.gpu,
                    max_containers=campaign.mpnn.max_concurrent,
                ).map(
                    chunks, kwargs={"campaign_payload": campaign_payload}
                )
            )

    # The sequences MPNN just wrote were committed by other containers.
    data_volume.reload()
    sequences = _sequence_ids(root)
    report["n_sequences"] = len(sequences)

    refolding = campaign.refolding
    refold = REFOLD_FUNCTIONS.get(refolding.model.value)
    if refold is None:
        raise ValueError(
            f"no refold function for model {refolding.model.value!r}; "
            f"have {sorted(REFOLD_FUNCTIONS)}. Add an image and a wrapper."
        )

    def refold_chunks(ids: list[str]) -> list[list[str]]:
        """Sized by the refolding config, not the design `chunk_size`: a fold
        amortises ~180 s of process start-up, so the right batch is different."""
        size = refolding.chunk_size
        return [ids[i : i + size] for i in range(0, len(ids), size)]

    def run_stage(stage: str, ids: list[str]) -> None:
        chunks = refold_chunks(ids)
        print(
            f"phase {stage}: {len(ids)} sequence(s), {len(chunks)} chunk(s) of "
            f"up to {refolding.chunk_size}, max_concurrent="
            f"{refolding.max_concurrent}",
            flush=True,
        )
        if chunks:
            report[stage] = list(
                refold.with_options(
                    gpu=refolding.gpu, max_containers=refolding.max_concurrent
                ).map(
                    chunks,
                    kwargs={"campaign_payload": campaign_payload, "stage": stage},
                )
            )

    # Gated like every other phase. An ungated one is invisible until a stage
    # exists beyond it -- which is exactly how the MPNN phase came to re-run on
    # `--start-at apo`.
    if reached <= stages.index(Stage.APO):
        run_stage("apo", sequences)

    if reached <= stages.index(Stage.HOLO):
        # Holo takes apo's survivors, not every sequence. Apo folds one chain
        # and holo folds the whole complex, so apo is the cheap filter that
        # should spend holo's budget only on sequences that folded as designed.
        data_volume.reload()
        survivors = _passing_sequences(root, "apo")
        report["n_passing_apo"] = len(survivors)
        if not survivors and sequences:
            # Loud, because the alternative is a holo phase that silently does
            # nothing and a summary with no holo columns at all.
            print(
                f"phase holo: SKIPPED -- 0 of {len(sequences)} sequences passed "
                f"apo (threshold_plddt_binder={refolding.threshold_plddt_binder}). "
                f"Nothing to fold.",
                flush=True,
            )
        run_stage("holo", survivors)

    if reached <= stages.index(Stage.CV):
        # Cross-validation takes holo's survivors. Its own chunking and
        # concurrency, because it is a different model on a different image
        # with a different per-process start-up cost.
        data_volume.reload()
        cv = campaign.cross_validation
        validate = CV_FUNCTIONS.get(cv.model.value)
        if validate is None:
            raise ValueError(
                f"no cross-validation function for model {cv.model.value!r}; "
                f"have {sorted(CV_FUNCTIONS)}. Add an image and a wrapper."
            )
        confirmed = _passing_sequences(root, "holo")
        report["n_passing_holo"] = len(confirmed)
        chunks = [
            confirmed[i : i + cv.chunk_size]
            for i in range(0, len(confirmed), cv.chunk_size)
        ]
        print(
            f"phase cv: {len(confirmed)} sequence(s), {len(chunks)} chunk(s) of "
            f"up to {cv.chunk_size}, {cv.num_seeds} seed(s) each, model "
            f"{cv.model.value}, max_concurrent={cv.max_concurrent}",
            flush=True,
        )
        if not confirmed and sequences:
            # Loud, for the same reason the holo skip is: a phase that silently
            # does nothing leaves a summary with no cv columns and no
            # explanation of why.
            print(
                f"phase cv: SKIPPED -- 0 sequences passed holo "
                f"(threshold_ipTM={refolding.threshold_ipTM}, "
                f"threshold_rmsd_apo_holo={refolding.threshold_rmsd_apo_holo}). "
                f"Nothing to cross-validate.",
                flush=True,
            )
        if chunks:
            report["cv"] = list(
                validate.with_options(
                    gpu=cv.gpu, max_containers=cv.max_concurrent
                ).map(chunks, kwargs={"campaign_payload": campaign_payload})
            )

    print("phase reduce", flush=True)
    report["reduce"] = reduce_summaries.remote(campaign_payload)
    print("orchestrate: done", flush=True)
    return report


def _sequence_ids(root) -> list[str]:
    """Every MPNN sequence on the volume, as `seq_id`s.

    Read from the shards rather than from the MPNN step's return value, so that
    restarting at `apo` selects exactly the set a continuous run would.
    """
    from rfd3_pipeline.report import read_shards

    return sorted(
        record["seq_id"]
        for record in read_shards(root / "shards")
        if record.get("seq_id") and record.get("sequence")
    )


def _passing_sequences(root, stage: str) -> list[str]:
    """Sequence ids whose shard says they passed `stage`.

    Read from the shards rather than from the stage's return value, so
    restarting at a later stage selects exactly the set a continuous run would.
    """
    from rfd3_pipeline.report import read_shards

    flag = f"{stage}_pass"
    return sorted(
        record["seq_id"]
        for record in read_shards(root / "shards")
        if record.get("seq_id") and record.get(flag) is True
    )


def _passing_designs(root) -> list[str]:
    """Design IDs whose backbone shard says they passed the filters.

    Reads the backbone shards specifically -- those with a `design_id` and no
    `seq_id` -- so a re-run over a directory that already holds sequence shards
    does not feed sequence rows back in as designs.
    """
    from rfd3_pipeline.report import read_shards

    return sorted(
        record["design_id"]
        for record in read_shards(root / "shards")
        if record.get("design_id")
        and not record.get("seq_id")
        and record.get("bb_pass") is True
    )


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------
def _check_models_are_wired(campaign) -> None:
    """Fail now if a configured model has no function to run it.

    `FoldingModel` lists every model the *config* understands, which is a
    larger set than the ones `app.py` can actually dispatch -- a model needs an
    image and a wrapper, and only the ones a campaign uses are worth building.
    Config validation cannot check this itself: the registries live here, and
    `rfd3_pipeline` must not import `app`.

    Without this, a campaign naming an unwired cross-validation model validates
    happily and then dies in `orchestrate` *after* RFD3, MPNN, apo and holo
    have all run -- hours of GPU time to learn about a one-word typo. Checked on
    the client, where it costs a second.
    """
    for stage, model, registry, names in (
        ("refolding", campaign.refolding.model, REFOLD_FUNCTIONS, "REFOLD_FUNCTIONS"),
        (
            "cross_validation",
            campaign.cross_validation.model,
            CV_FUNCTIONS,
            "CV_FUNCTIONS",
        ),
    ):
        if model.value not in registry:
            raise SystemExit(
                f"{stage}.model is {model.value!r}, which has no function in "
                f"{names}; wired models are {sorted(registry)}. Adding one "
                f"means an image, a wrapper and a registry entry -- see "
                f"`_cross_validate_af3`'s docstring for the full list."
            )


def _prepare(campaign_path: str, dry_run: bool = False) -> tuple[dict, str, str]:
    """Validate and resolve locally, then upload what the run needs.

    Resolution happens here so a bad campaign fails on the client in seconds,
    before a container is started.
    """
    from pathlib import Path

    import yaml

    from rfd3_pipeline.config import CampaignConfig, apply_dry_run, dry_run_banner
    from rfd3_pipeline.spec import dump_rfd3_specs, resolve_campaign

    repo = Path(__file__).resolve().parent
    payload = yaml.safe_load(Path(campaign_path).read_text())
    if dry_run:
        payload = apply_dry_run(payload)

    campaign = CampaignConfig.model_validate(payload)
    _check_models_are_wired(campaign)

    # Printed after validation so the banner reports the settings that will
    # actually run, not the ones requested.
    if campaign.run.dry_run:
        rule = "=" * 72
        print(f"\n{rule}\n{dry_run_banner(campaign)}\n{rule}\n")

    resolved = resolve_campaign(campaign, structure_root=repo)
    for warning in resolved.warnings:
        print(f"WARNING: {warning}")

    general = campaign.general
    remote_root = f"{general.name_design_campaign}/{general.run_name}"
    structure = campaign.rfdiffusion3.path_input_structure

    with data_volume.batch_upload(force=True) as upload:
        upload.put_file(repo / structure, structure)
        # The container path, not the client one: RFD3 resolves a relative
        # `input` against the specs file's own directory.
        upload.put_file(
            io_bytes(dump_rfd3_specs(resolved, input_path=f"{DATA}/{structure}")),
            f"{remote_root}/rfd3/specs.yaml",
        )
        upload.put_file(
            io_bytes(yaml.safe_dump(resolved.to_dict(), sort_keys=False)),
            f"{remote_root}/campaign.resolved.yaml",
        )

    print(f"uploaded specs.yaml and {structure} for {remote_root}")
    for spec in resolved.specs:
        print(f"  {spec.name}: expected length {spec.length_string}")
    return campaign.model_dump(mode="json"), remote_root, structure


def io_bytes(text: str):
    import io

    return io.BytesIO(text.encode())


@app.local_entrypoint()
def run(
    campaign: str,
    start_at: str = "rfd3",
    chunk_size: int = 32,
    dry_run: bool = False,
    spawn: bool = False,
):
    """Run a campaign.

    `--dry-run` reduces it to smoke-test scale -- one spec, two designs, one
    cross-validation seed -- and renames the run so its output lands in a
    separate directory. A campaign that mysteriously produced two designs is
    almost always this flag, which is why the rename and the banner exist.

    `--spawn` hands the orchestrator to the server and returns immediately,
    which is the only form that survives this client exiting. **`--detach` is
    not enough**, learned the expensive way on 2026-10-09: a campaign launched
    with `modal run --detach` produced all 80 RFD3 designs and then stopped
    dead before the filters, because detached mode keeps only the *last
    triggered* call alive -- and by the time the client died that was
    `run_rfd3`, spawned by the orchestrator rather than by this entrypoint. The
    children outlived the parent. A spawned call is owned by the server and has
    no parent to lose.

    Use `--spawn` for anything long, then follow it with `reduce_only` or by
    reading `summary.json` off the volume. Without it this blocks and streams,
    which is what you want for a dry run.
    """
    payload, _, _ = _prepare(campaign, dry_run=dry_run)
    if spawn:
        call = orchestrate.spawn(payload, start_at=start_at, chunk_size=chunk_size)
        print(
            f"\nspawned orchestrator, call id {call.object_id}\n"
            f"  it runs server-side and does not need this client.\n"
            f"  results land in summary.json / summary.csv under the run "
            f"directory; `reduce_only` rebuilds them from the shards.\n"
        )
        return
    report = orchestrate.remote(payload, start_at=start_at, chunk_size=chunk_size)
    _show(report)


@app.local_entrypoint()
def install_weights():
    _show(populate_weights.remote())


@app.local_entrypoint()
def install_mpnn_weights():
    _show(populate_mpnn_weights.remote())


@app.local_entrypoint()
def warm_esm_jit():
    _show(install_esm_jit.remote())


@app.local_entrypoint()
def install_af3_fork_weights(native: bool = False):
    """`--native` also fetches the official AlphaFold3 parameters.

    Opt-in because their terms are non-commercial, unlike OpenFold3's Apache-2.0.
    """
    _show(populate_af3_fork_weights.remote(native=native))


@app.local_entrypoint()
def refold_only(campaign: str, stage: str = "apo", dry_run: bool = False, limit: int = 0):
    """Run one refolding stage over the sequences already on the volume.

    For exercising a stage on its own, before or instead of wiring it into
    `orchestrate`. `--limit N` takes the first N sequences, which is how a new
    stage gets tried on one input rather than on all of them.
    """
    payload, _, _ = _prepare(campaign, dry_run=dry_run)
    _show(_refold_one_stage.remote(payload, stage=stage, limit=limit))


@app.local_entrypoint()
def cross_validate_only(campaign: str, dry_run: bool = False, limit: int = 0):
    """Cross-validate the sequences holo already confirmed.

    The counterpart to `refold_only`, and useful for the same reason: this is
    the last stage, so re-running it on its own costs nothing upstream.
    `--limit N` takes the first N, which is how it gets tried on one sequence
    rather than on all of them.
    """
    payload, _, _ = _prepare(campaign, dry_run=dry_run)
    _show(_cross_validate_stage.remote(payload, limit=limit))


@app.function(image=cpu_image, volumes=VOLUMES, timeout=24 * HOUR)
def _cross_validate_stage(campaign_payload: dict, limit: int = 0) -> dict:
    """Pick the sequences and dispatch cross-validation, remotely.

    A thin coordinator rather than part of the entrypoint, so the client is not
    in the control flow for a long fold.
    """
    data_volume.reload()

    campaign = _campaign_from(campaign_payload)
    cv = campaign.cross_validation
    root = Path(_run_root(campaign))
    # Holo's survivors, the same set `orchestrate` would pass. Falling back to
    # every sequence would quietly spend the most expensive stage in the
    # pipeline on designs holo already rejected.
    sequences = _passing_sequences(root, "holo")
    if limit:
        sequences = sequences[:limit]

    validate = CV_FUNCTIONS.get(cv.model.value)
    if validate is None:
        raise ValueError(
            f"no cross-validation function for model {cv.model.value!r}; "
            f"have {sorted(CV_FUNCTIONS)}"
        )

    chunks = [
        sequences[i : i + cv.chunk_size]
        for i in range(0, len(sequences), cv.chunk_size)
    ]
    print(
        f"phase cv: {len(sequences)} sequence(s), {len(chunks)} chunk(s) of up "
        f"to {cv.chunk_size}, {cv.num_seeds} seed(s) each, model {cv.model.value}",
        flush=True,
    )
    if not chunks:
        return {"stage": "cv", "n_sequences": 0}

    results = list(
        validate.with_options(
            gpu=cv.gpu, max_containers=cv.max_concurrent
        ).map(chunks, kwargs={"campaign_payload": campaign_payload})
    )
    return {"stage": "cv", "n_sequences": len(sequences), "cv": results}


@app.function(image=cpu_image, volumes=VOLUMES, timeout=24 * HOUR)
def _refold_one_stage(campaign_payload: dict, stage: str, limit: int = 0) -> dict:
    """Pick the sequences and dispatch one stage, remotely.

    A thin coordinator rather than part of `refold_only`, so the client is not
    in the control flow for a long fold.
    """
    data_volume.reload()

    campaign = _campaign_from(campaign_payload)
    refolding = campaign.refolding
    root = Path(_run_root(campaign))
    sequences = _sequence_ids(root)
    if limit:
        sequences = sequences[:limit]

    refold = REFOLD_FUNCTIONS.get(refolding.model.value)
    if refold is None:
        raise ValueError(
            f"no refold function for model {refolding.model.value!r}; "
            f"have {sorted(REFOLD_FUNCTIONS)}"
        )

    chunks = [
        sequences[i : i + refolding.chunk_size]
        for i in range(0, len(sequences), refolding.chunk_size)
    ]
    print(
        f"phase {stage}: {len(sequences)} sequence(s), {len(chunks)} chunk(s) "
        f"of up to {refolding.chunk_size}",
        flush=True,
    )
    if not chunks:
        return {"stage": stage, "n_sequences": 0}

    results = list(
        refold.with_options(
            gpu=refolding.gpu, max_containers=refolding.max_concurrent
        ).map(chunks, kwargs={"campaign_payload": campaign_payload, "stage": stage})
    )
    return {"stage": stage, "n_sequences": len(sequences), stage: results}


@app.local_entrypoint()
def cache_msas(campaign: str, dry_run: bool = False, force: bool = False):
    """Build the target MSA cache for a campaign, once.

    Separate from `run` because it is the one step that leaves the network, and
    because it is campaign-scoped rather than per run -- so it should be
    possible to do it deliberately and inspect the result before any folding.
    """
    payload, _, _ = _prepare(campaign, dry_run=dry_run)
    _show(build_msa_cache.remote(payload, force=force))


@app.local_entrypoint()
def filter_only(campaign: str, dry_run: bool = False, limit: int = 0):
    """Run the backbone filters over the designs already on the volume.

    Exists because the filters are the one stage with no way to see their own
    failures: `filter_designs` catches per design so a bad one cannot lose the
    chunk, and returns the reasons in its report -- which goes to whoever
    called it. Under `--spawn` that is the server, so a campaign where every
    design failed the filters produced a summary of zeros and no explanation
    anywhere. This is how to get the explanation, in seconds, on CPU.

    `--limit N` filters only the first N designs, which is enough to see why.
    """
    payload, _, _ = _prepare(campaign, dry_run=dry_run)
    _show(_filter_one_pass.remote(payload, limit=limit))


@app.function(image=cpu_image, volumes=VOLUMES, timeout=HOUR)
def _filter_one_pass(campaign_payload: dict, limit: int = 0) -> dict:
    """Pick the designs and filter them, reporting every failure reason."""
    data_volume.reload()

    campaign = _campaign_from(campaign_payload)
    root = Path(_run_root(campaign))
    out_dir = root / "rfd3" / "out"
    designs = sorted(p.name[: -len(".cif.gz")] for p in out_dir.glob("*.cif.gz"))
    if limit:
        designs = designs[:limit]
    if not designs:
        return {"n_designs": 0, "note": f"no designs under {out_dir}"}

    result = filter_designs.local(designs, campaign_payload)
    # The reasons are the point, so they are surfaced rather than counted.
    failures = result.get("failures", [])
    return {
        "n_designs": len(designs),
        "n_written": result.get("n_written"),
        "n_failed": len(failures),
        "first_failures": failures[:5],
    }


@app.local_entrypoint()
def reduce_only(campaign: str, dry_run: bool = False):
    """Rebuild the summaries from existing shards, without re-running anything.

    `--dry-run` has to be passed here too, and for the same reason the flag
    renames the run: without it this reduces `{run_name}`, while the dry run's
    shards are under `{run_name}-dryrun`, and the result is a summary of
    nothing rather than an error.
    """
    import yaml
    from pathlib import Path

    from rfd3_pipeline.config import CampaignConfig, apply_dry_run

    payload = yaml.safe_load(Path(campaign).read_text())
    if dry_run:
        payload = apply_dry_run(payload)
    _show(
        reduce_summaries.remote(
            CampaignConfig.model_validate(payload).model_dump(mode="json")
        )
    )


def _show(report: dict, indent: str = "") -> None:
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"{indent}{key}:")
            _show(value, indent + "  ")
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            print(f"{indent}{key}:")
            for item in value:
                _show(item, indent + "  ")
                print()
        else:
            shown = str(value)
            print(f"{indent}{key}: {shown[:300]}")
