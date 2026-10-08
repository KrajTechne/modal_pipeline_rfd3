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
PROJECT_DEPS = ("numpy>=1.26", "biotite>=1.0", "pyyaml>=6.0", "pydantic>=2.0")

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

HOUR = 60 * 60


def _stream(
    command: list[str],
    *,
    timeout: int,
    cwd: str | None = None,
    tail_lines: int = 60,
) -> tuple[int, str]:
    """Run a command, echoing its output so the Modal log view shows progress.

    `capture_output=True` holds every line in a pipe until the process exits,
    which on a six-hour design job leaves no way to tell a slow run from a hung
    one. Lines are printed as they arrive; a bounded tail is still returned for
    the report. stderr is merged into stdout to keep the interleaving the
    subprocess intended.
    """
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
    from rfd3_pipeline.spec import load_design_metadata, resolve_campaign
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

            record = backbone_record(structure, metadata, spec, campaign)
            record["design_id"] = design_id
            record["campaign"] = campaign.general.name_design_campaign
            record["run_name"] = campaign.general.run_name
            write_shard(record, shards)
            written.append(design_id)
        except Exception as error:  # one bad design must not lose the chunk
            failed.append({"design_id": design_id, "error": f"{type(error).__name__}: {error}"})

    data_volume.commit()
    return {"n_written": len(written), "failures": failed}


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

    from rfd3_pipeline.config import folding_variant
    from rfd3_pipeline.gates import Stage
    from rfd3_pipeline.report import read_shards, write_shard

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

    variant = folding_variant(refolding.model, supplies_msa)
    if variant is None:
        raise ValueError(
            f"{refolding.model.value} has no --variant for msa_present="
            f"{supplies_msa}; its MODEL_CAPABILITIES row needs filling in"
        )

    by_seq = {
        record["seq_id"]: record
        for record in read_shards(shards)
        if record.get("seq_id") and record.get("sequence")
    }

    # Target sequences come from the input structure once per campaign, not from
    # the shards -- they are identical for every prediction.
    target_sequences: dict[str, str] = {}
    if include_targets:
        from rfd3_pipeline.spec import resolve_campaign

        resolved = resolve_campaign(campaign, structure_root=DATA)
        target_sequences = dict(resolved.target_sequences)

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
                        # Set explicitly even when empty: `fill_missing_msas`
                        # gates the unpaired and paired queries on *different*
                        # fields, so leaving either as null sends the sequence
                        # out. The target MSA cache fills this in when built.
                        "msa": None,
                    }
                )

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
        "--seeds", str(campaign.general.base_seed),
        "--det", "1",
    ]
    print(f"{stage}: {len(prepared)} sequence(s), variant {variant}", flush=True)
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
                seq_id, stage, rows.get(seq_id), out_dir, root, binder, campaign
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


def _refold_metrics(seq_id, stage, row, out_dir, root, binder, campaign) -> dict:
    """The `{stage}_*` columns for one sequence.

    `pred_rows.jsonl` supplies what the kit measured; the rest is computed from
    the predicted structure. Every field is optional on purpose -- a missing
    metric should leave a null in the summary rather than lose the chunk.
    """
    prefix = f"{stage}_"
    metrics: dict[str, object] = {
        f"{prefix}status": (row or {}).get("status"),
        f"{prefix}ptm": _as_float((row or {}).get("ptm")),
        f"{prefix}iptm": _as_float((row or {}).get("iptm")),
        f"{prefix}n_tokens": _as_int((row or {}).get("n_tokens")),
        f"{prefix}wall_s": _as_float((row or {}).get("wall_s")),
        # Positive evidence about the MSA rather than trust in the flag: 0 means
        # the fold genuinely saw no alignment.
        f"{prefix}msa_depth": max((row or {}).get("msa_depths") or [0]),
    }

    cifs = sorted((out_dir / "cif_all").glob(f"{seq_id}__*.cif"))
    if not cifs:
        metrics[f"{prefix}error"] = "no cif produced"
        return metrics

    from rfd3_pipeline.structure import chain_subset, ca_only, load_structure, match_ca

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
            from rfd3_pipeline.metrics.rmsd import superimpose_rmsd

            design = load_structure(design_path)
            design_ca, predicted_ca, _ = match_ca(design, predicted, chains=[binder])
            metrics[f"{prefix}rmsd_to_design"] = superimpose_rmsd(design_ca, predicted_ca)
        except Exception as error:
            metrics[f"{prefix}rmsd_to_design"] = None
            metrics[f"{prefix}rmsd_error"] = f"{type(error).__name__}: {error}"

    # One gate is available from apo alone. `threshold_rmsd_apo_holo` and
    # `threshold_ipTM` need holo, so they are not evaluated here.
    plddt = metrics.get(f"{prefix}plddt_binder")
    if stage == "apo" and plddt is not None:
        metrics["apo_pass"] = plddt >= campaign.refolding.threshold_plddt_binder
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
# Reduce
# ---------------------------------------------------------------------------
@app.function(image=cpu_image, volumes=VOLUMES, timeout=HOUR)
def reduce_summaries(campaign_payload: dict) -> dict:
    """Rebuild summary.csv and summary.json from the shards.

    Derived artifacts, so this is safe to re-run at any point and is how a
    restarted campaign recovers its summaries.
    """
    from pathlib import Path

    from rfd3_pipeline.gates import Stage, backbone_gates
    from rfd3_pipeline.report import latest_records, read_shards, write_summaries

    # Every shard was written by some other container.
    data_volume.reload()

    campaign = _campaign_from(campaign_payload)
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
        stage_gates={Stage.BACKBONE_FILTERS: backbone_gates(campaign.rfdiffusion3)},
        dry_run=campaign.run.dry_run,
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
    stages = Stage.ORDER[: Stage.ORDER.index(Stage.APO) + 1]
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
            run_rfd3.with_options(max_containers=rfd3.max_concurrent).map(
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
                run_mpnn.with_options(max_containers=campaign.mpnn.max_concurrent).map(
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

    # Apo folds the binder alone, so its chunks are sized by the refolding
    # config rather than the design chunk_size.
    apo_chunks = [
        sequences[i : i + refolding.chunk_size]
        for i in range(0, len(sequences), refolding.chunk_size)
    ]
    print(
        f"phase apo: {len(sequences)} sequence(s), {len(apo_chunks)} chunk(s) "
        f"of up to {refolding.chunk_size}, max_concurrent="
        f"{refolding.max_concurrent}",
        flush=True,
    )
    if apo_chunks:
        report["apo"] = list(
            refold.with_options(
                gpu=refolding.gpu, max_containers=refolding.max_concurrent
            ).map(
                apo_chunks,
                kwargs={"campaign_payload": campaign_payload, "stage": "apo"},
            )
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
):
    """Run a campaign.

    `--dry-run` reduces it to smoke-test scale -- one spec, two designs, one
    cross-validation seed -- and renames the run so its output lands in a
    separate directory. A campaign that mysteriously produced two designs is
    almost always this flag, which is why the rename and the banner exist.
    """
    payload, _, _ = _prepare(campaign, dry_run=dry_run)
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
