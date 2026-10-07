"""Probe: can the ESMFold2 optimization kit run in a Modal image?

Staged cheapest-first, so a failure is diagnosed without paying for the step
after it. The weights are ~27 GB and the image build compiles CUDA extensions,
so the order matters more here than for the earlier probes:

    modal run experiments/esmfold2_image.py --stage env       # GPU, no weights
    modal run experiments/esmfold2_image.py --stage wheels    # keep the compiled wheels
    modal run experiments/esmfold2_image.py --stage install   # ~27 GB fetch
    modal run experiments/esmfold2_image.py --stage check     # kit dry run
    modal run experiments/esmfold2_image.py --stage warm      # Triton precompile
    modal run experiments/esmfold2_image.py --stage run_one   # one real apo fold

Built from the kit's own Dockerfile rather than its route C (venv). Route C
needs uv, a C++ compiler, the CUDA 13.0 toolkit and two Biohub packages at
pinned git commits, and the README is explicit that anything off-pin exits 3.
The Dockerfile is the supported path, and `STACK=img_esmfold2_a100` is the A100
build: on sm80 the compute-capability-9.0 transition kernel stays off, which
shows up as `card=unsupported_card:sm80` on a LEVER line and is informational.

The kit lives OUTSIDE this repo, sparse-cloned to `~/ubm` (only `esmfold2/` and
`common/opt_core`, which is what the Dockerfile copies). `common/opt_core` alone
is 307 MB of prebuilt kernels for many torch x python x CUDA combinations, which
is why it is not vendored the way LigandMPNN was. Prepare it with:

    git clone --depth 1 --filter=blob:none --sparse -c core.autocrlf=false \\
        https://github.com/anthropics/uplifting-biomolecular-modeling.git ~/ubm
    cd ~/ubm && git sparse-checkout set esmfold2 common

**`core.autocrlf=false` is not optional on Windows**, where it is the default
`true`. git rewrites LF to CRLF on checkout, `from_dockerfile` uploads those
files into a Linux image, and the build dies at `RUN bash run.sh install` with
`run.sh: line 19: set: pipefail: invalid option name` -- bash reading
`pipefail\\r`. Worse if it got past there: `--config a100` *sources*
`configs/a100.env`, so every variable would carry a trailing `\\r` and the mode
would refuse with a confusing `exit 3`. To repair an existing clone:

    git -C ~/ubm config core.autocrlf false
    git -C ~/ubm rm --cached -r -q . && git -C ~/ubm reset --hard -q

Note that repairing it invalidates the `COPY esmfold2` layer, which precedes the
wheel build, so the ~44 minute compile runs again.

The context also needs an empty jitcache tar, which the kit does not ship:

    mkdir -p ~/ubm/_jitcache
    tar -cf ~/ubm/_jitcache/esmfold2-empty-jit.tar --files-from /dev/null

The Dockerfile's last COPY is `esmfold2/environment/Dockerfile
_jitcache/esmfold2-*-jit.ta[r] /tmp/jit/`. The `.ta[r]` bracket is the standard
Docker idiom for an *optional* source -- a bracket glob matching nothing is
tolerated so long as another source matched -- and the kit's own comment says
"build is identical when no tar is present". **Modal's `from_dockerfile` does
not implement that**: it fails the build with `COPY source glob pattern ... did
not match any files`. The following `RUN` guards each file with `[ -f "$t" ]`,
so an empty archive extracts nothing and the image is identical to having no
tar at all. That is why this is a context fix rather than a patch to their
Dockerfile.

Facts this encodes, read off `esmfold2/README.md` rather than assumed:

  * `--variant fast|full_msa|full_nomsa` is **required**, no default, one per
    process. Apo is always `full_nomsa`, which is what
    `MODEL_CAPABILITIES.derived_variants[False]` already returns.
  * `--config h100|h200|a100` is required on every command; the card follows the
    GPU, not the model.
  * A mode that cannot engage prints `NOT ACTIVE: <reason>` and **exits 3**. It
    never falls back to stock silently. Exit 1 = failed, 2 = usage.
  * Every config sets `ESMFOLD2_OPT_REQUIRE_FAST_ENV=1`, so *every* route --
    `off` included -- exits 3 unless flash-attention and TransformerEngine
    import. There is no unaccelerated escape hatch.
  * Each process pays ~25 s of start-up in every mode, so inputs must be
    batched per process. `--input` takes a list, which is what makes
    `refolding.chunk_size` worth having.
  * `HF_HOME` must name the weights root. `install --weights` fetches if absent,
    else hash-checks, so a complete directory may be mounted read-only.
  * A writable `MODEL_OPT_JIT_ROOT` keeps the Triton cache across containers.

Open questions this is here to answer:

  * **Does the host driver satisfy >= 580?** Everything else is moot if not, so
    `env` checks it first.
  * Does the image build at all, and do flash-attention and TransformerEngine
    import inside it?
  * How long does one apo fold take, which is what sizes `chunk_size` and
    `max_concurrent`.
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("esmfold2-image-probe")

weights = modal.Volume.from_name("rfd3-weights", create_if_missing=True)
jit = modal.Volume.from_name("rfd3-jit", create_if_missing=True)
data = modal.Volume.from_name("rfd3-pipeline", create_if_missing=True)

WEIGHTS = "/weights"
JIT = "/jit"
DATA = "/data"
ESM_WEIGHTS = f"{WEIGHTS}/esmfold2"
ESM_JIT = f"{JIT}/esmfold2"
KIT = "/kit/esmfold2"
CARD = "a100"
VOLUMES = {WEIGHTS: weights, JIT: jit, DATA: data}

UBM = Path.home() / "ubm"

esmfold2_image = (
    modal.Image.from_dockerfile(
        UBM / "esmfold2" / "environment" / "Dockerfile",
        context_dir=UBM,
        build_args={
            # Default is img_ef2_fa; this is the A100 stack.
            "STACK": "img_esmfold2_a100",
            # Cap the compile jobs rather than trusting the Dockerfile's
            # heuristic. It computes `JFA = MemAvailable / 9` for flash-attn and
            # caps that by core count, which assumes ~9 GB per job. Observed
            # 2026-10-07: 17 jobs completed on a builder reporting 328 GB, then
            # the identical build was OOM-killed (exit 137) on one reporting
            # 203 GB -- so real peak is above ~12 GB per job and the heuristic
            # over-subscribes. Modal allocates whichever builder is free, so
            # leaving it to the default makes success a coin flip. 8 jobs is
            # ~96 GB peak, which fits either machine; the cost is a longer
            # build, and it applies to all three wheels since the Dockerfile
            # uses one job count for everything.
            "BUILD_JOBS": "8",
        },
    )
    .env(
        {
            # The kit reads the weights root from HF_HOME, not from a flag.
            "HF_HOME": ESM_WEIGHTS,
            "MODEL_OPT_JIT_ROOT": ESM_JIT,
            "PYTHONUNBUFFERED": "1",
        }
    )
)

TIMEOUT = 3 * 60 * 60
GPU = "A100-40GB"


def _stream(command: list[str], *, cwd: str = KIT, tail_lines: int = 80):
    """Run a command, echoing output as it arrives, returning (code, tail).

    The same reasoning as `app.py`'s `_stream`: a ~27 GB download or a Triton
    compile behind `capture_output=True` is indistinguishable from a hang.
    """
    import subprocess
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
    for line in process.stdout or ():
        line = line.rstrip()
        tail.append(line)
        print(line, flush=True)
    if process.stdout is not None:
        process.stdout.close()
    return process.wait(), "\n".join(tail)


@app.function(image=esmfold2_image, gpu=GPU, volumes=VOLUMES, timeout=TIMEOUT)
def env() -> dict:
    """Cheapest meaningful stage: does the host and image satisfy the kit?

    No weights are touched. The driver check comes first because the kit
    requires >= 580 and nothing else matters if that fails.
    """
    import subprocess

    report: dict[str, object] = {}

    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    report["nvidia_smi"] = (smi.stdout or smi.stderr).strip()

    probe = (
        "import torch;"
        "print('torch', torch.__version__, 'cuda', torch.version.cuda,"
        " 'avail', torch.cuda.is_available());"
        "import flash_attn; print('flash_attn', flash_attn.__version__);"
        "import transformer_engine; print('transformer_engine ok')"
    )
    imports = subprocess.run(["python", "-c", probe], capture_output=True, text=True)
    # These two are the ESMFOLD2_OPT_REQUIRE_FAST_ENV gate: without them every
    # mode exits 3, `off` included.
    report["fast_env_imports"] = (imports.stdout or "").strip() or imports.stderr[-900:]
    report["fast_env_ok"] = imports.returncode == 0

    report["kit_dir_exists"] = Path(KIT).is_dir()
    report["run_sh_exists"] = (Path(KIT) / "run.sh").is_file()
    report["configs"] = sorted(
        p.name for p in (Path(KIT) / "configs").glob("*.env")
    ) if (Path(KIT) / "configs").is_dir() else "<no configs dir>"
    report["weights_present"] = Path(ESM_WEIGHTS).is_dir()
    return report


@app.function(image=esmfold2_image, volumes=VOLUMES, timeout=TIMEOUT)
def wheels() -> dict:
    """Copy the three compiled wheels onto the weights volume, once.

    Building them takes ~44 minutes, 74% of it flash-attn compiling for sm80 and
    sm90, because the kit pins torch 2.13.0 + CUDA 13.0 and no flash-attn wheel
    is published for that combination. The Dockerfile accepts
    `WHEELS_FROM=prebuilt`, which installs these from `esmfold2/stock/wheels/`
    after checking each name and sha256 against `stock/PINS.json` -- turning
    that step into seconds.

    They ship nowhere, so the only way to obtain them is from a build that
    already paid. Pull them down from the volume into
    `~/ubm/esmfold2/stock/wheels/` and any later rebuild is cheap. Keeping them
    on the weights volume rather than only locally means a fresh machine can
    fetch them without compiling.
    """
    import hashlib
    import shutil

    source = Path(KIT) / "stock" / "wheels"
    target = Path(WEIGHTS) / "esmfold2-wheels"
    target.mkdir(parents=True, exist_ok=True)

    found = sorted(source.glob("*.whl"))
    copied = []
    for wheel in found:
        shutil.copy2(wheel, target / wheel.name)
        digest = hashlib.sha256((target / wheel.name).read_bytes()).hexdigest()
        copied.append(
            {"name": wheel.name, "mb": round(wheel.stat().st_size / 1e6, 1), "sha256": digest}
        )
    weights.commit()

    return {
        "source": str(source),
        "n_wheels": len(copied),
        "wheels": copied,
        "volume_path": str(target),
        "next": (
            "modal volume get rfd3-weights esmfold2-wheels/<name> "
            "~/ubm/esmfold2/stock/wheels/<name>  (then build with "
            "build_args={'STACK': 'img_esmfold2_a100', 'WHEELS_FROM': 'prebuilt'})"
        ),
    }


@app.function(image=esmfold2_image, volumes=VOLUMES, timeout=TIMEOUT)
def install() -> dict:
    """Fetch the ~27 GB of weights. No GPU: this is a download.

    Idempotent by hash-check, so re-running a complete directory is cheap.
    """
    target = Path(ESM_WEIGHTS)
    target.mkdir(parents=True, exist_ok=True)

    code, tail = _stream(["bash", "run.sh", "install", "--weights", ESM_WEIGHTS])
    weights.commit()

    files = [p for p in target.rglob("*") if p.is_file()]
    return {
        "returncode": code,
        "n_files": len(files),
        "total_gb": round(sum(p.stat().st_size for p in files) / 1e9, 2),
        "empty_files": [str(p.relative_to(target)) for p in files if p.stat().st_size == 0][:20],
        "output_tail": tail,
    }


@app.function(image=esmfold2_image, gpu=GPU, volumes=VOLUMES, timeout=TIMEOUT)
def check() -> dict:
    """The kit's own dry run: reads pins, digests the weights, loads no model.

    Exit 0 means the stack, the card and the variant all agree. Exit 3 means a
    mode could not engage, and the reason is on the `NOT ACTIVE` line.
    """
    code, tail = _stream(
        ["bash", "run.sh", "check", "--config", CARD,
         "--variant", "full_nomsa", "--mode", "fast"]
    )
    return {"returncode": code, "exit_3_means_not_active": code == 3, "output_tail": tail}


@app.function(image=esmfold2_image, gpu=GPU, volumes=VOLUMES, timeout=TIMEOUT)
def warm() -> dict:
    """Precompile the Triton kernels into the jit volume.

    Containers are ephemeral, so without a persistent `MODEL_OPT_JIT_ROOT`
    every one of ~640 apo predictions would pay a cold compile.
    """
    Path(ESM_JIT).mkdir(parents=True, exist_ok=True)
    code, tail = _stream(
        ["bash", "run.sh", "warm", "--config", CARD,
         "--variant", "full_nomsa", "--mode", "fast"]
    )
    jit.commit()
    cached = [p for p in Path(ESM_JIT).rglob("*") if p.is_file()]
    return {
        "returncode": code,
        "jit_files": len(cached),
        "jit_mb": round(sum(p.stat().st_size for p in cached) / 1e6, 1),
        "output_tail": tail,
    }


@app.function(image=esmfold2_image, gpu=GPU, volumes=VOLUMES, timeout=TIMEOUT)
def run_one(run_root: str = "motif_scaffold_alphav_beta3/alphav_beta3_01-dryrun") -> dict:
    """Fold one real MPNN sequence apo, from the shards on the data volume.

    Apo means the binder alone with `"msa": null` and `--variant full_nomsa`:
    no MSA is supplied and the variant would not read one if it were.
    """
    import json
    import time

    data.reload()
    shards = Path(DATA) / run_root / "shards"
    records = [
        json.loads(p.read_text())
        for p in sorted(shards.glob("*.json"))
    ]
    sequences = [r for r in records if r.get("seq_id") and r.get("sequence")]
    report: dict[str, object] = {
        "shards_found": len(records),
        "sequence_records": len(sequences),
    }
    if not sequences:
        report["error"] = f"no sequence shards under {shards}"
        return report

    record = sequences[0]
    seq_id, sequence = record["seq_id"], record["sequence"]
    report["seq_id"] = seq_id
    report["seq_length"] = len(sequence)

    work = Path("/tmp/apo")
    work.mkdir(parents=True, exist_ok=True)
    # Upstream's prediction-input JSON: a list of items, each with its own id,
    # chains and seeds. `id` names the output file, so seq_id makes the result
    # directly addressable.
    payload = [
        {
            "id": seq_id,
            "sequences": [
                {"type": "protein", "id": "A", "sequence": sequence, "msa": None}
            ],
            "seeds": [0],
            "num_diffusion_samples": 1,
        }
    ]
    input_json = work / "apo_input.json"
    input_json.write_text(json.dumps(payload, indent=2))
    out_dir = work / "out"

    started = time.monotonic()
    code, tail = _stream(
        ["bash", "run.sh", "pred", "--config", CARD, "--mode", "fast",
         "--variant", "full_nomsa", "--input", str(input_json),
         "--out_dir", str(out_dir), "--seeds", "0", "--det", "1"]
    )
    report["seconds"] = round(time.monotonic() - started, 1)
    report["returncode"] = code
    report["output_tail"] = tail
    if code != 0:
        return report

    produced = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file())
    report["produced"] = produced
    # Expected: cif_all/<id>__full__s<seed>_x<k>.cif plus ..._pae.npz, and an
    # appended pred_rows.jsonl. The token is `full`, not the variant name, so
    # full_nomsa and full_msa are told apart only by out_dir.
    report["cifs"] = [n for n in produced if n.endswith(".cif")]
    report["paes"] = [n for n in produced if n.endswith("_pae.npz")]
    rows = out_dir / "pred_rows.jsonl"
    if rows.is_file():
        report["pred_rows"] = rows.read_text()[:1200]
    return report


@app.local_entrypoint()
def main(stage: str = "env"):
    """`modal run experiments/esmfold2_image.py --stage env|install|check|warm|run_one`."""
    fn = {
        "env": env,
        "wheels": wheels,
        "install": install,
        "check": check,
        "warm": warm,
        "run_one": run_one,
    }[stage]
    for key, value in fn.remote().items():
        text = str(value)
        print(f"\n### {key}\n{text}" if len(text) > 80 else f"{key}: {text}")
