"""Probe: can the af3_jax kit run on Modal? For cross-validation.

Staged cheapest-first, the discipline that paid off on RFD3 and ESMFold2 and
cost six attempts when skipped:

    modal run experiments/af3_jax_image.py --stage env      # GPU, no weights
    modal run experiments/af3_jax_image.py --stage install  # ~2.3 GB fetch+convert
    modal run experiments/af3_jax_image.py --stage check    # kit dry run
    modal run experiments/af3_jax_image.py --stage warm     # JIT precompile

`run_one` is deliberately absent for now. The fork takes an `--input_dir` of
AF3 fold-input JSONs rather than one list, and it is the only model in
`MODEL_CAPABILITIES` with `msa_paired=True`, so its input assembly is a
separate piece of work from ESMFold2's. Attempting a prediction before that is
understood risks `fill_missing_msas` reaching api.colabfold.com, which is the
one thing this pipeline must not do by accident.

**This checkout is `~/ubm_af3`, separate from ESMFold2's `~/ubm`.** One shared
checkout was tried and rejected: `context_dir` must be the parent of
`common/opt_core`, so it cannot be narrowed per kit, and Modal hashes the
`ignore` spec itself rather than the post-ignore file set -- verified
2026-10-09 on a throwaway image, where two builds with identical contexts and
`ignore` naming only absent paths produced different image ids. Consolidating
would therefore cost a 38-minute ESMFold2 rebuild to save 306 MB. Prepare with:

    git clone --depth 1 --filter=blob:none --sparse -c core.autocrlf=false \\
        https://github.com/anthropics/uplifting-biomolecular-modeling.git ~/ubm_af3
    cd ~/ubm_af3 && git sparse-checkout set af3_jax common
    mkdir -p _jitcache
    tar -cf _jitcache/af3_jax-empty-jit.tar --files-from /dev/null

Both of those last two steps are load-bearing, and both are the traps ESMFold2
taught us:

* `core.autocrlf=false` -- git on Windows rewrites LF to CRLF, which goes into
  a Linux image and breaks `run.sh` with `set: pipefail: invalid option name`.
* The empty jitcache tar -- the Dockerfile's last COPY is
  `_jitcache/af3_jax-*-jit.ta[r]` (line 101), Docker's idiom for an optional
  source. Modal fails the build when the glob matches nothing.

What this kit does differently from ESMFold2, read off its Dockerfile and
README rather than assumed:

* **No CUDA extensions from source.** It installs from `upstream.lock` with
  `uv` into a venv, and compiles only the stock alphafold3 wheel. ESMFold2's
  ~38 minutes was flash-attn and TransformerEngine; neither is here.
* **Weights are ~2.3 GB, fetched *and converted*** -- `AF3_JAX_PARAMS_ROOT`
  must name a directory holding `p2/of3_ported_weights.bin.zst`.
* **`WHEELS_FROM=prebuilt` accepts a wheel you built yourself**, via
  `--build-arg STOCK_WHEEL_SHA256=<sha256>`. ESMFold2's equivalent only
  compared against `PINS.json`, so banking self-built wheels was impossible
  there; here it is the supported way to make a rebuild cheap.
* **`--variant p2` is required** on `pred`, `check` and `warm`, and it names
  the *weights* rather than encoding MSA presence -- which is why its
  capability row has `fixed_variant="p2"` rather than `derived_variants`.

Open questions this is here to answer:

* Does the host driver suit it? ESMFold2 needed >= 580 and Modal reports
  580.95.05, a margin of 0.95. This kit's requirement is unread.
* Does the image build, and does JAX see the GPU?
* **What does `install` actually land?** The user has approval for native AF3
  weights, but the kit is built around the ported `p2` checkpoint, so
  `weights: af3_native` may need its own route rather than a config flip. The
  install report should say what appears on the volume.
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("af3-jax-image-probe")

weights = modal.Volume.from_name("rfd3-weights", create_if_missing=True)
jit = modal.Volume.from_name("rfd3-jit", create_if_missing=True)

WEIGHTS = "/weights"
JIT = "/jit"
AF3_WEIGHTS = f"{WEIGHTS}/af3"
AF3_JIT = f"{JIT}/af3_jax"
KIT = "/kit/af3_jax"
CARD = "a100"
VOLUMES = {WEIGHTS: weights, JIT: jit}

CONTEXT = Path.home() / "ubm_af3"

af3_image = (
    modal.Image.from_dockerfile(
        CONTEXT / "af3_jax" / "environment" / "Dockerfile",
        context_dir=CONTEXT,
        build_args={
            # Capped rather than left at "every core". ESMFold2's analogous
            # default over-subscribed memory and got a build OOM-killed on a
            # smaller builder, making success depend on which machine Modal
            # happened to allocate. This wheel is far lighter, but the cap
            # costs nothing and removes the coin flip.
            "BUILD_JOBS": "8",
        },
    )
    .env(
        {
            # Required, and it names the directory holding p2/*.bin.zst.
            "AF3_JAX_PARAMS_ROOT": AF3_WEIGHTS,
            # Containers are ephemeral, so without a persistent root every
            # prediction pays a cold compile.
            "MODEL_OPT_JIT_ROOT": AF3_JIT,
            "PYTHONUNBUFFERED": "1",
        }
    )
)

TIMEOUT = 3 * 60 * 60
GPU = "A100-40GB"


def _stream(command: list[str], *, cwd: str = KIT, tail_lines: int = 80):
    """Run a command, echoing output as it arrives, returning (code, tail).

    Same reasoning as `app.py`'s `_stream`: a multi-gigabyte fetch or a JIT
    compile behind `capture_output=True` is indistinguishable from a hang.
    """
    import subprocess
    from collections import deque

    tail: deque[str] = deque(maxlen=tail_lines)
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, cwd=cwd,
    )
    for line in process.stdout or ():
        line = line.rstrip()
        tail.append(line)
        print(line, flush=True)
    if process.stdout is not None:
        process.stdout.close()
    return process.wait(), "\n".join(tail)


@app.function(image=af3_image, gpu=GPU, volumes=VOLUMES, timeout=TIMEOUT)
def env() -> dict:
    """Cheapest meaningful stage: does the host and image satisfy the kit?

    No weights touched. The driver check comes first, since on ESMFold2 that
    was the one thing a successful build could not tell us.
    """
    import subprocess

    report: dict[str, object] = {}

    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    report["nvidia_smi"] = (smi.stdout or smi.stderr).strip()

    report["kit_dir_exists"] = Path(KIT).is_dir()
    report["run_sh_exists"] = (Path(KIT) / "run.sh").is_file()
    configs = Path(KIT) / "configs"
    report["configs"] = (
        sorted(p.name for p in configs.glob("*.env")) if configs.is_dir() else "<none>"
    )
    report["weights_present"] = Path(AF3_WEIGHTS).is_dir()

    # The kit installs into its own venv (`/kit/venv` per the Dockerfile
    # header), so the interpreter that matters is not necessarily `python`.
    for candidate in ("/kit/venv/bin/python", "/alphafold3_venv/bin/python3", "python"):
        if not Path(candidate).is_file():
            continue
        probe = subprocess.run(
            [candidate, "-c",
             "import jax; print('jax', jax.__version__); "
             "print('devices', jax.devices())"],
            capture_output=True, text=True,
        )
        report[f"jax_via_{candidate}"] = (
            (probe.stdout or "").strip() or probe.stderr[-600:]
        )

    return report


@app.function(image=af3_image, volumes=VOLUMES, timeout=TIMEOUT)
def install() -> dict:
    """Fetch and convert the checkpoint. No GPU: this is a download plus a
    conversion.

    Reports exactly what lands, because the native-weights question turns on
    it: the kit is built around the ported `p2` file, and the user's approval
    for native AF3 weights may need a different route entirely.
    """
    target = Path(AF3_WEIGHTS)
    target.mkdir(parents=True, exist_ok=True)

    code, tail = _stream(
        ["bash", "run.sh", "install", "--config", CARD, "--weights", AF3_WEIGHTS]
    )
    weights.commit()

    files = [p for p in target.rglob("*") if p.is_file()]
    return {
        "returncode": code,
        "n_files": len(files),
        "total_gb": round(sum(p.stat().st_size for p in files) / 1e9, 2),
        "files": sorted(str(p.relative_to(target)) for p in files)[:30],
        # `wget`-style failures leave zero-byte files rather than erroring.
        "empty_files": [
            str(p.relative_to(target)) for p in files if p.stat().st_size == 0
        ][:20],
        "output_tail": tail,
    }


@app.function(image=af3_image, gpu=GPU, volumes=VOLUMES, timeout=TIMEOUT)
def check() -> dict:
    """The kit's own dry run: resolves the mode on this GPU, loads no model.

    Exit 3 means a mode could not engage and the reason is on the NOT ACTIVE
    line; it never falls back to stock silently.
    """
    code, tail = _stream(
        ["bash", "run.sh", "check", "--config", CARD, "--variant", "p2", "--mode", "fast"]
    )
    return {"returncode": code, "exit_3_means_not_active": code == 3, "output_tail": tail}


@app.function(image=af3_image, gpu=GPU, volumes=VOLUMES, timeout=TIMEOUT)
def warm() -> dict:
    """Precompile into the jit volume.

    Note the kit's `warm` takes `--input_dir`, unlike ESMFold2's, so this uses
    whatever example inputs the kit ships; the README's `X=` is in-image.
    """
    Path(AF3_JIT).mkdir(parents=True, exist_ok=True)
    # The README's `X=`: barnase-barstar, 5 seeds, MSAs inline. A directory of
    # JSONs, so point at the directory itself -- an earlier attempt passed its
    # `.parent`, which holds no JSONs, and the kit warmed nothing while
    # reporting `rc=0` internally and exiting 3. `WARM PARTIAL ...
    # executables=0 predictions=0` is the line that says so.
    inputs = Path(KIT) / "opt" / "forward" / "fast_inference" / "tests" / "inputs"
    report: dict[str, object] = {
        "input_dir": str(inputs),
        "input_files": sorted(p.name for p in inputs.glob("*.json")) if inputs.is_dir() else "<absent>",
    }
    if not inputs.is_dir() or not any(inputs.glob("*.json")):
        report["error"] = f"no AF3 input JSONs under {inputs}; warm needs them to trace"
        return report

    code, tail = _stream(
        ["bash", "run.sh", "warm", "--config", CARD, "--variant", "p2",
         "--mode", "fast", "--input_dir", str(inputs)]
    )
    jit.commit()
    cached = [p for p in Path(AF3_JIT).rglob("*") if p.is_file()]
    report.update(
        returncode=code,
        jit_files=len(cached),
        jit_mb=round(sum(p.stat().st_size for p in cached) / 1e6, 1),
        output_tail=tail,
    )
    return report


@app.local_entrypoint()
def main(stage: str = "env"):
    """`modal run experiments/af3_jax_image.py --stage env|install|check|warm`."""
    fn = {"env": env, "install": install, "check": check, "warm": warm}[stage]
    for key, value in fn.remote().items():
        text = str(value)
        print("\n### " + key + "\n" + text if len(text) > 80 else key + ": " + text)
