"""Probe: can RFD3 be installed and run in a Modal image?

The one component of the pipeline with no prior art in Optimus, so it gets
established before anything is built on top of it. Deliberately staged, cheapest
first, so a failure is diagnosed without having paid for the step after it:

    modal run experiments/rfd3_image.py::check       # CPU, no weights
    modal run experiments/rfd3_image.py::install     # CPU, downloads checkpoint
    modal run experiments/rfd3_image.py::gpu_check   # GPU, loads the model

Facts this encodes, from PyPI metadata and models/rfd3/README.md:
  * rc-foundry requires >=3.12,<3.13 -- not 3.13.
  * the `rfd3` extra adds only pydantic; atomworks[ml] and torch come with the
    base package, and torch is unpinned so the default CUDA wheel applies.
  * `foundry install rfd3 --checkpoint-dir DIR` sets FOUNDRY_CHECKPOINT_DIRS and
    is idempotent, which is what makes the weights volume a one-off step.
"""

from __future__ import annotations

import modal

app = modal.App("rfd3-image-probe")

# One subfolder per model, created only when that model is first needed.
weights = modal.Volume.from_name("rfd3-weights", create_if_missing=True)
WEIGHTS = "/weights"
CHECKPOINTS = f"{WEIGHTS}/rfd3"

rfd3_image = (
    modal.Image.debian_slim(python_version="3.12")
    # git for any VCS-sourced dependency; build-essential because atomworks'
    # dependency tree still compiles a few things from sdist on 3.12.
    .apt_install("git", "build-essential")
    .pip_install("rc-foundry[rfd3]==0.2.0")
    .env({"FOUNDRY_CHECKPOINT_DIRS": CHECKPOINTS})
)

TIMEOUT = 30 * 60


@app.function(image=rfd3_image, timeout=TIMEOUT)
def _probe_install() -> dict:
    """Does the package import, and is the CLI on PATH? No weights, no GPU."""
    import shutil
    import subprocess
    import sys

    report: dict[str, object] = {"python": sys.version.split()[0]}

    import torch

    # str() on every value: torch.__version__ is a TorchVersion, a str subclass,
    # so returning it directly makes the result unpicklable anywhere torch is not
    # installed -- which includes the local side of every `modal run`. Nothing
    # crossing a Modal boundary should be a library type.
    report["torch"] = str(torch.__version__)
    report["torch_cuda_build"] = str(torch.version.cuda)

    import rfd3

    report["rfd3"] = str(getattr(rfd3, "__version__", "(no __version__)"))

    import atomworks

    report["atomworks"] = str(getattr(atomworks, "__version__", "(no __version__)"))

    for name in ("rfd3", "foundry"):
        report[f"{name}_on_path"] = str(shutil.which(name) or "MISSING")

    # Hydra prints its config groups without loading a checkpoint, which is the
    # cheapest proof that the inference entry point is wired up.
    result = subprocess.run(
        ["rfd3", "design", "--help"], capture_output=True, text=True, timeout=300
    )
    report["rfd3_help_returncode"] = int(result.returncode)
    report["rfd3_help_head"] = str(result.stdout or result.stderr)[:600]
    return report


@app.function(
    image=rfd3_image, volumes={WEIGHTS: weights}, timeout=TIMEOUT
)
def _install_checkpoint() -> dict:
    """Populate the weights volume. Idempotent: re-runs hash-check rather than refetch."""
    import subprocess
    from pathlib import Path

    target = Path(CHECKPOINTS)
    target.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["foundry", "install", "rfd3", "--checkpoint-dir", CHECKPOINTS],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )
    weights.commit()

    files = sorted(
        (str(p.relative_to(target)), p.stat().st_size)
        for p in target.rglob("*")
        if p.is_file()
    )
    return {
        "returncode": result.returncode,
        "stdout_tail": (result.stdout or "")[-1500:],
        "stderr_tail": (result.stderr or "")[-1500:],
        "files": files,
        "total_bytes": sum(size for _, size in files),
    }


@app.function(
    image=rfd3_image, gpu="A100-80GB", volumes={WEIGHTS: weights}, timeout=TIMEOUT
)
def _probe_gpu() -> dict:
    """Confirm the GPU is visible and the checkpoint is readable from the volume."""
    from pathlib import Path

    import torch

    available = bool(torch.cuda.is_available())
    return {
        "cuda_available": available,
        "device": str(torch.cuda.get_device_name(0)) if available else "none",
        "checkpoint_files": sorted(
            str(p.relative_to(CHECKPOINTS)) for p in Path(CHECKPOINTS).rglob("*") if p.is_file()
        ),
    }


def _show(title: str, report: dict) -> None:
    print(f"\n=== {title} ===")
    for key, value in report.items():
        if isinstance(value, list):
            print(f"{key}:")
            for item in value[:20]:
                print(f"    {item}")
        elif isinstance(value, str) and "\n" in value:
            print(f"{key}:")
            for line in value.splitlines()[:20]:
                print(f"    {line}")
        else:
            print(f"{key}: {value}")


@app.local_entrypoint()
def check():
    _show("install probe", _probe_install.remote())


@app.local_entrypoint()
def install():
    _show("checkpoint install", _install_checkpoint.remote())


@app.local_entrypoint()
def gpu_check():
    _show("gpu probe", _probe_gpu.remote())
