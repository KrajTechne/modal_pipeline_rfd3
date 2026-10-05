"""Probe: generate one real RFD3 design for the alphaV/beta3 campaign.

The point is not the design. It is the output metadata JSON, which is the only
thing that can confirm or refute what `rfd3_pipeline.spec.design` assumes about
RFD3's output layout -- those readers were written from documentation and have
never seen a real file.

    modal run experiments/rfd3_design.py::design

Deliberately tiny: one spec key, one batch, two diffusion samples. Enough to
prove the spec file parses, the ligand resolves, the output chains come back
A/B/C, and to read a real `diffused_index_map`.
"""

from __future__ import annotations

from pathlib import Path

import modal

# NOTE: nothing that is not installed in the container may be imported at module
# level. Modal imports this file inside the container as well as locally, so a
# top-level `from rfd3_pipeline...` crashes every container at import time --
# before the function body runs, and on a GPU function that means paying for an
# A100 to fail on an import. rfd3_pipeline and yaml are imported inside the
# local entrypoint, which only ever runs on the client.

REPO = Path(__file__).resolve().parents[1]
INPUTS_DIR = "inputs_motif_scaffold_alphav_beta3"
CAMPAIGN = REPO / INPUTS_DIR / "alphav_rgd_unindex_claude.yaml"

app = modal.App("rfd3-design-probe")

weights = modal.Volume.from_name("rfd3-weights", create_if_missing=True)
data = modal.Volume.from_name("rfd3-pipeline", create_if_missing=True)
WEIGHTS, DATA = "/weights", "/data"
CHECKPOINTS = f"{WEIGHTS}/rfd3"

rfd3_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("rc-foundry[rfd3]==0.2.0")
    .env({"FOUNDRY_CHECKPOINT_DIRS": CHECKPOINTS})
    # The input structure, at the same relative path the spec file names, so
    # `input:` resolves without rewriting it for the container.
    .add_local_dir(str(REPO / INPUTS_DIR), f"/work/{INPUTS_DIR}")
)


@app.function(
    image=rfd3_image,
    gpu="A100-80GB",
    volumes={WEIGHTS: weights, DATA: data},
    timeout=60 * 60,
)
def _run_design(specs_yaml: str, spec_name: str, out_dir: str) -> dict:
    """One RFD3 design, returning the output listing and metadata verbatim."""
    import subprocess

    work = Path("/work")
    spec_path = work / "specs.yaml"
    spec_path.write_text(specs_yaml)

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    command = [
        "rfd3",
        "design",
        f"out_dir={output}",
        f"inputs={spec_path}",
        f"json_keys_subset=[{spec_name}]",
        "n_batches=1",
        "diffusion_batch_size=2",
        "inference_sampler.num_timesteps=200",
        "inference_sampler.step_scale=1.5",
        "inference_sampler.gamma_0=0.6",
        "prevalidate_inputs=True",
        "skip_existing=False",
        "dump_prediction_metadata_json=True",
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=work, timeout=55 * 60
    )
    data.commit()

    produced = sorted(p for p in output.rglob("*") if p.is_file())
    metadata = {
        str(p.relative_to(output)): p.read_text()[:20000]
        for p in produced
        if p.suffix == ".json"
    }

    return {
        "command": " ".join(command),
        "returncode": int(result.returncode),
        "stdout_tail": str(result.stdout or "")[-4000:],
        "stderr_tail": str(result.stderr or "")[-4000:],
        "files": [
            (str(p.relative_to(output)), int(p.stat().st_size)) for p in produced
        ],
        "metadata_json": metadata,
    }


@app.local_entrypoint()
def design(spec_name: str = "rgd_core"):
    import yaml

    from rfd3_pipeline.config import CampaignConfig
    from rfd3_pipeline.spec import dump_rfd3_specs, resolve_campaign

    campaign = CampaignConfig.model_validate(yaml.safe_load(CAMPAIGN.read_text()))
    resolved = resolve_campaign(campaign, structure_root=REPO)
    for warning in resolved.warnings:
        print(f"WARNING: {warning}")

    specs_yaml = dump_rfd3_specs(resolved)
    print(f"--- specs.yaml ({spec_name}) ---\n{specs_yaml}")

    report = _run_design.remote(specs_yaml, spec_name, f"{DATA}/probe/{spec_name}")

    print(f"\n=== returncode {report['returncode']} ===")
    print(f"\n--- stdout tail ---\n{report['stdout_tail']}")
    if report["returncode"] != 0:
        print(f"\n--- stderr tail ---\n{report['stderr_tail']}")

    print("\n--- files produced ---")
    for name, size in report["files"]:
        print(f"  {size:>12,}  {name}")

    local = REPO / "experiments" / "rfd3_output"
    local.mkdir(parents=True, exist_ok=True)
    for name, text in report["metadata_json"].items():
        destination = local / Path(name).name
        destination.write_text(text)
        print(f"\n--- {name} (saved to {destination}) ---")
        print(text[:3000])
