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

VOLUMES = {DATA: data_volume, WEIGHTS: weights_volume, JIT: jit_volume}

# ---------------------------------------------------------------------------
# Images. `add_local_python_source` must come last: no build step may follow it.
# ---------------------------------------------------------------------------
PROJECT_DEPS = ("numpy>=1.26", "biotite>=1.0", "pyyaml>=6.0", "pydantic>=2.0")

# CPU work: spec resolution, backbone filters, the reducer. No model, no GPU.
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*PROJECT_DEPS)
    .add_local_python_source("rfd3_pipeline")
)

# RFD3. rc-foundry pins >=3.12,<3.13 and brings torch with the default CUDA
# wheel, so no index-url juggling is needed.
rfd3_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("rc-foundry[rfd3]==0.2.0")
    .pip_install(*PROJECT_DEPS)
    .env({"FOUNDRY_CHECKPOINT_DIRS": RFD3_CHECKPOINTS, "MODEL_OPT_JIT_ROOT": JIT})
    .add_local_python_source("rfd3_pipeline")
)

HOUR = 60 * 60


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
    import subprocess
    from pathlib import Path

    target = Path(RFD3_CHECKPOINTS)
    target.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["foundry", "install", "rfd3", "--checkpoint-dir", RFD3_CHECKPOINTS],
        capture_output=True,
        text=True,
        timeout=2 * HOUR,
    )
    weights_volume.commit()

    files = sorted(
        (str(p.relative_to(target)), int(p.stat().st_size))
        for p in target.rglob("*")
        if p.is_file()
    )
    return {
        "returncode": int(result.returncode),
        "files": files,
        "stderr_tail": str(result.stderr or "")[-1000:],
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
    import subprocess
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
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=str(root), timeout=6 * HOUR
    )
    data_volume.commit()

    designs = sorted(p.stem for p in out_dir.glob(f"*{spec_name}*.json"))
    return {
        "spec_name": spec_name,
        "returncode": int(result.returncode),
        "n_designs": len(designs),
        "designs": designs,
        "stderr_tail": str(result.stderr or "")[-2000:],
    }


# ---------------------------------------------------------------------------
# Backbone filters
# ---------------------------------------------------------------------------
@app.function(image=cpu_image, cpu=4.0, memory=16384, volumes=VOLUMES, timeout=2 * HOUR)
def filter_designs(design_ids: list[str], campaign_payload: dict) -> dict:
    """Filter a chunk of designs, writing one shard each.

    Chunked rather than one design per container: the per-call overhead of
    starting a container dwarfs the few hundred milliseconds of filtering.
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
    from rfd3_pipeline.report import read_shards, write_summaries

    campaign = _campaign_from(campaign_payload)
    root = Path(_run_root(campaign))
    records = read_shards(root / "shards")

    paths = write_summaries(
        root,
        records,
        campaign=campaign.general.name_design_campaign,
        run_name=campaign.general.run_name,
        n_specs=len(campaign.rfdiffusion3.specs),
        stage_gates={Stage.BACKBONE_FILTERS: backbone_gates(campaign.rfdiffusion3)},
    )
    data_volume.commit()

    passing = sum(1 for r in records if r.get("bb_pass") is True)
    return {
        "n_records": len(records),
        "n_passing_backbone_filters": passing,
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
    campaign = _campaign_from(campaign_payload)
    rfd3 = campaign.rfdiffusion3
    report: dict[str, object] = {"start_at": start_at}

    if start_at == "rfd3":
        spec_names = [spec.name for spec in rfd3.specs]
        rfd3_results = list(
            run_rfd3.map(spec_names, kwargs={"campaign_payload": campaign_payload})
        )
        report["rfd3"] = rfd3_results
        designs = [d for result in rfd3_results for d in result["designs"]]
    else:
        from pathlib import Path

        out_dir = Path(_run_root(campaign)) / "rfd3" / "out"
        designs = sorted(p.stem for p in out_dir.glob("*.json"))
    report["n_designs"] = len(designs)

    chunks = [designs[i : i + chunk_size] for i in range(0, len(designs), chunk_size)]
    report["backbone_filters"] = list(
        filter_designs.map(chunks, kwargs={"campaign_payload": campaign_payload})
    )

    report["reduce"] = reduce_summaries.remote(campaign_payload)
    return report


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------
def _prepare(campaign_path: str) -> tuple[dict, str, str]:
    """Validate and resolve locally, then upload what the run needs.

    Resolution happens here so a bad campaign fails on the client in seconds,
    before a container is started.
    """
    from pathlib import Path

    import yaml

    from rfd3_pipeline.config import CampaignConfig
    from rfd3_pipeline.spec import dump_rfd3_specs, resolve_campaign

    repo = Path(__file__).resolve().parent
    campaign = CampaignConfig.model_validate(yaml.safe_load(Path(campaign_path).read_text()))
    resolved = resolve_campaign(campaign, structure_root=repo)
    for warning in resolved.warnings:
        print(f"WARNING: {warning}")

    general = campaign.general
    remote_root = f"{general.name_design_campaign}/{general.run_name}"
    structure = campaign.rfdiffusion3.path_input_structure

    with data_volume.batch_upload(force=True) as upload:
        upload.put_file(repo / structure, structure)
        upload.put_file(io_bytes(dump_rfd3_specs(resolved)), f"{remote_root}/rfd3/specs.yaml")
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
def run(campaign: str, start_at: str = "rfd3", chunk_size: int = 32):
    payload, _, _ = _prepare(campaign)
    report = orchestrate.remote(payload, start_at=start_at, chunk_size=chunk_size)
    _show(report)


@app.local_entrypoint()
def install_weights():
    _show(populate_weights.remote())


@app.local_entrypoint()
def reduce_only(campaign: str):
    import yaml
    from pathlib import Path

    from rfd3_pipeline.config import CampaignConfig

    payload = CampaignConfig.model_validate(
        yaml.safe_load(Path(campaign).read_text())
    ).model_dump(mode="json")
    _show(reduce_summaries.remote(payload))


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
