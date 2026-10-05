"""Probe: validate the emitted RFD3 spec on CPU, before any GPU is allocated.

`DesignInputSpecification.build()` parses the input structure, applies the
contig, resolves `unindex` / `select_fixed_atoms`, and appends the ligand -- all
before a checkpoint is loaded. So every failure mode of our emitted spec except
diffusion itself can be caught on a CPU container:

  * field names and types our writer got wrong
  * contig or unindex that RFD3 rejects
  * `ligand: "E703"` failing to resolve, or needing the CCD mirror atomworks
    warned about
  * the input path not resolving inside the container

    modal run experiments/rfd3_validate.py::validate

This exists because the first design attempt crash-looped an A100 on a module
import. Anything checkable without a GPU gets checked without a GPU.
"""

from __future__ import annotations

from pathlib import Path

import modal

# Nothing outside the container's own installed packages at module level: Modal
# imports this file inside the container too, so a top-level project import
# crashes every container before its function body runs.

REPO = Path(__file__).resolve().parents[1]
INPUTS_DIR = "inputs_motif_scaffold_alphav_beta3"
CAMPAIGN = REPO / INPUTS_DIR / "alphav_rgd_unindex_claude.yaml"

app = modal.App("rfd3-validate-probe")

rfd3_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("rc-foundry[rfd3]==0.2.0")
    .add_local_dir(str(REPO / INPUTS_DIR), f"/work/{INPUTS_DIR}")
)


@app.function(image=rfd3_image, cpu=4.0, memory=16384, timeout=30 * 60)
def _validate(specs_yaml: str) -> dict:
    """Build every spec key against the real structure. CPU only."""
    import os
    import traceback

    import yaml

    os.chdir("/work")
    Path("/work/specs.yaml").write_text(specs_yaml)
    specs = yaml.safe_load(specs_yaml)

    from rfd3.inference.input_parsing import DesignInputSpecification

    results: dict[str, dict] = {}
    for name, entry in specs.items():
        outcome: dict[str, object] = {}
        try:
            spec = DesignInputSpecification.safe_init(**entry)
            outcome["constructed"] = True
            outcome["sampled_length"] = str(getattr(spec, "length", None))

            built = spec.build(return_metadata=True)
            atoms, metadata = built if isinstance(built, tuple) else (built, None)

            chains = sorted({str(c) for c in atoms.chain_id})
            residues_per_chain = {
                chain: int(len({int(r) for r in atoms.res_id[atoms.chain_id == chain]}))
                for chain in chains
            }
            outcome["built"] = True
            outcome["n_atoms"] = int(atoms.array_length())
            outcome["output_chains"] = chains
            outcome["residues_per_chain"] = residues_per_chain
            outcome["residue_names_present"] = sorted(
                {str(n) for n in atoms.res_name}
            )[:40]
            outcome["has_MN"] = bool((atoms.res_name == "MN").any())

            if metadata is not None:
                outcome["metadata_type"] = str(type(metadata).__name__)
                outcome["metadata_keys"] = (
                    sorted(str(k) for k in metadata) if hasattr(metadata, "keys") else "(not a mapping)"
                )
                # Anything the build already knows is worth seeing -- the sampled
                # contig is decided here, not during diffusion.
                if hasattr(metadata, "get"):
                    for key in ("sampled_contig", "diffused_index_map", "extra"):
                        if metadata.get(key) is not None:
                            outcome[f"metadata.{key}"] = str(metadata[key])[:2000]
        except Exception as error:
            outcome["error_type"] = type(error).__name__
            outcome["error"] = str(error)[:2000]
            outcome["traceback_tail"] = traceback.format_exc()[-2500:]
        results[str(name)] = outcome

    return results


@app.local_entrypoint()
def validate():
    import yaml

    from rfd3_pipeline.config import CampaignConfig
    from rfd3_pipeline.spec import dump_rfd3_specs, resolve_campaign

    campaign = CampaignConfig.model_validate(yaml.safe_load(CAMPAIGN.read_text()))
    resolved = resolve_campaign(campaign, structure_root=REPO)
    for warning in resolved.warnings:
        print(f"WARNING: {warning}")

    specs_yaml = dump_rfd3_specs(resolved)
    print(f"--- specs.yaml ---\n{specs_yaml}")
    for spec in resolved.specs:
        print(f"derived length for {spec.name}: {spec.length_string}")

    for name, outcome in _validate.remote(specs_yaml).items():
        print(f"\n=== {name} ===")
        for key, value in outcome.items():
            if isinstance(value, str) and "\n" in value:
                print(f"{key}:")
                for line in value.splitlines()[-25:]:
                    print(f"    {line}")
            else:
                print(f"{key}: {value}")
