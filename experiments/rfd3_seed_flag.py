"""Probe: what does RFD3 call its seed, and where does it live?

Two runs of the pipeline with byte-identical config produced different
backbones, and `run_rfd3` passed no seed, so RFD3 samples unseeded. That makes
a campaign unrepeatable and makes a controlled sampler comparison impossible:
`step_scale=3` against the stock value cannot be attributed if the backbones
differ between runs too.

RFD3 is Hydra, so the seed is a config key rather than a flag -- `--help` shows
nothing for it. Two findings so far, each correcting the previous guess:

* `seed: null` appears in `configs/inference_engine/base.yaml`, which is why
  unseeded is the default.
* But overriding `inference_engine.seed` is **refused** ("Could not override
  ... To append use +inference_engine.seed="). `configs/inference.yaml` is
  `@package _global_` and selects `inference_engine: rfdiffusion3`, so the
  group's keys may merge at the top level instead.

Appending with `+` would be worse than failing: it would set a key nothing
reads, leaving sampling unseeded while looking configured. So this composes the
config exactly as `rfd3/cli.py` does and reports where `seed` actually sits.

CPU only; the image is already built.
"""

from __future__ import annotations

import modal

app = modal.App("rfd3-seed-probe")

weights = modal.Volume.from_name("rfd3-weights", create_if_missing=True)
WEIGHTS = "/weights"
RFD3_CHECKPOINTS = f"{WEIGHTS}/rfd3"

PROJECT_DEPS = ("numpy>=1.26", "biotite>=1.0", "pyyaml>=6.0", "pydantic>=2.0")

rfd3_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("rc-foundry[rfd3]==0.2.0")
    .pip_install(*PROJECT_DEPS)
    .env({"FOUNDRY_CHECKPOINT_DIRS": RFD3_CHECKPOINTS, "PYTHONUNBUFFERED": "1"})
)


@app.function(image=rfd3_image, volumes={WEIGHTS: weights}, timeout=20 * 60)
def seed_keys() -> dict:
    """Read the config tree in-process; no need to shell out from inside it."""
    import pathlib

    import rfd3

    root = pathlib.Path(rfd3.__file__).parent / "configs"
    report: dict[str, object] = {"configs_root": str(root)}

    engine_dir = root / "inference_engine"
    report["inference_engine_files"] = sorted(p.name for p in engine_dir.glob("*.yaml"))

    # The two that matter: the option the CLI selects, and the base it may or
    # may not inherit from.
    for name in ("rfdiffusion3.yaml", "base.yaml"):
        path = engine_dir / name
        report["engine_" + name] = path.read_text()[:2000] if path.is_file() else "<absent>"

    # Every `seed` in the tree, quoted so the indentation is visible -- nesting
    # is the whole question.
    hits = []
    for path in sorted(root.rglob("*.yaml")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if "seed" in line.lower():
                hits.append(str(path.relative_to(root)) + ":" + str(number) + ": " + repr(line))
    report["seed_lines"] = hits or "<none>"

    # What the CLI actually composes, which is the only authority on the key's
    # path. Mirrors `rfd3/cli.py`.
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        with initialize_config_dir(config_dir=str(root), version_base="1.3"):
            cfg = compose(
                config_name="inference", overrides=["inference_engine=rfdiffusion3"]
            )
        flat = OmegaConf.to_container(cfg, resolve=False)

        def walk(node, prefix=""):
            if isinstance(node, dict):
                for key, value in node.items():
                    child = prefix + "." + str(key) if prefix else str(key)
                    yield from walk(value, child)
            else:
                yield prefix, node

        report["composed_seed_keys"] = [
            path + " = " + repr(value)
            for path, value in walk(flat)
            if "seed" in path.lower()
        ] or "<none>"
        report["composed_top_level_keys"] = sorted(flat) if isinstance(flat, dict) else "?"
    except Exception as error:
        report["compose_error"] = type(error).__name__ + ": " + str(error)

    return report


@app.local_entrypoint()
def main():
    for key, value in seed_keys.remote().items():
        text = str(value)
        print("\n### " + key + "\n" + text if len(text) > 80 else key + ": " + text)
