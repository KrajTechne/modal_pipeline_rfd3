"""Probe: can the vendored LigandMPNN run in a Modal image?

Staged, cheapest first, so a failure is diagnosed without having paid for the
step after it:

    modal run experiments/mpnn_image.py --stage check      # CPU, no weights
    modal run experiments/mpnn_image.py --stage install    # CPU, fetches params
    modal run experiments/mpnn_image.py --stage run_one    # GPU, one real design

Why the checkout is vendored at `LigandMPNN/` rather than cloned in the build:
LigandMPNN's bundled openfold used three numpy aliases removed in 1.24
(`np.int` x3, `np.object` x2, `np.bool` x1). Upstream therefore pins
numpy==1.23.5, which has no cp312 wheels, which would have forced this image to
Python 3.11 and a container of its own. With those six occurrences fixed in the
vendored copy, MPNN runs on 3.12 with modern numpy alongside the rest of the
pipeline. Verified 2026-10-07: those three files are the *only* difference from
upstream `26ec57a`; `run.py` and friends are untouched.

So `requirements.txt` is deliberately **not** installed -- it would drag back
numpy 1.23.5 and pin torch to 2.2.1. Installed instead is what `run.py`
actually imports.

Open questions this probe answers, in order:

  * Do the imports hold on 3.12 with modern numpy? (`check`)
  * What does `get_model_params.sh` put where, and how big? 15 checkpoints, of
    which `solublempnn_v_48_020.pt` is the one `mpnn_type: soluble` needs.
    (`install`)
  * **Does run.py accept what RFD3 produces?** RFD3 writes gzipped mmCIF and
    the flag is `--pdb_path`. Also open: whether its residues arrive as `UNK`,
    which `wrapper.py` rewrites to `GLY` before running. (`run_one`)
  * How long does one design take on a GPU, which is what sizes
    `max_concurrent` later. (`run_one`)

Nothing here writes to the pipeline's data volume.
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("mpnn-image-probe")

weights = modal.Volume.from_name("rfd3-weights", create_if_missing=True)
data = modal.Volume.from_name("rfd3-pipeline", create_if_missing=True)

WEIGHTS = "/weights"
PARAMS = f"{WEIGHTS}/ligandmpnn"
DATA = "/data"
CHECKOUT = "/opt/LigandMPNN"

VENDORED = Path(__file__).resolve().parent.parent / "LigandMPNN"

# What `run.py` imports: `from prody import *` (data_utils.py:6), then sc_utils
# -> the bundled openfold. numpy/scipy/biopython come along for the ride.
MPNN_DEPS = ("prody", "ml-collections", "dm-tree", "scipy", "biopython", "numpy>=1.26")

mpnn_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget")
    # Default PyPI torch, i.e. the CUDA build. MPNN runs on a GPU
    # (`MpnnConfig.gpu`, L40S by default; A10G is also fine -- MPNN is not a
    # kit, so no `--config` card is involved).
    .pip_install("torch")
    .pip_install(*MPNN_DEPS)
    # copy=True so the tree lands in an image layer: a copy=False local source
    # may not be followed by further build steps, and it keeps the probe
    # reproducible rather than re-uploading on every run.
    .add_local_dir(VENDORED, CHECKOUT, copy=True, ignore=["__pycache__", "*.pyc"])
)

TIMEOUT = 30 * 60


@app.function(image=mpnn_image, timeout=TIMEOUT)
def check() -> dict:
    """Cheapest stage: do the imports hold, and what flags does run.py take?

    No weights, no GPU. The point is to confirm the alias fixes are sufficient
    on modern numpy -- previous attempts never reached argparse.
    """
    import subprocess

    report: dict[str, object] = {}

    versions = subprocess.run(
        [
            "python",
            "-c",
            "import numpy, torch, prody, scipy; "
            "print(numpy.__version__, torch.__version__, prody.__version__, scipy.__version__)",
        ],
        capture_output=True,
        text=True,
    )
    report["versions_numpy_torch_prody_scipy"] = (
        versions.stdout.strip() or versions.stderr[-500:]
    )

    # The import that used to fail: run.py -> sc_utils -> openfold -> np.int.
    openfold = subprocess.run(
        ["python", "-c", "import openfold.np.residue_constants as rc; print('openfold ok')"],
        cwd=CHECKOUT,
        capture_output=True,
        text=True,
    )
    report["openfold_import"] = (openfold.stdout.strip() or openfold.stderr[-800:])

    helped = subprocess.run(
        ["python", "run.py", "--help"], cwd=CHECKOUT, capture_output=True, text=True
    )
    report["help_returncode"] = helped.returncode
    # Just the flag names; the full help was already read on 2026-10-07.
    import re

    report["flags"] = sorted(
        set(re.findall(r"--[a-z_]+", helped.stdout))
    ) if helped.returncode == 0 else (helped.stderr or helped.stdout)[-1500:]

    return report


@app.function(image=mpnn_image, volumes={WEIGHTS: weights}, timeout=TIMEOUT)
def install() -> dict:
    """Fetch the model params into the weights volume and report what landed."""
    import subprocess
    from pathlib import Path as P

    target = P(PARAMS)
    target.mkdir(parents=True, exist_ok=True)

    # The script takes its destination as $1 (verified by reading it), so this
    # is not relying on a hardcoded ./model_params.
    result = subprocess.run(
        ["bash", f"{CHECKOUT}/get_model_params.sh", str(target)],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )
    weights.commit()

    landed = sorted(
        (str(p.relative_to(target)), p.stat().st_size)
        for p in target.rglob("*")
        if p.is_file()
    )
    return {
        "returncode": result.returncode,
        "n_files": len(landed),
        "total_mb": round(sum(size for _, size in landed) / 1e6, 1),
        "files": landed,
        # The one `mpnn_type: soluble` needs. Absent means the step is a no-op
        # for this campaign however many other checkpoints arrived.
        "has_soluble_v_48_020": any(
            name == "solublempnn_v_48_020.pt" for name, _ in landed
        ),
        "empty_files": [name for name, size in landed if size == 0],
        "stderr_tail": (result.stderr or "")[-1500:],
    }


@app.function(
    image=mpnn_image,
    gpu="L40S",
    volumes={WEIGHTS: weights, DATA: data},
    memory=16384,
    timeout=TIMEOUT,
)
def run_one(
    design: str = (
        "motif_scaffold_alphav_beta3/alphav_beta3_01-dryrun/rfd3/out/"
        "rgd_core_0_model_0.cif.gz"
    ),
    fixed_residues: str = "A49 A50 A51",
    chains_to_design: str = "A",
) -> dict:
    """One real design, with the real derived `fixed_residues`.

    Defaults are the design from the 2026-10-07 dry run and the
    `mpnn_fixed_residues` its shard recorded (`A49 A50 A51`), so this exercises
    the actual output numbering rather than a toy case. If MPNN silently used
    input numbering the motif would be redesigned, which is the failure this is
    here to rule out.
    """
    import gzip
    import shutil
    import subprocess
    import time
    from pathlib import Path as P

    source = P(DATA) / design
    work = P("/tmp/mpnn")
    work.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"input": str(source), "exists": source.exists()}
    if not source.exists():
        return report

    checkpoint = P(PARAMS) / "solublempnn_v_48_020.pt"
    report["checkpoint_present"] = checkpoint.exists()
    if not checkpoint.exists():
        report["error"] = "run `--stage install` first"
        return report

    # ProDy is MPNN's own parser, so converting with it answers whether MPNN
    # can read what RFD3 wrote, rather than introducing a third opinion.
    import prody

    plain_cif = work / "input.cif"
    with gzip.open(source, "rb") as src, plain_cif.open("wb") as dst:
        shutil.copyfileobj(src, dst)

    parsed = prody.parseMMCIF(str(plain_cif))
    if parsed is None:
        report["error"] = "prody.parseMMCIF returned None"
        return report
    report["atoms"] = int(parsed.numAtoms())
    report["chains"] = sorted({str(c) for c in parsed.getChids()})
    residue_names = {str(n) for n in parsed.getResnames()}
    # Does RFD3 hand back UNK? wrapper.py rewrites UNK->GLY, which only matters
    # if UNK actually appears.
    report["has_UNK"] = "UNK" in residue_names
    report["n_residue_names"] = len(residue_names)

    as_pdb = work / "input.pdb"
    prody.writePDB(str(as_pdb), parsed)
    report["pdb_bytes"] = as_pdb.stat().st_size

    out_folder = work / "out"
    command = [
        "python", "run.py",
        "--model_type", "soluble_mpnn",
        "--checkpoint_soluble_mpnn", str(checkpoint),
        "--pdb_path", str(as_pdb),
        "--out_folder", str(out_folder),
        # Without this, MPNN designs EVERY chain, inventing new target
        # sequences and optimising the binder against them. Verified
        # 2026-10-07: omitting it rewrote chains B and C. The targets must be
        # context, not substrate.
        "--chains_to_design", chains_to_design,
        "--fixed_residues", fixed_residues,
        "--temperature", "0.1",
        "--batch_size", "8",
        "--omit_AA", "C",
        "--seed", "1",
    ]
    report["command"] = " ".join(command)

    started = time.monotonic()
    result = subprocess.run(
        command, cwd=CHECKOUT, capture_output=True, text=True, timeout=TIMEOUT
    )
    report["seconds"] = round(time.monotonic() - started, 1)
    report["returncode"] = result.returncode
    report["stdout_tail"] = (result.stdout or "")[-1500:]
    if result.returncode != 0:
        report["stderr_tail"] = (result.stderr or "")[-3000:]
        return report

    report["produced"] = sorted(
        str(p.relative_to(out_folder)) for p in out_folder.rglob("*") if p.is_file()
    )
    # wrapper.py reads `{out_folder}/seqs/*.fa` and drops the first record,
    # which is the input sequence rather than a design.
    fastas = sorted(out_folder.glob("seqs/*.fa"))
    if fastas:
        text = fastas[0].read_text()
        report["fasta_name"] = fastas[0].name
        report["fasta"] = text[:2000]
        records = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.startswith(">")
        ]
        report["n_records"] = len(records)
        report["n_designs_excluding_native"] = max(len(records) - 1, 0)
        if records:
            # The motif must survive at the fixed positions. `fixed_residues` is
            # 1-based output numbering on the binder chain, and the first
            # record is the native sequence to compare against.
            report["native_len"] = len(records[0].split(":")[0])
            report["design_lens"] = [len(r.split(":")[0]) for r in records[1:]]
            positions = [int(p[1:]) for p in fixed_residues.split()]
            report["motif_in_native"] = "".join(
                records[0].split(":")[0][p - 1] for p in positions
            )
            report["motif_in_designs"] = [
                "".join(r.split(":")[0][p - 1] for p in positions) for r in records[1:]
            ]
            # The target chains must come back byte-identical to the native.
            # Anything else means MPNN treated them as designable.
            native_targets = records[0].split(":")[1:]
            report["target_chains_unchanged"] = all(
                r.split(":")[1:] == native_targets for r in records[1:]
            )
            report["n_target_segments"] = len(native_targets)

    return report


@app.local_entrypoint()
def main(stage: str = "check"):
    """`modal run experiments/mpnn_image.py --stage check|install|run_one`."""
    report = {"check": check, "install": install, "run_one": run_one}[stage].remote()
    for key, value in report.items():
        text = str(value)
        print(f"\n### {key}\n{text}" if len(text) > 80 else f"{key}: {text}")
