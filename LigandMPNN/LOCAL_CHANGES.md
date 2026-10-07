# Local changes to the vendored LigandMPNN

Vendored from <https://github.com/dauparas/LigandMPNN> at commit
`26ec57ac976ade5379920dbd43c7f97a91cf82de`. Trimmed to the runtime-essential
files: `inputs/`, `outputs/` and `training/` are not kept.

Everything below is a deliberate deviation from upstream. Nothing else differs
-- verified 2026-10-07 by diffing the whole tree (`run.py`, `data_utils.py`,
`model_utils.py`, `sc_utils.py`, `score.py` are byte-identical to upstream
modulo line endings).

## 1. numpy aliases removed in 1.24 (6 occurrences, 3 files)

Upstream's bundled openfold uses `np.int`, `np.object` and `np.bool`, removed in
numpy 1.24. This is why upstream's `requirements.txt` pins `numpy==1.23.5` --
and that pin has no cp312 wheels, which would have forced this component onto
Python 3.11 and into a container of its own, away from the rest of the pipeline.

| File | Change |
|---|---|
| `openfold/np/residue_constants.py` | `dtype=np.int` -> `dtype=int` (x3) |
| `openfold/data/templates.py` | `np.object` -> `object` (x2) |
| `openfold/np/relax/utils.py` | `dtype=np.bool` -> `dtype=bool` (x1) |

With these fixed, the tree runs on Python 3.12 with numpy 2.5.3 and current
torch. `requirements.txt` is therefore **not** installed by the Modal image;
see `mpnn_image` in `app.py` for what is.

## 2. `--seed 0` was silently ignored (`run.py`)

`main()` tested `if args.seed:`, which is falsy for `0`, so an explicitly
requested seed of zero was replaced with
`np.random.randint(0, 99999)`. Combined with `default=0` in argparse, the
documented default was in fact "random", and seed 0 could not be requested at
all.

This matters here because the campaign sets `general.base_seed: 0`. A run would
report whatever random seed it happened to draw -- observed: 87657 -- and was
not reproducible, with no error and no warning.

Changed to `if args.seed is not None:` with `default=None`. Omitting `--seed`
still randomises, which keeps upstream's intent.

## Not changed, but worth knowing

`wrapper.py` (added here, sourced from the ProteinHunter repository) passes
`--return_logits` and `--temperature_per_residue`, which **this** `run.py` does
not define; those flags come from a fork. Calls using them will exit 2. It also
uses `subprocess.run(capture_output=True)`, which hides all progress -- the
pipeline calls `run.py` directly through `app.py`'s `_stream` instead.
