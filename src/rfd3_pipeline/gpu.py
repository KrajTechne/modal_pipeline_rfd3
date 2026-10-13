"""GPU-dependent tuning that the models need told to them.

Lives in the package rather than in `app.py` so the probe scripts under
`experiments/` share one copy -- and so it can be tested without a GPU, which
is the only way the boundary cases below ever get exercised.
"""

from __future__ import annotations

__all__ = ["attention_tuning", "AttentionTuning"]

from typing import NamedTuple


class AttentionTuning(NamedTuple):
    """What to tell the AF3 fork about attention on this card."""

    implementation: str
    """`--flash_attention_implementation`: "triton" or "xla"."""

    xla_flags: list[str]
    """Added to `XLA_FLAGS`, not replacing it."""

    nojit: bool
    """`--nojit`, to skip a compile that would never pay for itself."""


def attention_tuning(compute_caps: list[float] | None = None) -> AttentionTuning:
    """Pick the attention kernel from the GPU's compute capability.

    From the fork's own reference notebook, by way of the Optimus repo. The
    boundaries are not a tidy ordering, and the middle test is strict on both
    sides deliberately:

    * Triton flash attention needs Ampere or newer (cc >= 8.0).
    * Ada and consumer Ampere (cc 8.6/8.9, e.g. L40S) are newer *and* have too
      little shared memory for Triton's GEMM kernels, so they must fall back.
    * A100 is cc **8.0** exactly, and takes the Triton branch.

    So "newer is better" is false here, and a naive `cap >= 8.0` test would
    mistune every L40S. `compute_caps` is for testing; production passes
    nothing and the capabilities are read from `nvidia-smi`.
    """
    if compute_caps is None:
        compute_caps = _query_compute_caps()

    if not compute_caps:
        # No GPU: XLA attention, and skip the compile entirely.
        return AttentionTuning("xla", [], True)

    cap = min(compute_caps)
    if cap < 8.0:
        return AttentionTuning(
            "xla", ["--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"], False
        )
    if 8.0 < cap < 9.0:
        return AttentionTuning("xla", ["--xla_gpu_enable_triton_gemm=false"], False)
    return AttentionTuning("triton", ["--xla_gpu_enable_triton_gemm=false"], False)


def _query_compute_caps() -> list[float]:
    """Every visible GPU's compute capability. Empty when there is no GPU.

    Swallows every failure on purpose: a missing or broken `nvidia-smi` should
    tune for CPU, not crash a fold.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return [float(value) for value in out.stdout.split() if value.strip()]
    except Exception:
        return []
