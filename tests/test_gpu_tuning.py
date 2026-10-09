"""Attention tuning across the compute-capability boundaries.

These cases are the reason `attention_tuning` takes its input as an argument:
on a real card only one branch is ever reachable, so the boundary that actually
bites -- L40S being *newer* than A100 and still needing the slower kernel --
could not otherwise be exercised at all.
"""

from __future__ import annotations

import pytest

from rfd3_pipeline.gpu import attention_tuning


class TestAttentionTuning:
    def test_a100_takes_triton(self):
        # cc 8.0 exactly, the card the pipeline actually runs on.
        tuning = attention_tuning([8.0])
        assert tuning.implementation == "triton"
        assert not tuning.nojit

    @pytest.mark.parametrize("cap", [8.6, 8.9])
    def test_ada_and_consumer_ampere_fall_back_despite_being_newer(self, cap):
        """The case a naive `cap >= 8.0` test would get wrong.

        These are newer than A100 and still cannot run Triton's GEMM kernels:
        too little shared memory.
        """
        assert attention_tuning([cap]).implementation == "xla"

    def test_h100_takes_triton(self):
        assert attention_tuning([9.0]).implementation == "triton"

    @pytest.mark.parametrize("cap", [7.0, 7.5])
    def test_pre_ampere_disables_the_fusion_rewriter(self, cap):
        tuning = attention_tuning([cap])
        assert tuning.implementation == "xla"
        assert "custom-kernel-fusion-rewriter" in tuning.xla_flags[0]

    def test_no_gpu_skips_the_compile(self):
        """`--nojit` only here: a compile with no GPU never pays for itself."""
        tuning = attention_tuning([])
        assert tuning == ("xla", [], True)

    def test_the_weakest_card_decides(self):
        """Mixed hosts tune for the card that cannot keep up."""
        assert attention_tuning([9.0, 8.6]).implementation == "xla"

    def test_triton_still_disables_triton_gemm(self):
        """Not a contradiction, and worth pinning so it is not "tidied" away:
        flash attention is Triton while its GEMM kernels stay off."""
        tuning = attention_tuning([8.0])
        assert tuning.implementation == "triton"
        assert tuning.xla_flags == ["--xla_gpu_enable_triton_gemm=false"]
