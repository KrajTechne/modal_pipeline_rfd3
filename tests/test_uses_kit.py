"""The kit-only settings, and that a directly-called model is not held to them.

`af3_fork` drives the sokrypton/alphafold3 fork through `run_alphafold.py`,
with no `run.sh`, no `--config` card, no `--mode` and no `--variant`. Before
`uses_kit` existed, `_FoldingBase` applied the kit's rules to every model: the
GPU had to appear in `GPU_TO_CARD`, which ruled out cards the fork runs on
perfectly well, and `n_gpu_shard` was accepted while reaching nothing.
"""

from __future__ import annotations

import pytest

from rfd3_pipeline.config import (
    MODEL_CAPABILITIES,
    CrossValidationConfig,
    FoldingModel,
    Weights,
)


class TestUsesKit:
    def test_the_fork_is_the_only_model_called_directly(self):
        direct = {
            model for model, caps in MODEL_CAPABILITIES.items() if not caps.uses_kit
        }
        assert direct == {FoldingModel.AF3_FORK}

    def test_a_directly_called_model_accepts_a_gpu_with_no_kit_card(self):
        # L40S has no card, so a kit model cannot run on it. The fork can.
        config = CrossValidationConfig(model=FoldingModel.AF3_FORK, gpu="L40S")
        assert config.gpu == "L40S"

    def test_a_kit_model_still_requires_a_card(self):
        with pytest.raises(ValueError, match="no kit card"):
            CrossValidationConfig(model=FoldingModel.AF3_JAX, gpu="L40S")

    def test_gpu_sharding_is_refused_rather_than_ignored(self):
        # Silently accepting this would promise a sharded prediction and
        # deliver a single-GPU one.
        with pytest.raises(ValueError, match="kit setting"):
            CrossValidationConfig(model=FoldingModel.AF3_FORK, n_gpu_shard=2)

    def test_the_fork_takes_no_variant(self):
        # `--model_dir` names the weights outright, so a variant would select
        # nothing. Rejected rather than dropped.
        with pytest.raises(ValueError, match="takes no variant"):
            CrossValidationConfig(model=FoldingModel.AF3_FORK, variant="p2")

    def test_asking_a_direct_model_for_a_kit_card_raises(self):
        config = CrossValidationConfig(model=FoldingModel.AF3_FORK)
        with pytest.raises(ValueError, match="no kit card"):
            config.kit_config

    def test_both_weight_sets_are_selectable(self):
        # The fork loads either into one codebase, which is why `af3_native`
        # here is a weights choice rather than a conversion project.
        assert MODEL_CAPABILITIES[FoldingModel.AF3_FORK].accepts_weights
        config = CrossValidationConfig(
            model=FoldingModel.AF3_FORK, weights=Weights.AF3_NATIVE
        )
        assert config.weights is Weights.AF3_NATIVE

    def test_the_default_weights_are_the_apache_licensed_ones(self):
        # OpenFold3 by default: the official AF3 parameters are
        # non-commercial, so they should never arrive by omission.
        config = CrossValidationConfig(model=FoldingModel.AF3_FORK)
        assert config.weights is Weights.OPENFOLD3

    def test_the_fork_asks_the_caller_for_paired_alignments(self):
        # The row that makes the MSA cache need a paired path at all.
        caps = MODEL_CAPABILITIES[FoldingModel.AF3_FORK]
        assert caps.msa_unpaired and caps.msa_paired and caps.consumes_msa
