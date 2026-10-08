"""Which MSA fields each folding model's input carries.

The distinction is load-bearing and was nearly got wrong. ESMFold2 takes a
single a3m per chain and pairs internally; the AF3 fork takes `unpaired_msa`
and `paired_msa` separately and expects the caller to have paired them. Applying
the fork's two-field mechanics to an ESMFold2 input would write a key the schema
has no place for -- silently ignored, while looking configured.
"""

from __future__ import annotations

import dataclasses

import pytest

from rfd3_pipeline.config import MODEL_CAPABILITIES, FoldingModel, ModelCapabilities, SeedStyle


class TestRows:
    def test_esmfold2_takes_per_chain_alignments_only(self):
        """One `"msa"` path per chain; the model does its own pairing."""
        caps = MODEL_CAPABILITIES[FoldingModel.ESMFOLD2]
        assert caps.msa_unpaired is True
        assert caps.msa_paired is False

    def test_esmfold2_fast_takes_none(self):
        caps = MODEL_CAPABILITIES[FoldingModel.ESMFOLD2_FAST]
        assert (caps.msa_unpaired, caps.msa_paired) == (False, False)

    def test_af3_jax_expects_the_caller_to_pair(self):
        """`unpaired_msa` and `paired_msa` are separate fields on the fork."""
        caps = MODEL_CAPABILITIES[FoldingModel.AF3_JAX]
        assert caps.msa_unpaired is True
        assert caps.msa_paired is True

    def test_every_model_has_both_flags(self):
        """A row missing one would be a TypeError at import, which is the
        point; this asserts the table is actually complete."""
        for model, caps in MODEL_CAPABILITIES.items():
            assert isinstance(caps.msa_unpaired, bool), model
            assert isinstance(caps.msa_paired, bool), model


class TestConsumesMsaIsDerived:
    def test_it_follows_from_the_two_flags(self):
        for caps in MODEL_CAPABILITIES.values():
            assert caps.consumes_msa == (caps.msa_unpaired or caps.msa_paired)

    def test_it_cannot_be_set_independently(self):
        """Derived rather than stored, so it cannot contradict the flags."""
        assert not any(
            f.name == "consumes_msa" for f in dataclasses.fields(ModelCapabilities)
        )

    def test_the_values_are_unchanged_by_the_refactor(self):
        """The field became a property; the answers must not move."""
        expected = {
            FoldingModel.ESMFOLD2: True,
            FoldingModel.ESMFOLD2_FAST: False,
            FoldingModel.BOLTZ2: True,
            FoldingModel.AF3_JAX: True,
        }
        for model, consumes in expected.items():
            assert MODEL_CAPABILITIES[model].consumes_msa is consumes


class TestPairedImpliesUnpaired:
    def test_paired_without_unpaired_is_rejected(self):
        """Pairing refines per-chain alignments; it does not replace them."""
        with pytest.raises(ValueError, match="msa_paired requires msa_unpaired"):
            ModelCapabilities(
                supports_templates=False,
                msa_unpaired=False,
                msa_paired=True,
                seed_style=SeedStyle.LIST,
            )

    def test_the_three_legal_combinations_are_accepted(self):
        for unpaired, paired in [(False, False), (True, False), (True, True)]:
            caps = ModelCapabilities(
                supports_templates=False,
                msa_unpaired=unpaired,
                msa_paired=paired,
                seed_style=SeedStyle.LIST,
            )
            assert caps.consumes_msa == (unpaired or paired)


class TestWhatThisDrivesDownstream:
    def test_a_campaign_can_ask_whether_pairing_is_needed_at_all(self):
        """The question the MSA cache has to answer. For an esmfold2 refolding
        stage the answer is no, which is why only unpaired alignments are worth
        caching."""
        assert MODEL_CAPABILITIES[FoldingModel.ESMFOLD2].msa_paired is False
        assert MODEL_CAPABILITIES[FoldingModel.AF3_JAX].msa_paired is True

    def test_no_msa_model_needs_no_cache(self):
        caps = MODEL_CAPABILITIES[FoldingModel.ESMFOLD2_FAST]
        assert not caps.consumes_msa
        assert not caps.msa_unpaired and not caps.msa_paired
