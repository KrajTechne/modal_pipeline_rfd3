"""Which MSA fields each folding model's input carries.

ESMFold2 takes a single a3m per chain and pairs internally; the AF3 fork takes
`unpaired_msa` and `paired_msa` separately and expects the caller to have paired
them. Applying the fork's two-field mechanics to an ESMFold2 input writes a key
its schema has no place for -- silently ignored, while looking configured.

Only the logic is tested here. The per-model values are the table's content, and
asserting each one back is restating the literal rather than checking anything;
one sweep over the whole table covers the invariant that matters.
"""

from __future__ import annotations

import dataclasses

import pytest

from rfd3_pipeline.config import MODEL_CAPABILITIES, FoldingModel, ModelCapabilities, SeedStyle


class TestTheInvariants:
    def test_consumes_msa_follows_from_the_two_flags(self):
        """Derived, so no row can claim to read MSAs while declaring no fields
        -- or the reverse."""
        for model, caps in MODEL_CAPABILITIES.items():
            assert caps.consumes_msa == (caps.msa_unpaired or caps.msa_paired), model

    def test_consumes_msa_is_not_storable(self):
        """It was a stored field; storing it alongside the flags allowed a state
        nothing would catch. This keeps it from quietly returning."""
        assert not any(
            f.name == "consumes_msa" for f in dataclasses.fields(ModelCapabilities)
        )

    def test_no_row_pairs_without_unpaired(self):
        for model, caps in MODEL_CAPABILITIES.items():
            if caps.msa_paired:
                assert caps.msa_unpaired, model


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


class TestTheDistinctionThatMatters:
    def test_the_two_models_in_use_differ_in_who_pairs(self):
        """The one fact worth pinning by name: it decides whether the MSA cache
        needs a paired path, and it is the thing that would be got wrong."""
        assert MODEL_CAPABILITIES[FoldingModel.ESMFOLD2].msa_paired is False
        assert MODEL_CAPABILITIES[FoldingModel.AF3_JAX].msa_paired is True
