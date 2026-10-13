"""A spec name has to survive becoming a filename.

Every output path in the pipeline descends from `SpecConfig.name`: it reaches
`design_id`, then `seq_id`, then shard filenames, RFD3 output keys and the AF3
fork's job directories. The fork is the strict one -- it writes to
`sanitised_name()`, dropping anything outside `[A-Za-z0-9_-.]`
(`folding_input.py:1031-1035`).

This was a live bug rather than a hypothetical: cross-validation looked up its
outputs under the raw `seq_id`, so a spec named "rgd core" would have exited 0
having reported `no output directory` for every sequence -- indistinguishable,
from the summary, from a model that produced nothing.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rfd3_pipeline.config import SpecConfig

CONTIG = "100-120,/0,B1-177,/0,C1-242"


def spec(name: str) -> SpecConfig:
    return SpecConfig(name=name, contig=CONTIG)


class TestSpecNameSafety:
    @pytest.mark.parametrize("name", ["rgd_core", "rgd-core", "rgd.core2", "core2"])
    def test_filename_safe_names_are_accepted(self, name):
        assert spec(name).name == name

    @pytest.mark.parametrize(
        "name, offender",
        [
            ("rgd core", " "),
            ("rgd+core", "+"),
            ("rgd/core", "/"),
            ("rgd:core", ":"),
            ("rgd*core", "*"),
        ],
    )
    def test_names_that_would_not_survive_are_rejected(self, name, offender):
        with pytest.raises(ValidationError) as caught:
            spec(name)
        # The message has to name the character; "invalid name" would send
        # someone hunting through a contig for a problem that is one space.
        assert offender in str(caught.value)

    def test_the_empty_name_is_rejected(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            spec("")

    def test_rejection_rather_than_silent_renaming(self):
        """Sanitising "rgd core" to "rgd_core" would leave the output directory
        not matching the campaign file, which is harder to notice than an
        error at validation time."""
        with pytest.raises(ValidationError):
            spec("rgd core")

    def test_names_differing_only_in_a_stripped_character_cannot_both_exist(self):
        """The worse half of the bug. "rgd+core" and "rgdcore" both sanitise to
        "rgdcore", so their results would overwrite each other in one output
        directory with no error at all. Rejecting the first prevents the pair."""
        assert spec("rgdcore").name == "rgdcore"
        with pytest.raises(ValidationError):
            spec("rgd+core")
