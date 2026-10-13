from __future__ import annotations

import numpy as np
import pytest

from rfd3_pipeline.metrics import d0_from_n0, directional_ipsae, ipsae_from_pae


class TestD0:
    def test_floored_below_the_switch_point(self):
        assert d0_from_n0(0) == 1.0
        assert d0_from_n0(26) == 1.0

    def test_the_step_at_27_is_real_and_matches_the_reference(self):
        """It is *not* continuous here, and the old name claimed it was.

        `1.24 * (n0 - 15)**(1/3) - 1.8` crosses 1.0 at n0 ~ 26.5, so n0 = 26
        gives 1.0 and n0 = 27 gives 1.0389. The reference reaches the same step
        by clamping L to >= 26 and flooring at 1.0, so the step is a property of
        the score and not of this implementation. Pinned to the exact value
        because the previous assertion was `approx(1.0, abs=0.05)`, which passed
        while encoding the wrong claim.
        """
        assert d0_from_n0(27) == pytest.approx(1.0389, abs=1e-4)
        assert d0_from_n0(26) == 1.0

    def test_grows_with_interface_size(self):
        assert d0_from_n0(100) > d0_from_n0(50) > d0_from_n0(27)


class TestDirectional:
    def test_perfect_confidence_scores_one(self):
        pae = np.zeros((20, 20))
        assert directional_ipsae(pae, range(10), range(10, 20), 10.0) == pytest.approx(1.0)

    def test_no_passing_pair_scores_zero(self):
        pae = np.full((20, 20), 30.0)
        assert directional_ipsae(pae, range(10), range(10, 20), 10.0) == 0.0

    def test_cutoff_is_strict(self):
        pae = np.full((20, 20), 10.0)
        assert directional_ipsae(pae, range(10), range(10, 20), 10.0) == 0.0

    def test_only_passing_pairs_contribute(self):
        """A residue with one confident partner is not penalised for the rest."""
        pae = np.full((4, 4), 100.0)
        pae[0, 2] = 0.0  # residue 0 has exactly one confident partner
        score = directional_ipsae(pae, [0, 1], [2, 3], 10.0)
        assert score == pytest.approx(1.0)


class TestIpsaeFromPae:
    def test_rejects_pae_that_does_not_match_the_chains(self):
        """The load-bearing guard: wrong offsets would give plausible wrong numbers."""
        pae = np.zeros((25, 25))
        with pytest.raises(ValueError, match="Refusing to slice"):
            ipsae_from_pae(pae, ["A", "B"], [10, 10])

    def test_rejects_non_square(self):
        with pytest.raises(ValueError, match="square"):
            ipsae_from_pae(np.zeros((10, 12)), ["A"], [10])

    def test_rejects_duplicate_chain_ids(self):
        with pytest.raises(ValueError, match="unique"):
            ipsae_from_pae(np.zeros((20, 20)), ["A", "A"], [10, 10])

    def test_rejects_unknown_binder_chain(self):
        with pytest.raises(ValueError, match="not among chains"):
            ipsae_from_pae(np.zeros((20, 20)), ["A", "B"], [10, 10], binder_chain="Z")

    def test_reports_both_directions(self):
        pae = np.zeros((20, 20))
        out = ipsae_from_pae(pae, ["A", "B"], [10, 10])
        assert set(out) == {"ipsae_A_B", "ipsae_B_A"}

    def test_score_is_asymmetric(self):
        """Confident in one direction only; the reverse must not inherit it."""
        pae = np.full((20, 20), 100.0)
        pae[0:10, 10:20] = 0.0  # A aligned, B measured -> fully confident
        out = ipsae_from_pae(pae, ["A", "B"], [10, 10])
        assert out["ipsae_A_B"] == pytest.approx(1.0)
        assert out["ipsae_B_A"] == 0.0

    def test_score_tracks_pae_through_d0(self):
        """Anchors the exact arithmetic, not just its ordering.

        Ten passing partners puts d0 below the continuity point, so d0 == 1.0
        and a uniform PAE of 1.0 gives 1 / (1 + (1/1)^2) == 0.5. A regression in
        either the d0 floor or the ratio would move this number.
        """
        pae = np.full((20, 20), 100.0)
        pae[0:10, 10:20] = 1.0
        out = ipsae_from_pae(pae, ["A", "B"], [10, 10])
        assert out["ipsae_A_B"] == pytest.approx(0.5)

    def test_binder_min_takes_the_worse_direction(self):
        pae = np.full((20, 20), 100.0)
        pae[0:10, 10:20] = 1.0
        out = ipsae_from_pae(pae, ["A", "B"], [10, 10], binder_chain="A")
        assert out["ipsae_min_B"] == 0.0
        assert out["ipsae_min"] == 0.0

    def test_max_ipsae_min_only_with_several_targets(self):
        single = ipsae_from_pae(np.zeros((20, 20)), ["A", "B"], [10, 10], binder_chain="A")
        assert "max_ipsae_min" not in single

        multi = ipsae_from_pae(
            np.zeros((30, 30)), ["A", "B", "C"], [10, 10, 10], binder_chain="A"
        )
        assert multi["max_ipsae_min"] == pytest.approx(1.0)
        assert set(multi) >= {"ipsae_min_B", "ipsae_min_C", "ipsae_min", "max_ipsae_min"}

    def test_binder_keys_named_for_the_target_not_pair_position(self):
        """Keys must follow the binder, not alphabetical order of the pair."""
        out = ipsae_from_pae(
            np.zeros((30, 30)), ["A", "B", "C"], [10, 10, 10], binder_chain="B"
        )
        assert "ipsae_min_A" in out
        assert "ipsae_min_C" in out
        assert "ipsae_min_B" not in out

    def test_does_not_mutate_the_caller_array(self):
        pae = np.full((20, 20), 100.0)
        before = pae.copy()
        ipsae_from_pae(pae, ["A", "B"], [10, 10])
        np.testing.assert_array_equal(pae, before)

    def test_chain_offsets_follow_declared_lengths(self):
        """Unequal chain lengths must slice at the right boundary."""
        pae = np.full((30, 30), 100.0)
        pae[0:5, 5:30] = 0.0  # chain A is the first 5 rows only
        out = ipsae_from_pae(pae, ["A", "B"], [5, 25], binder_chain="A")
        assert out["ipsae_A_B"] == pytest.approx(1.0)
        assert out["ipsae_B_A"] == 0.0


class TestTheVectorisedD0MatchesItsReference:
    """`d0_from_n0` calls itself "the scalar reference the tests pin" and
    `_d0_array` is what the hot path actually runs. Nothing compared them, so
    the two could have drifted with every test still passing -- the scalar one
    is not on any code path."""

    def test_they_agree_across_the_range(self):
        import numpy as np

        from rfd3_pipeline.metrics.ipsae import _d0_array, d0_from_n0

        n0 = np.arange(0, 400)
        vectorised = _d0_array(n0).ravel()
        scalar = np.array([d0_from_n0(int(n)) for n in n0])
        assert vectorised == pytest.approx(scalar)

    def test_they_agree_at_the_continuity_point(self):
        """n0 = 27 is where the expression crosses the 1.0 floor, so it is
        where a floor applied differently in the two forms would show."""
        import numpy as np

        from rfd3_pipeline.metrics.ipsae import _d0_array, d0_from_n0

        for n in (25, 26, 27, 28):
            assert _d0_array(np.array([n])).ravel()[0] == pytest.approx(
                d0_from_n0(n)
            )


class TestAgreesWithTheDunbrackReference:
    """Numeric agreement with DunbrackLab/IPSAE, so a rewrite cannot drift.

    The reference is a CLI that parses structures and opens output files at
    import, so its arithmetic is transcribed here rather than imported -- and
    transcribed deliberately, since a copy that adapted our code would agree
    with us by construction and prove nothing.

    Transcribed from `ipsae.py` at `main` (verified 2026-10-13): `ptm_func`
    line 111, `calc_d0_array` line 126, and the `ipSAE_d0res` loop at lines
    783-801 whose per-pair value is the max over aligned residues (line 844).
    """

    @staticmethod
    def reference_d0_array(n0):
        """`calc_d0_array(L, 'protein')` -- clamp to >= 26, then floor at 1.0."""
        length = np.maximum(26, np.asarray(n0, dtype=float))
        return np.maximum(1.0, 1.24 * (length - 15) ** (1.0 / 3.0) - 1.8)

    @classmethod
    def reference_ipsae(cls, pae, chains, aligned, measured, pae_cutoff):
        """`ipsae_d0res_asym[aligned][measured]`."""
        valid = np.outer(chains == aligned, chains == measured) & (pae < pae_cutoff)
        d0 = cls.reference_d0_array(valid.sum(axis=1))
        per_residue = np.zeros(pae.shape[0])
        for i in range(pae.shape[0]):
            if chains[i] != aligned or not valid[i].any():
                continue
            scores = 1.0 / (1.0 + (pae[i] / d0[i]) ** 2.0)
            per_residue[i] = scores[valid[i]].mean()
        return float(per_residue.max())

    def test_d0_matches_across_the_whole_range(self):
        for n0 in range(201):
            assert d0_from_n0(n0) == pytest.approx(
                float(self.reference_d0_array(n0)), abs=1e-12
            ), f"d0 diverges at n0={n0}"

    def test_directional_scores_match_on_an_asymmetric_matrix(self):
        """Both directions of both pairs, on a matrix with real structure.

        Seeded rather than hand-written: a matrix simple enough to write out is
        also simple enough to agree by accident. The confident block is what
        keeps the scores off 0.0 and exercises the d0 switch.
        """
        rng = np.random.default_rng(20261013)
        lengths = {"A": 20, "B": 35, "C": 15}
        total = sum(lengths.values())
        pae = rng.uniform(2.0, 30.0, size=(total, total))
        # A confident patch between A and B, so n0 crosses 27 for some residues.
        pae[0:12, 20:50] = rng.uniform(1.0, 9.0, size=(12, 30))
        chains = np.array(
            [chain for chain, n in lengths.items() for _ in range(n)]
        )
        blocks, start = {}, 0
        for chain, n in lengths.items():
            blocks[chain] = slice(start, start + n)
            start += n

        for aligned in lengths:
            for measured in lengths:
                if aligned == measured:
                    continue
                ours = directional_ipsae(
                    pae, blocks[aligned], blocks[measured], 10.0
                )
                theirs = self.reference_ipsae(pae, chains, aligned, measured, 10.0)
                assert ours == pytest.approx(theirs, abs=1e-12), (
                    f"{aligned}->{measured} diverges: {ours} vs {theirs}"
                )

    def test_both_agree_that_nothing_under_the_cutoff_is_zero(self):
        pae = np.full((10, 10), 50.0)
        chains = np.array(["A"] * 5 + ["B"] * 5)
        assert directional_ipsae(pae, slice(0, 5), slice(5, 10), 10.0) == 0.0
        assert self.reference_ipsae(pae, chains, "A", "B", 10.0) == 0.0


class TestPerTargetMinAndMax:
    """Both directions summarised, with the two `max` names kept apart.

    `ipsae_max_<target>` maximises over the two *directions* of one target and
    exists to be compared with a published threshold, which the reference
    derives from the maximum. `max_ipsae_min` maximises over *targets* of each
    one's minimum. They sit one letter apart and mean different things.
    """

    @staticmethod
    def two_targets():
        """A binder A on targets B and C, deliberately asymmetric.

        B is confident from A's side only; C is confident from both. So B's
        two directions disagree and C's agree, which is what separates the
        min from the max.
        """
        lengths = {"A": 8, "B": 8, "C": 8}
        pae = np.full((24, 24), 40.0)
        pae[0:8, 8:16] = 3.0      # A -> B confident
        pae[8:16, 0:8] = 25.0     # B -> A not
        pae[0:8, 16:24] = 6.0     # A -> C confident
        pae[16:24, 0:8] = 6.0     # C -> A likewise
        return pae, list(lengths), list(lengths.values())

    def test_min_and_max_bracket_the_two_directions(self):
        pae, ids, lengths = self.two_targets()
        out = ipsae_from_pae(
            pae, ids, lengths, pae_cutoff=10.0, binder_chain="A", pairs="binder"
        )
        for target in ("B", "C"):
            pair = (out[f"ipsae_A_{target}"], out[f"ipsae_{target}_A"])
            assert out[f"ipsae_min_{target}"] == min(pair)
            assert out[f"ipsae_max_{target}"] == max(pair)

    def test_an_asymmetric_interface_separates_them(self):
        """B is confident one way only, so its min and max must differ."""
        pae, ids, lengths = self.two_targets()
        out = ipsae_from_pae(
            pae, ids, lengths, pae_cutoff=10.0, binder_chain="A", pairs="binder"
        )
        # B -> A has nothing under the cutoff, so the minimum is 0 while the
        # maximum is whatever A -> B scored. No magnitude is asserted: with
        # 8-residue chains n0 caps at 8, d0 floors at 1.0, and PAE 3.0 scores
        # 1/(1+9) = 0.1 -- small, but entirely confident relative to the other
        # direction, which is the only thing this test is about.
        assert out["ipsae_min_B"] == 0.0
        assert out["ipsae_max_B"] == out["ipsae_A_B"] > out["ipsae_min_B"]
        assert out["ipsae_min_C"] == pytest.approx(out["ipsae_max_C"])

    def test_the_two_max_names_are_not_the_same_quantity(self):
        """The hazard the docstring warns about, pinned.

        `ipsae_max_B` is high because A->B is confident; `max_ipsae_min` is C's
        value, because B's minimum is 0.0. Reading one as the other would claim
        a confident B interface that the reverse direction does not support.
        """
        pae, ids, lengths = self.two_targets()
        out = ipsae_from_pae(
            pae, ids, lengths, pae_cutoff=10.0, binder_chain="A", pairs="binder"
        )
        assert out["max_ipsae_min"] == out["ipsae_min_C"]
        assert out["ipsae_max_B"] > out["max_ipsae_min"]

    def test_max_matches_the_reference_headline(self):
        """`ipsae_max_<target>` is the reference's `ipsae_d0res_max`."""
        pae, ids, lengths = self.two_targets()
        chains = np.array([c for c, n in zip(ids, lengths) for _ in range(n)])
        out = ipsae_from_pae(
            pae, ids, lengths, pae_cutoff=10.0, binder_chain="A", pairs="binder"
        )
        reference = TestAgreesWithTheDunbrackReference.reference_ipsae
        for target in ("B", "C"):
            headline = max(
                reference(pae, chains, "A", target, 10.0),
                reference(pae, chains, target, "A", 10.0),
            )
            assert out[f"ipsae_max_{target}"] == pytest.approx(headline, abs=1e-12)

    def test_the_pipeline_still_decides_on_the_minimum(self):
        """Adding the max must not change which number the gate reads."""
        from rfd3_pipeline.gates import refold_gates
        from rfd3_pipeline.config import FoldingModel, RefoldingConfig

        gates = refold_gates(
            RefoldingConfig(model=FoldingModel.ESMFOLD2), "holo", has_motif=True
        )
        assert not any("ipsae" in gate.metric for gate in gates)
