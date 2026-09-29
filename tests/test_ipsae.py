from __future__ import annotations

import numpy as np
import pytest

from rfd3_pipeline.metrics import d0_from_n0, directional_ipsae, ipsae_from_pae


class TestD0:
    def test_floored_below_the_continuity_point(self):
        assert d0_from_n0(0) == 1.0
        assert d0_from_n0(26) == 1.0

    def test_continuous_at_27(self):
        # 1.24 * 12**(1/3) - 1.8 crosses 1.0 here, which is why the floor sits
        # at 27 rather than at an arbitrary residue count.
        assert d0_from_n0(27) == pytest.approx(1.0, abs=0.05)

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
