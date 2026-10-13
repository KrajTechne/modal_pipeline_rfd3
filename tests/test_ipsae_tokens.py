"""ipSAE from per-token chain labels, which is what both models actually give.

ESMFold2 writes `asym_id` beside the PAE in its `*_pae.npz`; the AF3 fork
writes `token_chain_ids` in its per-sample confidences. Neither reports chain
lengths, so deriving the blocks from the labels is what stops the caller
counting residues and disagreeing with the matrix it is slicing.

The `keep` argument is not cosmetic. Measured on this campaign's holo
prediction, including the single-token MN ion pulled `ipsae_min` from 0.598 to
0.199 -- the binder's worst "interface" became the one with the metal.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfd3_pipeline.metrics.ipsae import ipsae_from_pae, ipsae_from_tokens


def pae(n: int, *, confident: float = 2.0) -> np.ndarray:
    """A PAE where everything is confidently aligned."""
    matrix = np.full((n, n), confident, dtype=np.float32)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def labels(*blocks: tuple[str, int]) -> list[str]:
    out: list[str] = []
    for chain, length in blocks:
        out.extend([chain] * length)
    return out


class TestAgreesWithTheLengthForm:
    def test_same_answer_as_passing_lengths_directly(self):
        matrix = pae(12)
        tokens = labels(("A", 4), ("B", 8))
        assert ipsae_from_tokens(matrix, tokens, binder_chain="A") == pytest.approx(
            ipsae_from_pae(matrix, ["A", "B"], [4, 8], binder_chain="A")
        )

    def test_block_order_follows_the_tokens_not_the_alphabet(self):
        """A model is free to emit the binder last; the blocks must follow the
        matrix, not a sort."""
        matrix = pae(12)
        out = ipsae_from_tokens(matrix, labels(("B", 8), ("A", 4)), binder_chain="A")
        assert "ipsae_min_B" in out


class TestKeep:
    def test_a_ligand_chain_can_be_excluded(self):
        matrix = pae(13)
        tokens = labels(("A", 4), ("B", 8), ("E", 1))
        out = ipsae_from_tokens(matrix, tokens, binder_chain="A", keep=["A", "B"])
        assert "ipsae_min_E" not in out
        assert "ipsae_min_B" in out

    def test_excluding_it_changes_the_binder_minimum(self):
        """The reason `keep` exists. A one-token chain scores differently, and
        unfiltered it lands in `ipsae_min` as though it were a target."""
        # The ion is poorly aligned to everything else.
        matrix = pae(13)
        matrix[12, :12] = 25.0
        matrix[:12, 12] = 25.0
        tokens = labels(("A", 4), ("B", 8), ("E", 1))
        with_ion = ipsae_from_tokens(matrix, tokens, binder_chain="A")
        without = ipsae_from_tokens(matrix, tokens, binder_chain="A", keep=["A", "B"])
        assert with_ion["ipsae_min"] < without["ipsae_min"]

    def test_the_slice_keeps_the_remaining_blocks_aligned(self):
        """Dropping a middle chain must not shift the others' offsets."""
        matrix = pae(13)
        tokens = labels(("A", 4), ("E", 1), ("B", 8))
        dropped = ipsae_from_tokens(matrix, tokens, binder_chain="A", keep=["A", "B"])
        direct = ipsae_from_pae(pae(12), ["A", "B"], [4, 8], binder_chain="A")
        assert dropped == pytest.approx(direct)

    def test_asking_for_an_absent_chain_raises(self):
        with pytest.raises(ValueError, match="not among the tokens"):
            ipsae_from_tokens(pae(12), labels(("A", 4), ("B", 8)), keep=["A", "Z"])


class TestRefusesToGuess:
    def test_label_count_must_match_the_matrix(self):
        """The silent-error guard: a model returning tokens the caller did not
        account for would otherwise mis-slice every chain."""
        with pytest.raises(ValueError, match="refusing to guess"):
            ipsae_from_tokens(pae(12), labels(("A", 4), ("B", 7)))

    def test_a_split_chain_is_refused(self):
        """Non-contiguous tokens would make every slice describe the wrong
        residues while returning entirely plausible numbers."""
        tokens = labels(("A", 4), ("B", 4), ("A", 4))
        with pytest.raises(ValueError, match="more than one block"):
            ipsae_from_tokens(pae(12), tokens)

    def test_a_non_square_matrix_is_refused(self):
        with pytest.raises(ValueError, match="square"):
            ipsae_from_tokens(np.zeros((4, 8), dtype=np.float32), labels(("A", 4)))


class TestRealisticShape:
    def test_this_campaigns_layout(self):
        """108 binder + 177 + 242 + 1 ion = 528 tokens, as measured."""
        tokens = labels(("A", 108), ("B", 177), ("C", 242), ("E", 1))
        out = ipsae_from_tokens(
            pae(528), tokens, binder_chain="A", keep=["A", "B", "C"]
        )
        assert set(out) >= {
            "ipsae_A_B", "ipsae_B_A", "ipsae_A_C", "ipsae_C_A",
            "ipsae_min_B", "ipsae_min_C", "ipsae_min", "max_ipsae_min",
        }
        # Two targets, so the spread between them is reported.
        assert out["ipsae_min"] <= out["max_ipsae_min"]

    def test_float16_input_is_promoted(self):
        """The kit stores PAE as float16; the score divides and squares."""
        tokens = labels(("A", 4), ("B", 8))
        out = ipsae_from_tokens(pae(12).astype(np.float16), tokens, binder_chain="A")
        assert np.isfinite(out["ipsae_min"])


class TestBinderCentricPairs:
    """`pairs="binder"` is what both stages use.

    The ordered pairs grow as (N+1)N in the number of targets, and the
    target-to-target ones never involve the designed chain -- at eight targets
    that is 82 columns per stage against 26, doubled because holo and
    cross-validation both write them.
    """

    def test_target_to_target_pairs_are_dropped(self):
        tokens = labels(("A", 20), ("B", 20), ("C", 20))
        out = ipsae_from_tokens(
            pae(60), tokens, binder_chain="A", pairs="binder"
        )
        assert "ipsae_B_C" not in out and "ipsae_C_B" not in out

    def test_both_directions_of_each_binder_pair_are_kept(self):
        """The score is asymmetric, so dropping a direction would hide which
        chain is confidently placed relative to the other."""
        tokens = labels(("A", 20), ("B", 20), ("C", 20))
        out = ipsae_from_tokens(
            pae(60), tokens, binder_chain="A", pairs="binder"
        )
        assert {"ipsae_A_B", "ipsae_B_A", "ipsae_A_C", "ipsae_C_A"} <= set(out)

    def test_the_summaries_are_unchanged_by_dropping_them(self):
        """`ipsae_min` is a minimum over binder-target pairs, so target-target
        scores never contributed to it in the first place."""
        tokens = labels(("A", 20), ("B", 20), ("C", 20))
        matrix = pae(60)
        matrix[20:40, 40:60] = 20.0  # make the two targets disagree badly
        matrix[40:60, 20:40] = 20.0
        lean = ipsae_from_tokens(matrix, tokens, binder_chain="A", pairs="binder")
        full = ipsae_from_tokens(matrix, tokens, binder_chain="A", pairs="all")
        for key in ("ipsae_min", "ipsae_min_B", "ipsae_min_C", "max_ipsae_min"):
            assert lean[key] == pytest.approx(full[key])

    @pytest.mark.parametrize("n_targets", [1, 2, 3, 6])
    def test_it_generalises_past_two_targets(self, n_targets):
        chains = ["A"] + [chr(ord("B") + i) for i in range(n_targets)]
        tokens = labels(*[(c, 20) for c in chains])
        out = ipsae_from_tokens(
            pae(20 * len(chains)), tokens, binder_chain="A", pairs="binder"
        )
        # One minimum per target, and one overall.
        assert sum(1 for k in out if k.startswith("ipsae_min_")) == n_targets
        assert "ipsae_min" in out
        # Nothing to choose between with a single target.
        assert ("max_ipsae_min" in out) is (n_targets > 1)

    def test_it_needs_to_be_told_which_chain_is_the_binder(self):
        with pytest.raises(ValueError, match="needs binder_chain"):
            ipsae_from_tokens(pae(40), labels(("A", 20), ("B", 20)), pairs="binder")

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="must be 'all' or 'binder'"):
            ipsae_from_tokens(
                pae(40), labels(("A", 20), ("B", 20)),
                binder_chain="A", pairs="binder-only",
            )
