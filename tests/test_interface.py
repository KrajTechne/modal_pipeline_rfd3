from __future__ import annotations

import numpy as np
import pytest
from conftest import make_atoms

from rfd3_pipeline.metrics import binding_interface, epitope_coverage


class TestEpitopeCoverage:
    def test_perfect_match(self):
        assert epitope_coverage([1, 2, 3], [1, 2, 3]) == (1.0, 1.0, 1.0)

    def test_recall_is_the_hotspot_contact_fraction(self):
        # Two of four requested hotspots contacted.
        recall, _, _ = epitope_coverage([1, 2], [1, 2, 3, 4])
        assert recall == pytest.approx(0.5)

    def test_precision_measures_off_target_spillover(self):
        # Everything requested was hit, but half the epitope was elsewhere.
        recall, precision, _ = epitope_coverage([1, 2, 9, 10], [1, 2])
        assert recall == pytest.approx(1.0)
        assert precision == pytest.approx(0.5)

    def test_f1_balances_both(self):
        _, _, f1 = epitope_coverage([1, 2, 9], [1, 2, 3])
        assert f1 == pytest.approx(2 / 4)  # |intersection| / |union|

    def test_empty_sets_score_zero(self):
        assert epitope_coverage([], [1, 2]) == (0.0, 0.0, 0.0)
        assert epitope_coverage([1, 2], []) == (0.0, 0.0, 0.0)


def _two_chain_complex(gap: float):
    """Chain A and chain B as parallel rows of atoms, `gap` angstrom apart."""
    binder = make_atoms("A", [1, 2, 3], [[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 6.0, 0.0]])
    target = make_atoms("B", [10, 11, 12], [[gap, 0.0, 0.0], [gap, 3.0, 0.0], [gap, 6.0, 0.0]])
    return binder + target


class TestBindingInterface:
    def test_finds_contacts_within_cutoff(self):
        result = binding_interface(_two_chain_complex(4.0), "A", "B", cutoff=4.5)
        assert result.epitope == (10, 11, 12)
        assert result.paratope == (1, 2, 3)

    def test_finds_nothing_beyond_cutoff(self):
        result = binding_interface(_two_chain_complex(10.0), "A", "B", cutoff=4.5)
        assert result.epitope == ()
        assert result.paratope == ()

    def test_recall_against_hotspots(self):
        result = binding_interface(
            _two_chain_complex(4.0), "A", "B", hotspots=[10, 11, 99], cutoff=4.5
        )
        assert result.recall == pytest.approx(2 / 3)  # 10 and 11 hit, 99 missed
        assert result.precision == pytest.approx(2 / 3)  # 12 was off-target

    def test_min_contacts_tightens_the_definition(self):
        """One contacting heavy atom passes at 1, fails at 2."""
        binder = make_atoms("A", [1], [[0.0, 0.0, 0.0]])
        # Residue 10 has a single atom near the binder; residue 11 has two.
        target = make_atoms(
            "B", [10, 11, 11], [[4.0, 0.0, 0.0], [4.0, 1.0, 0.0], [4.0, -1.0, 0.0]]
        )
        complex_ = binder + target

        loose = binding_interface(complex_, "A", "B", cutoff=4.5, min_contacts=1)
        assert loose.epitope == (10, 11)

        strict = binding_interface(complex_, "A", "B", cutoff=4.5, min_contacts=2)
        assert strict.epitope == (11,)

    def test_ignores_hydrogens(self):
        """A hydrogen inside the cutoff must not create a contact."""
        binder = make_atoms("A", [1], [[0.0, 0.0, 0.0]])
        # Residue 10 is a hydrogen well inside the cutoff; residue 11 is a
        # heavy atom far outside it, so the chain has heavy atoms overall.
        target = make_atoms("B", [10, 11], [[2.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        target.element = np.array(["H", "C"])
        result = binding_interface(binder + target, "A", "B", cutoff=4.5)
        assert result.epitope == ()

    def test_raises_when_a_chain_has_no_heavy_atoms(self):
        binder = make_atoms("A", [1], [[0.0, 0.0, 0.0]])
        target = make_atoms("B", [10], [[4.0, 0.0, 0.0]])
        target.element = np.array(["H"])
        with pytest.raises(ValueError, match="no heavy atoms"):
            binding_interface(binder + target, "A", "B", cutoff=4.5)

    def test_raises_on_absent_chain(self):
        with pytest.raises(ValueError, match="target chain"):
            binding_interface(_two_chain_complex(4.0), "A", "Z")

    def test_columns_are_suffixed_by_target_chain(self):
        result = binding_interface(
            _two_chain_complex(4.0), "A", "B", hotspots=[10], cutoff=4.5
        )
        columns = result.to_columns()
        assert columns["epitope_length_B"] == 3
        assert columns["epitope_indices_B"] == "10,11,12"
        assert "epitope_recall_B" in columns

    def test_columns_carry_no_stage_prefix(self):
        """Which stage computed a metric is the reporting layer's concern."""
        columns = binding_interface(_two_chain_complex(4.0), "A", "B", cutoff=4.5).to_columns()
        assert not any(name.startswith(("bb_", "holo_", "apo_")) for name in columns)
