from __future__ import annotations

import numpy as np
import pytest
from conftest import make_atoms, rotation_matrix

from rfd3_pipeline.metrics import (
    expected_radius_of_gyration,
    rg_report,
    rmsd_binder_apo_holo,
    rmsd_binder_on_target,
    rmsd_motif_on_target,
    target_superposition,
)
from rfd3_pipeline.structure import match_ca


class TestExpectedRadiusOfGyration:
    def test_empirical_scaling(self):
        assert expected_radius_of_gyration(100) == pytest.approx(2.38 * 100**0.365)
        assert expected_radius_of_gyration(200) > expected_radius_of_gyration(100)

    def test_rejects_empty_chain(self):
        with pytest.raises(ValueError, match="must be positive"):
            expected_radius_of_gyration(0)


class TestRgReport:
    def test_returns_numbers_not_a_verdict(self):
        """Gating is the gate layer's job; metrics report values."""
        coords = np.random.default_rng(1).normal(size=(20, 3)) * 0.5
        report = rg_report(make_atoms("A", range(1, 21), coords), "A")
        assert set(report) == {"binder_length", "rg", "rg_expected", "rg_ratio"}
        assert report["binder_length"] == 20

    def test_compact_binder_beats_its_expectation(self):
        coords = np.random.default_rng(1).normal(size=(20, 3)) * 0.5
        report = rg_report(make_atoms("A", range(1, 21), coords), "A")
        assert report["rg_ratio"] < 1.0

    def test_extended_binder_exceeds_it(self):
        coords = np.stack([np.arange(20) * 10.0, np.zeros(20), np.zeros(20)], axis=1)
        report = rg_report(make_atoms("A", range(1, 21), coords), "A")
        assert report["rg_ratio"] > 1.1

    def test_uses_only_the_binder_chain(self):
        """The whole-complex Rg from RFD3's JSON is why this is recomputed."""
        binder = make_atoms("A", range(1, 6), np.random.default_rng(2).normal(size=(5, 3)))
        target = make_atoms("B", range(1, 51), np.random.default_rng(3).normal(size=(50, 3)) * 30)
        assert rg_report(binder + target, "A")["binder_length"] == 5

    def test_raises_on_absent_chain(self):
        atoms = make_atoms("A", [1, 2], [[0, 0, 0], [1, 1, 1]])
        with pytest.raises(ValueError, match="binder chain"):
            rg_report(atoms, "Z")


class TestLoadStructure:
    """Format dispatch is biotite's; gzip is the one thing it does not cover."""

    def test_plain_and_gzipped_agree(self, tmp_path, alphav_structure):
        import gzip
        import shutil

        from conftest import ALPHAV_PDB

        from rfd3_pipeline.structure import load_structure

        gzipped = tmp_path / "copy.pdb.gz"
        with open(ALPHAV_PDB, "rb") as source, gzip.open(gzipped, "wb") as target:
            shutil.copyfileobj(source, target)

        loaded = load_structure(gzipped)
        assert loaded.array_length() == alphav_structure.array_length()
        np.testing.assert_allclose(loaded.coord, alphav_structure.coord)

    def test_gzip_without_an_inner_extension_raises(self, tmp_path):
        import gzip

        from rfd3_pipeline.structure import load_structure

        path = tmp_path / "mystery.gz"
        with gzip.open(path, "wt") as handle:
            handle.write("nothing useful")
        with pytest.raises(ValueError, match="needs its format extension"):
            load_structure(path)


class TestMatchCa:
    def test_pairs_on_residue_id_not_file_order(self):
        """The whole point: one writer's chain ordering must not mis-pair atoms."""
        coords = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
        forward = make_atoms("A", [1, 2, 3], coords)
        reversed_order = make_atoms("A", [3, 2, 1], coords[::-1])

        first, second, keys = match_ca(forward, reversed_order)
        assert keys == [("A", 1), ("A", 2), ("A", 3)]
        np.testing.assert_allclose(first.coord, second.coord)

    def test_keys_are_plain_python_types(self):
        """numpy scalars in the keys hash slower for no benefit."""
        atoms = make_atoms("A", [1, 2], np.zeros((2, 3)))
        _, _, keys = match_ca(atoms, atoms)
        assert all(type(c) is str and type(r) is int for c, r in keys)

    def test_intersects_on_shared_residues(self):
        a = make_atoms("A", [1, 2, 3], np.zeros((3, 3)))
        b = make_atoms("A", [2, 3, 4], np.zeros((3, 3)))
        _, _, keys = match_ca(a, b)
        assert keys == [("A", 2), ("A", 3)]

    def test_raises_when_nothing_is_shared(self):
        a = make_atoms("A", [1, 2], np.zeros((2, 3)))
        b = make_atoms("B", [1, 2], np.zeros((2, 3)))
        with pytest.raises(ValueError, match="no shared"):
            match_ca(a, b)


class TestApoHolo:
    def test_identical_binder_scores_zero(self, helix_holo):
        holo, binder = helix_holo
        assert rmsd_binder_apo_holo(holo, binder, "A") == pytest.approx(0.0, abs=1e-6)

    def test_rigid_motion_is_removed_by_superposition(self, helix_holo, helix_coords):
        holo, _ = helix_holo
        moved = helix_coords @ rotation_matrix([0.3, 1.0, 0.2], 0.9).T + [12.0, -4.0, 7.0]
        apo = make_atoms("A", range(1, 13), moved)
        # Tolerance is float32 coordinate noise, not slack: biotite stores coords
        # as float32, so a large rotation plus translation leaves ~1e-6 residue.
        assert rmsd_binder_apo_holo(holo, apo, "A") == pytest.approx(0.0, abs=1e-5)

    def test_conformational_change_is_detected(self, helix_holo, helix_coords):
        holo, _ = helix_holo
        bent = helix_coords.copy()
        bent[6:, 0] += 6.0
        apo = make_atoms("A", range(1, 13), bent)
        assert rmsd_binder_apo_holo(holo, apo, "A") > 1.0

    def test_raises_when_binder_absent_from_holo(self, helix_coords):
        holo = make_atoms("B", range(1, 13), helix_coords)
        apo = make_atoms("A", range(1, 13), helix_coords)
        with pytest.raises(ValueError, match="absent from the holo"):
            rmsd_binder_apo_holo(holo, apo, "A")


class TestTargetAligned:
    def _pair(self, binder_shift):
        """A designed and a predicted complex sharing identical target chains."""
        binder = np.array([[0.0, 0, 0], [3.0, 0, 0], [6.0, 0, 0], [9.0, 0, 0]])
        target = np.array([[0.0, 10, 0], [3.0, 10, 0], [6.0, 10, 0], [9.0, 10, 0]])
        designed = make_atoms("A", [1, 2, 3, 4], binder) + make_atoms("B", [1, 2, 3, 4], target)
        predicted = make_atoms("A", [1, 2, 3, 4], binder + binder_shift) + make_atoms(
            "B", [1, 2, 3, 4], target
        )
        return designed, predicted

    def test_zero_when_the_motif_is_placed_identically(self):
        designed, predicted = self._pair(0.0)
        assert rmsd_motif_on_target(designed, predicted, "A", ["B"], [2, 3]) == pytest.approx(
            0.0, abs=1e-6
        )

    def test_target_alignment_does_not_absorb_binder_displacement(self):
        """Aligning on the target is what makes binder motion visible."""
        designed, predicted = self._pair(np.array([0.0, 0.0, 5.0]))
        assert rmsd_motif_on_target(designed, predicted, "A", ["B"], [2, 3]) == pytest.approx(
            5.0, abs=1e-6
        )

    def test_one_superposition_serves_both_metrics(self):
        """Motif and whole-binder RMSD differ only in what they measure."""
        designed, predicted = self._pair(np.array([0.0, 0.0, 5.0]))
        fit = target_superposition(designed, predicted, "A", ["B"])

        assert fit.rmsd_over([2, 3]) == pytest.approx(
            rmsd_motif_on_target(designed, predicted, "A", ["B"], [2, 3])
        )
        assert fit.rmsd_over(None) == pytest.approx(
            rmsd_binder_on_target(designed, predicted, "A", ["B"])
        )

    def test_raises_on_unknown_motif_residue(self):
        designed, predicted = self._pair(0.0)
        with pytest.raises(ValueError, match="absent from the shared set"):
            rmsd_motif_on_target(designed, predicted, "A", ["B"], [2, 99])

    def test_raises_without_motif_residues(self):
        designed, predicted = self._pair(0.0)
        with pytest.raises(ValueError, match="no motif residues"):
            rmsd_motif_on_target(designed, predicted, "A", ["B"], [])

    def test_raises_when_target_chains_are_missing(self):
        designed, predicted = self._pair(0.0)
        with pytest.raises(ValueError, match="absent from the shared residue set"):
            target_superposition(designed, predicted, "A", ["Z"])
