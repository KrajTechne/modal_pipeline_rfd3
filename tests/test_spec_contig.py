from __future__ import annotations

import numpy as np
import pytest
from conftest import ALPHAV_CONTIG, make_atoms, min_distance, rgd_carboxylate

from rfd3_pipeline.contig import (
    ChainBreak,
    Designed,
    FromInput,
    IndeterminateMapping,
    derive_length,
    map_input_to_output,
    output_chain_ids,
    parse_contig,
    parse_residue_selection,
    segments,
)


class TestParseContig:
    def test_designed_range(self):
        assert parse_contig("100-120") == (Designed(100, 120),)

    def test_designed_fixed_length(self):
        component = parse_contig("70")[0]
        assert component == Designed(70, 70)
        assert component.is_fixed

    def test_from_input_range(self):
        assert parse_contig("B1-177") == (FromInput("B", 1, 177),)

    def test_from_input_single_residue(self):
        assert parse_contig("A203") == (FromInput("A", 203, 203),)

    def test_chain_break(self):
        assert parse_contig("/0") == (ChainBreak(),)

    def test_the_campaign_contig(self):
        assert parse_contig(ALPHAV_CONTIG) == (
            Designed(100, 120),
            ChainBreak(),
            FromInput("B", 1, 177),
            ChainBreak(),
            FromInput("C", 1, 242),
        )

    def test_documented_mixed_example(self):
        """From input.md:199 -- motif, design, motif, break, motif, design."""
        assert parse_contig("A40-60,70,A120-170,A203,/0,B3-45,60-80") == (
            FromInput("A", 40, 60),
            Designed(70, 70),
            FromInput("A", 120, 170),
            FromInput("A", 203, 203),
            ChainBreak(),
            FromInput("B", 3, 45),
            Designed(60, 80),
        )

    def test_tolerates_whitespace(self):
        assert parse_contig(" 100-120 , /0 , B1-177 ") == parse_contig("100-120,/0,B1-177")

    def test_parses_the_sampled_contig_rfd3_writes_back(self):
        """RFD3's `extra.sampled_contig` is fully expanded and type-annotated.

        A real one reads `115P,/0,B1,B2,...,B177,/0,C1,...` -- the drawn length
        carries a polymer-type suffix and every input residue is listed
        individually, so the campaign-contig parser has to accept both.
        """
        sampled = "115P,/0,B1,B2,B3,/0,C1,C2"
        components = parse_contig(sampled)
        assert components[0] == Designed(115, 115)
        assert components[0].is_fixed
        assert components[2] == FromInput("B", 1, 1)
        assert len(segments(components)) == 3

    def test_polymer_suffix_does_not_change_the_count(self):
        assert parse_contig("115P") == parse_contig("115")

    @pytest.mark.parametrize("bad", ["", "   ", ",,,"])
    def test_rejects_empty(self, bad):
        with pytest.raises(ValueError, match="empty contig|no components"):
            parse_contig(bad)

    def test_rejects_unparseable_component(self):
        with pytest.raises(ValueError, match="cannot parse"):
            parse_contig("100-120,??,B1-177")

    def test_rejects_undocumented_chain_break(self):
        with pytest.raises(ValueError, match="only '/0'"):
            parse_contig("100,/5,B1-10")

    def test_rejects_backwards_ranges(self):
        with pytest.raises(ValueError, match="backwards"):
            parse_contig("B177-1")
        with pytest.raises(ValueError, match="backwards"):
            parse_contig("120-100")


class TestParseResidueSelection:
    def test_the_campaign_unindex_strings(self):
        assert parse_residue_selection("A77-79") == (("A", 77), ("A", 78), ("A", 79))

    def test_extended_motif(self):
        assert parse_residue_selection("A75-81") == tuple(("A", r) for r in range(75, 82))

    def test_multiple_components(self):
        assert parse_residue_selection("A13-15,A53,A125") == (
            ("A", 13), ("A", 14), ("A", 15), ("A", 53), ("A", 125),
        )

    def test_ignores_sequence_offsets(self):
        """`A11,0,A12` -- the bare 0 is an offset, not a designed region."""
        assert parse_residue_selection("A11,0,A12") == (("A", 11), ("A", 12))

    def test_ignores_trailing_chain_breaks(self):
        """The original YAML carried a trailing `/0,/0` on every unindex."""
        assert parse_residue_selection("A77-79,/0,/0") == parse_residue_selection("A77-79")

    def test_empty(self):
        assert parse_residue_selection("") == ()

    def test_hotspot_string(self):
        hotspots = parse_residue_selection("B59,B86,C10,C225")
        assert hotspots == (("B", 59), ("B", 86), ("C", 10), ("C", 225))


class TestSegments:
    def test_splits_on_chain_breaks(self):
        assert len(segments(parse_contig(ALPHAV_CONTIG))) == 3

    def test_drops_empty_groups(self):
        assert len(segments(parse_contig("/0,100,/0"))) == 1

    def test_chain_labels_follow_contig_order(self):
        assert output_chain_ids(3) == ["A", "B", "C"]

    def test_rejects_more_chains_than_letters(self):
        with pytest.raises(ValueError, match="A-Z"):
            output_chain_ids(27)


class TestDeriveLength:
    def test_the_campaign_length(self):
        """100-120 designed + 177 + 242 == 519-539.

        The YAML shipped 510-570, which matches neither this contig nor the
        earlier 80-120 one, and whose range was 60 wide where the designed
        segment is 20. This is why `length` is derived and never typed.
        """
        assert derive_length(parse_contig(ALPHAV_CONTIG)) == (519, 539)

    def test_fixed_length_contig_has_no_range(self):
        assert derive_length(parse_contig("70,/0,B1-10")) == (80, 80)

    def test_counts_only_residues_present_in_the_structure(self):
        """A gapped chain must not be counted by its nominal span."""
        gapped = make_atoms("B", [1, 2, 3, 7, 8], np.zeros((5, 3)))
        assert derive_length(parse_contig("10,/0,B1-8")) == (18, 18)
        assert derive_length(parse_contig("10,/0,B1-8"), structure=gapped) == (15, 15)


class TestMapInputToOutput:
    def test_campaign_targets_map_to_themselves(self):
        """Identity here -- and that is exactly why it is a poor sole fixture."""
        mapping = map_input_to_output(parse_contig(ALPHAV_CONTIG))
        assert mapping[("B", 59)] == ("B", 59)
        assert mapping[("C", 225)] == ("C", 225)

    def test_designed_region_produces_no_input_keys(self):
        mapping = map_input_to_output(parse_contig(ALPHAV_CONTIG))
        assert not any(chain == "A" for chain, _ in mapping)

    def test_offset_start_shifts_every_residue(self):
        """`B5-181` renumbers to 1-177, so input B77 becomes output B73."""
        mapping = map_input_to_output(parse_contig("100-120,/0,B5-181"))
        assert mapping[("B", 5)] == ("B", 1)
        assert mapping[("B", 77)] == ("B", 73)

    def test_contig_order_decides_the_output_chain_letter(self):
        """Input C listed first becomes output chain B."""
        mapping = map_input_to_output(parse_contig("100-120,/0,C1-242,/0,B1-177"))
        assert mapping[("C", 10)] == ("B", 10)
        assert mapping[("B", 59)] == ("C", 59)

    def test_several_components_in_one_output_chain_accumulate(self):
        mapping = map_input_to_output(parse_contig("B1-10,A20-25"))
        assert mapping[("B", 10)] == ("A", 10)
        assert mapping[("A", 20)] == ("A", 11)
        assert mapping[("A", 25)] == ("A", 16)

    def test_fixed_designed_region_shifts_deterministically(self):
        mapping = map_input_to_output(parse_contig("B1-5,10,B20-22"))
        assert mapping[("B", 20)] == ("A", 16)

    def test_variable_designed_region_makes_later_residues_indeterminate(self):
        with pytest.raises(IndeterminateMapping, match="sampled length"):
            map_input_to_output(parse_contig("B1-5,10-20,B30-32"))

    def test_variable_region_is_fine_when_nothing_follows_it(self):
        mapping = map_input_to_output(parse_contig("B1-5,10-20"))
        assert mapping[("B", 5)] == ("A", 5)

    def test_variable_region_does_not_affect_a_later_chain(self):
        """The break resets the position counter, so chain B stays exact."""
        mapping = map_input_to_output(parse_contig("100-120,/0,B1-177"))
        assert mapping[("B", 1)] == ("B", 1)

    def test_gaps_do_not_shift_later_residues_when_structure_is_given(self):
        gapped = make_atoms("B", [1, 2, 3, 7, 8], np.zeros((5, 3)))
        mapping = map_input_to_output(parse_contig("B1-8"), structure=gapped)
        assert mapping[("B", 7)] == ("A", 4)
        assert ("B", 4) not in mapping


class TestRealStructure:
    """Against the checked-in alphaV/beta3 input."""

    def test_chain_composition(self, alphav_structure):
        """Five chains, not three: A/B/C protein plus two Mn(2+) chains.

        The contig names only B and C, so D and E are dropped from the design
        unless RFD3's `ligand` field asks for them. For alphaV/beta3 that is a
        real modelling decision rather than housekeeping -- see
        test_metal_ions_are_present.
        """
        chains = sorted({str(c) for c in alphav_structure.chain_id})
        assert chains == ["A", "B", "C", "D", "E"]

    def test_metal_ions_are_present(self, alphav_structure):
        """Four Mn(2+): one in chain D, three in chain E.

        The beta3 MIDAS metal is coordinated directly by the RGD aspartate, so
        these are part of the binding mechanism the campaign is scaffolding.
        """
        metals = alphav_structure[alphav_structure.res_name == "MN"]
        assert metals.array_length() == 4
        assert sorted({str(c) for c in metals.chain_id}) == ["D", "E"]

    def test_protein_chain_sizes(self, alphav_structure):
        from rfd3_pipeline.structure import ca_only

        alpha_carbons = ca_only(alphav_structure)
        counts = {
            chain: int((alpha_carbons.chain_id == chain).sum())
            for chain in ("A", "B", "C")
        }
        assert counts == {"A": 93, "B": 177, "C": 242}

    def test_contig_does_not_reference_the_metal_chains(self):
        """Metals enter through `ligand`, never through the contig."""
        mapping = map_input_to_output(parse_contig(ALPHAV_CONTIG))
        assert not any(chain in ("D", "E") for chain, _ in mapping)

    def test_midas_metal_contacts_the_rgd_aspartate(self, alphav_structure):
        """E703 is the MIDAS metal: the RGD Asp completes its coordination.

        This is why the campaign sets `ligand: "E703"`. If a future input
        structure renumbers or renames it, this fails rather than silently
        scaffolding RGD without its principal interaction.
        """
        import numpy as np

        metal = alphav_structure[
            (alphav_structure.chain_id == "E") & (alphav_structure.res_id == 703)
        ]
        carboxylate = alphav_structure[
            (alphav_structure.chain_id == "A")
            & (alphav_structure.res_id == 79)
            & np.isin(alphav_structure.atom_name, ["OD1", "OD2"])
        ]
        assert metal.array_length() == 1
        assert carboxylate.array_length() == 2

        closest = np.linalg.norm(carboxylate.coord - metal.coord, axis=1).min()
        assert closest < 2.5  # a coordination bond, not a passing contact

    def test_chain_e_holds_three_ligand_residues(self, alphav_structure):
        """Why only E703 is requested.

        RFD3 raises when one chain carries several ligand residues
        (input_parsing.py:696-706), so asking for the full beta3 triad needs
        E703/E704/E705 split onto separate chains in the input first.
        """
        chain_e = alphav_structure[
            (alphav_structure.chain_id == "E") & (alphav_structure.res_name == "MN")
        ]
        assert sorted({int(r) for r in chain_e.res_id}) == [703, 704, 705]

    def test_propeller_metal_is_far_from_the_interface(self, alphav_structure):
        """D1004 is excluded on evidence, not by oversight."""
        import numpy as np

        metal = alphav_structure[
            (alphav_structure.chain_id == "D") & (alphav_structure.res_id == 1004)
        ]
        carboxylate = alphav_structure[
            (alphav_structure.chain_id == "A")
            & (alphav_structure.res_id == 79)
            & np.isin(alphav_structure.atom_name, ["OD1", "OD2"])
        ]
        closest = np.linalg.norm(carboxylate.coord - metal.coord, axis=1).min()
        assert closest > 30.0

    def test_length_matches_the_structure(self, alphav_structure):
        """Derived with and without the structure agree: no gaps in B or C."""
        components = parse_contig(ALPHAV_CONTIG)
        assert derive_length(components) == (519, 539)
        assert derive_length(components, structure=alphav_structure) == (519, 539)

    def test_declared_yaml_length_is_wrong(self, alphav_structure):
        derived = derive_length(parse_contig(ALPHAV_CONTIG), structure=alphav_structure)
        assert derived != (510, 570)

    def test_every_hotspot_lies_inside_the_contig(self, alphav_structure, alphav_hotspots):
        mapping = map_input_to_output(
            parse_contig(ALPHAV_CONTIG), structure=alphav_structure
        )
        assert all(hotspot in mapping for hotspot in alphav_hotspots)

    def test_rgd_motif_residues_exist_in_the_input(self, alphav_structure):
        """A77-79 must be ARG-GLY-ASP for the unindex spec to mean anything."""
        names = []
        for resi in (77, 78, 79):
            mask = (alphav_structure.chain_id == "A") & (alphav_structure.res_id == resi)
            names.append(str(alphav_structure.res_name[mask][0]))
        assert names == ["ARG", "GLY", "ASP"]

    def test_unindex_does_not_overlap_the_contig(self, alphav_structure):
        """Unindexed motif residues must not also be pulled in by the contig."""
        contig_residues = set(map_input_to_output(parse_contig(ALPHAV_CONTIG)))
        for spec in ("A77-79", "A75-81"):
            assert not (set(parse_residue_selection(spec)) & contig_residues)
