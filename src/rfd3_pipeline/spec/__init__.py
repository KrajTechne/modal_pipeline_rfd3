"""Campaign resolution and RFD3 design bookkeeping.

GPU-free, Modal-free and network-free. The contig grammar and output-numbering
arithmetic live one level up in `rfd3_pipeline.contig`, so the campaign schema
can depend on them too; this package is what combines them with a structure and
with a realised design.

This is the riskiest code in the pipeline. Every downstream number -- hotspot
contact fraction, motif RMSD, the MPNN fixed-residue set, the final ranking --
is expressed in the coordinates produced here, and an error in them does not
raise. It yields a plausible, entirely wrong answer.
"""

from .design import (
    DesignMetadata,
    MotifMap,
    assert_output_chains,
    cross_check_target_mapping,
    load_design_metadata,
    parse_design_metadata,
    renumber_hotspots_for_design,
)
from .resolve import (
    ResolvedCampaign,
    ResolvedSpec,
    build_rfd3_specs,
    dump_rfd3_specs,
    resolve_campaign,
)

__all__ = [
    "DesignMetadata",
    "MotifMap",
    "ResolvedCampaign",
    "ResolvedSpec",
    "assert_output_chains",
    "build_rfd3_specs",
    "cross_check_target_mapping",
    "dump_rfd3_specs",
    "load_design_metadata",
    "parse_design_metadata",
    "renumber_hotspots_for_design",
    "resolve_campaign",
]
