"""The backbone filter: one RFD3 design in, one shard record out.

Most of what the filter gates on, RFD3 already measured and wrote into the
design's metadata -- interresidue clashes and loop fraction come straight from
`metrics`. Two things do not:

* **Binder radius of gyration.** RFD3's own `radius_of_gyration` appears to
  describe the diffused region (it tracked our binder figure to ~1% on the one
  design measured), but ours is computed explicitly on `chain_binder`, so it is
  unambiguous and stays the gated value. RFD3's is recorded alongside it, which
  is what makes the two comparable over a real batch.
* **Hotspot contact fraction.** Needs the renumbered hotspots and a contact
  search, so it is computed from the structure.

The record this produces is one shard file. Nothing here writes to a shared
file, and nothing here decides a threshold -- `gates` owns that.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .gates import Stage, backbone_gates, evaluate_gates
from .metrics import binding_interface, rg_report
from .structure import ca_only, chain_subset

__all__ = ["backbone_record", "RFD3_METRIC_COLUMNS"]

# RFD3's metric names, mapped to the pipeline's. The left side is what the
# design metadata calls them; the right is what summary.csv does.
RFD3_METRIC_COLUMNS = {
    "n_clashing.interresidue_clashes_w_backbone": "clashes_backbone",
    "n_clashing.interresidue_clashes_w_sidechain": "clashes_sidechain",
    "loop_fraction": "loop_fraction",
    "helix_fraction": "helix_fraction",
    "sheet_fraction": "sheet_fraction",
    "radius_of_gyration": "rfd3_rg",
    "n_clashing.ligand_clashes": "ligand_clashes",
    "n_clashing.ligand_min_distance": "ligand_min_distance",
    "max_ca_deviation": "max_ca_deviation",
}

# Gated values RFD3 supplies directly; the rest are computed here.
_REQUIRED_RFD3_METRICS = (
    "n_clashing.interresidue_clashes_w_backbone",
    "n_clashing.interresidue_clashes_w_sidechain",
    "loop_fraction",
)


def _hotspot_contacts(
    design,
    binder_chain: str,
    hotspots_by_chain: Mapping[str, Sequence[int]],
    cutoff: float,
    min_contacts: int,
) -> tuple[dict[str, object], int, int]:
    """Per-target interface columns, plus pooled hotspot hit counts.

    The fraction is pooled across target chains rather than averaged per chain:
    the campaign's threshold is a fraction of *all* the hotspots it listed, and
    an epitope that spans two chains should not be penalised for the split.
    """
    columns: dict[str, object] = {}
    hit = total = 0

    for chain, hotspots in hotspots_by_chain.items():
        result = binding_interface(
            design,
            binder_chain,
            chain,
            hotspots,
            cutoff=cutoff,
            min_contacts=min_contacts,
        )
        columns.update(result.to_columns())
        hit += len(set(result.epitope) & set(hotspots))
        total += len(set(hotspots))

    return columns, hit, total


def backbone_record(
    design,
    metadata,
    spec,
    campaign,
    *,
    contact_cutoff: float = 4.5,
    min_contacts: int = 1,
) -> dict[str, object]:
    """Every `bb_*` column for one design, plus its gate verdict.

    Args:
        design: the loaded RFD3 output structure.
        metadata: that design's `DesignMetadata`.
        spec: the `ResolvedSpec` it came from.
        campaign: the `CampaignConfig`, for chains and thresholds.
        contact_cutoff: heavy-atom contact distance, angstrom.
        min_contacts: contacting heavy atoms a hotspot residue needs. The
            campaign uses 1.

    Raises:
        KeyError: RFD3's metadata lacks a metric the filter gates on.
    """
    binder_chain = campaign.general.chain_binder
    rfd3 = campaign.rfdiffusion3

    absent = [name for name in _REQUIRED_RFD3_METRICS if name not in metadata.metrics]
    if absent:
        raise KeyError(
            f"design metadata is missing gated RFD3 metric(s) {absent}; "
            f"available: {sorted(metadata.metrics)}"
        )

    values: dict[str, float] = {
        column: metadata.metrics[name]
        for name, column in RFD3_METRIC_COLUMNS.items()
        if name in metadata.metrics
    }

    # Ours, on the binder chain explicitly, and the gated Rg.
    values.update(rg_report(design, binder_chain))

    interface_columns, hit, total = _hotspot_contacts(
        design,
        binder_chain,
        spec.hotspots_by_output_chain(),
        contact_cutoff,
        min_contacts,
    )
    values["hotspot_fraction"] = hit / total if total else 0.0

    gates = backbone_gates(rfd3)
    verdict = evaluate_gates(gates, values, prefix="bb")

    motif = metadata.motif_map(binder_chain)
    record: dict[str, object] = {
        "spec_name": spec.name,
        "stage_reached": Stage.BACKBONE_FILTERS,
        "bb_n_hotspots_hit": hit,
        "bb_n_hotspots_total": total,
        "motif_map": motif.compact(),
        "mpnn_fixed_residues": " ".join(
            f"{binder_chain}{resi}" for resi in motif.output_residues
        ),
        "bb_pass": verdict.passed,
        "fail_stage": None if verdict.passed else Stage.BACKBONE_FILTERS,
        "fail_reasons": verdict.fail_reasons,
    }
    record.update({f"bb_{name}": value for name, value in values.items()})
    record.update({f"bb_{name}": value for name, value in interface_columns.items()})
    record.update(verdict.columns)
    return record
