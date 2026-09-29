"""RFD3 protein binder design pipeline for Modal.

See modal_rfd3_pipeline_plan_claude.md for the design this implements.

Layering, innermost first:

* `structure` -- loading, chain selection, CA matching. Depends on nothing.
* `contig` -- the contig grammar and RFD3's output numbering. A leaf over
  `structure`, so both the schema and the design readers share one parser.
* `config` -- the campaign YAML schema. Everything checkable without
  coordinates, including contig syntax.
* `metrics` -- numbers from structures and PAE arrays. No thresholds, no
  verdicts, no assumptions about which chain is the binder.
* `spec` -- campaign resolution against a structure, and reading designs back.
"""
