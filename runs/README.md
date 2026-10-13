# Per-run records

One file per campaign run, named for its `run_name`. Each records what that
run measured and nothing that outlives it.

**The split is deliberate.** A number here describes one run and stays true of
that run forever. A conclusion in `Notes.md` claims something about the
*pipeline* and has to hold across runs — so a finding only graduates from here
to there once more than one run supports it, and `Notes.md` says which runs.

Write a record from the artifacts, never from the logs:

```
python experiments/metrics_table.py <run>/summary.csv <run>/summary.json
```

or the notebook (`experiments/campaign_analysis.ipynb`). Both call
`rfd3_pipeline.analysis`, so two records cannot disagree about a median the way
the first hand-made tables did.

Keep the section order identical between files. The point of these records is
comparison, and comparison is tedious when the headings move.

| run | date | scale | result |
|---|---|---|---|
| [alphav_beta3_02](alphav_beta3_02.md) | 2026-10-12 | 80 designs, 8 seqs, 5 CV seeds | 16 agreed sequences / 9 designs |
| [alphav_beta3_03](alphav_beta3_03.md) | 2026-10-13 | identical to 02 | running |

`alphav_beta3_01` has no record: it holds only a stale summary pair from a
`reduce_only` against the wrong directory.
