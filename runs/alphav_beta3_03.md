# alphav_beta3_03 — the second full campaign

**Finished 2026-10-13 16:15 CDT**, 3 h 17 m, app `ap-X9kbLGAtJYQItM6bWhm8bQ`,
orchestrator call `fc-01M4V6M3PE0H7A7N1740P3QKKK`.

**Everything but `run_name` was identical to
[alphav_beta3_02](alphav_beta3_02.md)** — a second independent sample under the
same conditions. `rfdiffusion3.seed` is null for a full campaign, so sampling
was unseeded and the 80 backbones are new rather than a resumed replay.

Scale: 2 specs × 5 batches × 8 = 80 designs, 8 MPNN sequences each, 5
cross-validation seeds, `ipsae.pae_cutoff: 10`.

## Funnel

```
rfd3                80 designs
backbone_filters    65 / 80    81.3%     (02: 70/80, 87.5%)
mpnn               520 sequences         (02: 560)
apo                260 / 520   50.0%     (02: 292/560, 52.1%)
holo               144 / 260   55.4%     (02: 138/292, 47.3%)
cv                 144 reported          (02: 138)
```

Filter rejections: `clashes_backbone` 7, `rg_ratio` 6, `loop_fraction` 3,
`clashes_sidechain` **0**, `hotspot_fraction` **0**.

## Result

**28 agreed sequences over 12 designs** — cross-validation confirms holo's
placement (≤ 5 Å) and motif (≤ 2 Å) on 28 of 144, **19.4%** against 02's 11.6%.

```
rgd_core_4_model_5       6        rgd_core_4_model_1       2
rgd_core_2_model_2       5        rgd_core_4_model_3       2
rgd_core_4_model_6       3        rgd_core_1_model_5       1
rgd_core_0_model_1       2        rgd_core_2_model_4       1
rgd_core_2_model_6       2        rgd_core_3_model_2       1
rgd_core_4_model_0       2        rgd_extended_3_model_3   1
```

Still concentrated: the top design gives 6 of the 28 and the top two give 11,
the same shape as 02 (4 of 16, and 7 from two backbones). Twelve distinct
designs against 02's nine.

**`rgd_extended` scored its first survivor** — `rgd_extended_3_model_3`, 1 of
28. See the per-spec section; the near-total failure is reproduced.

Best by `cv_ipsae_min`: `rgd_core_2_model_0_seq6` at 0.624, then
`rgd_core_1_model_1_seq8` 0.499 and `rgd_core_2_model_0_seq7` 0.477.

## Distributions

```
stage   metric            n     min     p25     med     p75     max
apo     pLDDT           520  42.860  65.781  70.019  75.015  85.965
apo     fold->design    520   0.659   1.806   2.770   4.888  21.240
holo    pLDDT           260  51.498  75.201  80.178  84.738  90.769
holo    ipTM            260   0.634   0.790   0.854   0.903   0.934
holo    ipSAE           260   0.000   0.043   0.306   0.566   0.805
holo    fold->design    260   0.574   1.070   1.541   2.562  17.069
holo    fold apo/holo   260   0.343   1.055   1.703   2.419  16.995
holo    placement       260   1.205   2.542   3.801  11.281  49.791
holo    motif           260   0.342   0.730   1.112   2.998  27.600
holo    ipSAE min B     260   0.000   0.047   0.317   0.572   0.807
holo    ipSAE min C     260   0.000   0.056   0.306   0.592   0.825
holo    PAE floor B     260   0.521   1.021   1.558   3.533  20.969
cv      pLDDT           144  35.793  78.562  81.778  86.608  95.157
cv      ipTM (mean)     144   0.540   0.632   0.676   0.726   0.856
cv      ipSAE           144   0.000   0.000   0.000   0.096   0.624
cv      fold->design    144   0.553   1.235   1.616   2.570  20.724
cv      placement       144   1.460   4.183  13.515  23.091  56.346
cv      motif           144   0.638   2.637   6.766  19.253  48.611
cv      fold->holo      144   0.361   1.043   1.581   2.403  20.693
cv      placement->holo 144   0.958   3.924  14.194  23.038  57.395
cv      PAE floor B     144   1.300   4.500   9.400  13.900  25.800
```

**No `apo_iptm` column** — dropped after 02, where it read 0.000 for all 560
rows because apo folds one chain. First run without it.

The per-target `ipSAE min B` / `min C` rows appear here and not in 02's table
because the analysis module stopped hardcoding chain ids; 02's CSV has the same
columns and would show them on a re-reduce.

## Gate terms

| term | rejected | sole reason | 02 for comparison |
|---|---|---|---|
| `iptm >= 0.7` | 16 | **0** | 43 rejected, **0** sole |
| `rmsd_binder_on_target <= 5.0` | 106 | 37 | 138 / 42 |
| `rmsd_motif_on_target <= 2.0` | 79 | 10 | 112 / 16 |
| `hotspot_fraction >= 0.4` | 0 | — | 0 |
| `clashes_sidechain <= 3` | 0 | — | 1 |

Placement is bimodal again, with the valley in the same place: p25 2.54,
med 3.80, p75 11.28 (02: 2.77 / 4.52 / 19.61). The 5 Å threshold sits in the
gap in both runs.

## Holo against CV

On the 144 sequences that reached both:

```
metric                 holo_p25  holo_med  cv_p25   cv_med
fold -> design            0.890     1.174   1.235    1.616
placement on target       2.125     2.634   4.183   13.515
motif on target           0.678     0.838   2.637    6.766
ipTM                      0.843     0.888   0.632    0.676
ipSAE min                 0.253     0.492   0.000    0.000
pLDDT binder             78.690    83.037  78.562   81.778
ipSAE min B               0.272     0.505   0.000    0.000
PAE floor B               0.952     1.177   4.500    9.400
```

The same signature as 02 to within a rounding error: the two models reproduce
each other's **fold** (1.17 → 1.62 median) and disagree about **docking**
(2.63 → 13.52). CV ipTM again runs well below holo's — 0.676 against 0.888,
a gap of 0.21 where 02's was 0.22.

## ipSAE at cutoff 10

```
holo (n=260)                       cv (n=144)
placement     n  zeros    med      placement     n  zeros    med
< 5 A       154      2  0.478      < 5 A        50     16  0.058
5-15 A       49      5  0.176      5-15 A       24     15  0.000
> 15 A       57     27  0.010      > 15 A       70     55  0.000
```

Exact zeros overall: holo 34/260 (**13%**, against 02's 22%), cv 86/144
(**60%**, against 02's 57%). `ipsae_min_B` 56%, `ipsae_min_C` 58% — the two
targets behave alike, so `ipsae_min` is not being dragged down by one chain,
as in 02.

At holo the metric tracks placement cleanly for a second run. At CV it is
weaker here than in 02 — the best band medians 0.058 against 02's 0.191, with
16 of 50 at exactly zero.

**The case against gating CV on ipSAE is stronger than in 02.** Of the 28
agreed sequences, **6 have `cv_ipsae_min` of exactly 0.000** — 21%, where 02
had 2 of 16 (12%). A CV ipSAE gate would have discarded a fifth of this run's
confirmed designs.

## Per-spec, same thresholds

```
                 designs  filters  seqs  apo pass  holo pass  agreed
rgd_core              40    36/40   288       179        114      27
rgd_extended          40    29/40   232        81         30       1
```

Compare 02, where both specs passed the filters 40/40 and `rgd_extended`
yielded nothing from 115 holo passes… against `rgd_core`'s. Here the filters
already separate them (36 vs 29), and the gap widens at every later stage:
apo 62% against 35%, and 1 agreed sequence against 27.

**Two runs, 80 designs each, 1 surviving sequence.** `rgd_extended` returning
almost nothing against the bar `rgd_core` clears is now a property of the spec
rather than one run's luck. The thresholds are shared on purpose — the
comparison is why two specs are in one campaign — so this is the result, not a
threshold to relax.

## Against run 02 — the five questions this run was launched to answer

1. **Do `iptm >= 0.7` and `hotspot_fraction >= 0.4` still reject nothing?**
   **Settled, both.** ipTM rejected 16 and was the sole reason for **zero** of
   them; across both runs that is 59 rejections and not one that turned on
   ipTM alone. `hotspot_fraction` fired 0 times in 160 designs.
   `clashes_sidechain` has now rejected 1 design in 160 and is close behind.
2. **Are the pass rates stable?** Mixed. Apo is remarkably so (52.1% → 50.0%).
   Filters moved 87.5% → 81.3% and holo moved 47.3% → **55.4%**, so a single
   run's rate is worth ±5–8 points, not ±1.
3. **Is placement still bimodal with the valley at 5 Å?** **Yes**, in both
   runs, and `rmsd_binder_on_target <= 5.0` remains the workhorse with 37 sole
   rejections.
4. **Does CV still confirm ~12%?** **No — 19.4% here against 11.6%.** The
   agreement rate is not stable, so treat it as a per-run observation. The
   *concentration* is stable: both runs have one backbone supplying 5–6 of the
   survivors.
5. **Does `rgd_extended` fail again?** **Yes**, 1 agreed sequence from 40
   designs, and this time it fails earlier — at the filters.

### What has now earned a place in Notes.md

Only the findings that held across both runs: ipTM at 0.7 does no independent
work at holo, `hotspot_fraction >= 0.4` never fires, placement is bimodal with
the valley near 5 Å, holo and CV reproduce fold while disagreeing on docking,
and ipSAE must not gate CV because confirmed designs score exactly 0.

### What is still a single-run number

The CV agreement rate, every pass rate at better than ±8 points, and the CV
ipSAE band medians. Two runs is enough to retire a gate term that fired zero
times twice; it is not enough to move a threshold.
