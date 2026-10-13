# alphav_beta3_02 — the first full campaign

**Finished 2026-10-12 14:52 CDT**, 3 h 08 m, app `ap-VKsrrzbsxfRIscIxy2oktu`.
Campaign `motif_scaffold_alphav_beta3`, two specs, `openfold3` weights for
cross-validation, `ipsae.pae_cutoff: 10`.

Scale: 2 specs × 5 batches × 8 = **80 designs**, 8 MPNN sequences each, 5
cross-validation seeds at one diffusion sample per seed.

Written up retrospectively on 2026-10-13 from `summary.csv` / `summary.json`,
which are the run's own artifacts. Numbers reproduce with
`python experiments/metrics_table.py <run>/summary.csv <run>/summary.json`.

## Funnel

```
rfd3                80 designs
backbone_filters    70 / 80    87.5%   (had been assumed 67%, from 3 designs)
mpnn               560 sequences       never the 640 ceiling: only 70 designs passed
apo                292 / 560   52.1%   (had been assumed 50% -- dead on)
holo               138 / 292   47.3%   the gate had never executed before this run
cv                 138 reported
```

Filter rejections: `clashes_backbone` 4, `rg_ratio` 4, `loop_fraction` 3,
`clashes_sidechain` 1, `hotspot_fraction` **0**.

## Result

**16 agreed sequences over 9 designs, every one `rgd_core`.** Cross-validation
confirms holo's placement (≤ 5 Å) and motif (≤ 2 Å) on 16 of 138.

Concentration matters more than the count: `rgd_core_2_model_1` contributes 4
of the 16 and `rgd_core_2_model_0` another 3, so this is at most 9 pieces of
evidence and arguably fewer.

```
rgd_core_2_model_1   4        rgd_core_0_model_1   1
rgd_core_2_model_0   3        rgd_core_1_model_7   1
rgd_core_4_model_2   2        rgd_core_3_model_5   1
rgd_core_4_model_5   2        rgd_core_4_model_3   1
                              rgd_core_4_model_4   1
```

### Best design: `rgd_core_2_model_1_seq4`

| | holo | cv |
|---|---|---|
| placement on target | 1.83 | 2.26 |
| motif on target | 0.66 | 0.84 |
| fold → design | 0.70 | 0.66 |
| ipTM | 0.912 | 0.830 (best 0.89 / worst 0.69) |
| ipSAE min | 0.639 | 0.580 |
| pLDDT binder | 87.98 | 92.52 |
| PAE floor B | 0.93 | 1.70 |

`cv_rmsd_to_holo` 0.46 Å and `cv_rmsd_placement_vs_holo` 1.00 Å, against
campaign medians of 1.72 and 13.66 — the one case where an independent model
reproduced both the fold and the docking. Four of this backbone's eight MPNN
sequences survive everything, which is evidence about the *backbone*.

**Apo nearly rejected it.** `apo_plddt_binder` 72.37 against a threshold of 70,
and `apo_rmsd_to_design` 4.88 Å — its worst number anywhere. Bound, it folds to
0.70 Å of the design. A fold gate at apo would have killed the campaign's best
design, which is why apo stays gated on pLDDT with fold only watched.

## Distributions

```
stage   metric            n     min     p25     med     p75     max
apo     pLDDT           560  44.558  64.759  70.478  76.769  93.129
apo     fold->design    560   0.739   1.643   2.691   6.381  18.161
holo    pLDDT           292  57.766  76.342  80.292  84.422  91.906
holo    ipTM            292   0.627   0.759   0.838   0.899   0.942
holo    ipSAE           292   0.000   0.012   0.218   0.539   0.835
holo    fold->design    292   0.620   1.091   1.720   2.563  11.572
holo    fold apo/holo   292   0.244   1.073   1.594   2.384  11.996
holo    placement       292   0.890   2.773   4.521  19.610  51.716
holo    motif           292   0.374   0.798   1.301   6.433  33.972
cv      pLDDT           138  51.163  75.946  80.541  86.417  93.011
cv      ipTM (mean)     138   0.528   0.634   0.664   0.728   0.878
cv      ipSAE           138   0.000   0.000   0.000   0.150   0.580
cv      fold->design    138   0.664   1.236   1.689   3.195  17.278
cv      placement       138   1.069   3.837  13.096  22.613  55.787
cv      motif           138   0.632   3.112   7.363  19.299  54.990
cv      fold->holo      138   0.449   1.191   1.719   3.270  17.088
cv      placement->holo 138   0.996   4.157  13.661  22.712  57.555
```

This run's CSV still carries `apo_iptm`, all 560 rows reading 0.000 — apo folds
one chain, so there was no interface to score. The column was dropped
afterwards and will be absent from later runs.

## Gate terms

| term | rejected | sole reason | verdict |
|---|---|---|---|
| `iptm >= 0.7` | 43 | **0** | no independent work |
| `rmsd_binder_on_target <= 5.0` | 138 | 42 | the workhorse |
| `rmsd_motif_on_target <= 2.0` | 112 | 16 | real, secondary |
| `hotspot_fraction >= 0.4` | 0 | — | never fired |

Every one of the 43 ipTM rejections also failed both RMSD terms.

Placement is sharply **bimodal** — p25 2.77, median 4.52, p75 19.61 — because
the binder either lands on the epitope or sits tens of Å away, folded correctly
and docked elsewhere. 5.0 sits in the valley, and 2.0 does the same for motif
(p25 0.80, p75 6.43). Both landed well.

## Holo against CV

On the 138 sequences that reached both:

```
metric                 holo_p25  holo_med  cv_p25   cv_med
fold -> design            0.908     1.111   1.236    1.689
placement on target       2.088     2.717   3.837   13.096
motif on target           0.707     0.849   3.112    7.363
ipTM                      0.858     0.898   0.634    0.664
ipSAE min                 0.347     0.531   0.000    0.000
pLDDT binder             80.634    84.229  75.946   80.541
PAE floor B               0.820     1.091   4.600    8.800
```

The models reproduce each other's *fold* and disagree about *docking*. Read the
agreed set as the result, not the disagreement as a shortfall.

**No holo metric predicts CV agreement**, and that is the premise rather than a
finding — if it did, CV would be measuring nothing. Comparing the 16 against
the other 122, every distribution overlaps: ipSAE 0.626 vs 0.528 median, ipTM
0.910 vs 0.894, placement 2.095 vs 2.772, pLDDT 84.2 vs 84.3. An agreed
sequence scored ipSAE 0.040 while non-agreed ones reach 0.726.

So tightening holo to shrink the CV bill trades designs for money:
`holo_ipsae_min >= 0.5` cuts CV to 57% of its cost and loses 6 of the 16;
`placement <= 2.5` cuts it to 45% and loses 5.

## ipSAE at cutoff 10

`zeros` counts values of *exactly* 0.0 — no residue pair under the PAE cutoff,
which is a verdict about the interface rather than missing data.

```
holo (n=292)                       cv (n=138)
placement     n  zeros    med      placement     n  zeros    med     max
< 5 A       154      2  0.511      < 5 A        40      8  0.191   0.580
5-15 A       52     17  0.048      5-15 A       32     22  0.000   0.364
> 15 A       86     45  0.000      > 15 A       66     49  0.000   0.435
```

At holo the metric tracks placement cleanly: no zeros at all among the 138 gate
passers, median 0.532, against 64 zeros and a median of 0.013 among the 154
failures. At CV it does not — sequences placed more than 15 Å out still reach
0.435, above the median of CV's own best bucket, because OpenFold3's PAE floor
medians 8.8 Å against holo's 1.09 and half the distribution sits the wrong side
of the cutoff.

There is no cliff at the cutoff. Of 72 `cv_ipsae_min_B` zeros, 58 have a floor
at or above 10; the other 14 have floors of 7.3–9.9 where a pair or two scrape
under and one direction reports 0.000 against the other's 0.011. Values run
continuously 0.000 → 0.010 → 0.013 and all mean the same thing.

**Two of the 16 agreed sequences score `cv_ipsae_min` of exactly 0.000** —
`rgd_core_4_model_2_seq8` at 2.04 Å placement and `rgd_core_4_model_5_seq4` at
4.22 Å. A CV ipSAE gate would have discarded 12% of the best designs. This is
the measured reason CV ranks on ipSAE and never gates on it.

## Per-spec, same thresholds

```
                 designs  filters  apo pass   holo pass  agreed
rgd_core              40    40/40   217/304     115        16
rgd_extended          40    40/40    75/256      23         0
```

`rgd_extended` returned **zero** survivors against the same bar `rgd_core`
cleared, having passed the filters 40/40. Its backbones differ upstream of
anything the filters test — helix fraction 0.515 against 0.769 median, more
loop — and it shows up immediately in apo pLDDT (66.4 vs 75.3) and holo
placement (8.4 vs 4.1 Å).

That is the result, not a threshold to relax. One bar for every spec is the
whole reason two specs share a campaign.

## Cost shape

Apo 3.1 s/sequence, holo 12.8 s (528 tokens), cross-validation **~80 s per
seed**. CV at 5 seeds is roughly 75% of the campaign; the folding stages are
nearly free. CV seeds is the only lever that materially changes the bill — and
per the agreement finding above, the holo gate cannot be tightened to shrink it
without losing designs.

RFD3 ran ~19 s/design, 8 designs per ~150 s inference batch, ~13 min per spec,
both specs in parallel.

## What this run cannot settle

Nine surviving designs from two parent backbones is not a basis for moving a
threshold. `iptm >= 0.7` and `hotspot_fraction >= 0.4` are shown to do no work
*here*; whether that holds is what [alphav_beta3_03](alphav_beta3_03.md)
exists to check.
