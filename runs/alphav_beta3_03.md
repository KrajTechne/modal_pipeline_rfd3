# alphav_beta3_03 — the second full campaign

**Launched 2026-10-13 12:58 CDT**, detached, app `ap-X9kbLGAtJYQItM6bWhm8bQ`,
orchestrator call `fc-01M4V6M3PE0H7A7N1740P3QKKK`. Expect ~3 h 10 m from then,
roughly three quarters of it cross-validation.

**Status: running.** Sections below are filled from `summary.csv` /
`summary.json` once `reduce` has written them — not from the logs, which say
nothing the artifacts do not.

## What it is for

**Everything but `run_name` is identical to
[alphav_beta3_02](alphav_beta3_02.md).** That is the point: a second
independent sample under the same conditions. `rfdiffusion3.seed` is null for a
full campaign, so sampling is unseeded and the 80 backbones are new rather than
a replay (`app.py:384` records this as verified).

Run 02 produced numbers for three rates that had never been measured, and
showed two gate terms doing no work. None of that can move a threshold on its
own, because 02's result was 16 sequences collapsing into 9 designs from two
parent backbones. This run asks whether those findings are properties of the
pipeline or of one sample:

1. **Do `iptm >= 0.7` and `hotspot_fraction >= 0.4` still reject nothing?**
   In 02, ipTM rejected 43 holo rows and was the sole reason for **zero** of
   them; `hotspot_fraction` fired not once in 80 designs. Two runs agreeing
   would retire both terms.
2. **Are the pass rates stable?** 02 gave filters 87.5%, apo 52.1%, holo 47.3%.
   The apo figure matched its assumed 50% exactly, which is either a good
   threshold or a coincidence.
3. **Is placement still bimodal with the valley at 5 Å?** That is what makes
   `rmsd_binder_on_target <= 5.0` defensible rather than lucky.
4. **Does CV still confirm roughly 12% of holo passes?** And does the agreed
   set still concentrate into a few backbones?
5. **Does `rgd_extended` fail again?** Zero survivors from 40 designs in 02,
   against the same bar `rgd_core` cleared. Once is a result; twice is a
   property of the spec.

A dry run at this config passed all six stages before launch
(`alphav_beta3_03-dryrun`, 4 sequences cross-validated).

## Funnel

_pending_

## Result

_pending_

## Distributions

_pending_

## Gate terms

_pending_

## Holo against CV

_pending_

## ipSAE at cutoff 10

_pending_

## Per-spec, same thresholds

_pending_

## Against run 02

_pending — the comparison this run exists for._
