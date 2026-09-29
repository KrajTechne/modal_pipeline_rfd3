---
name: RFD3 Modal pipeline
overview: "Build a new Modal pipeline from scratch for design: one campaign YAML drives RFD3, backbone filters, SolubleMPNN, structure prediction, then cross validation structure prediction with five seeds, and ranks by cross-model ipSAE."
---

# RFD3 pipeline on Modal

One campaign YAML is the source of truth. Modal volume `rfd3_pipeline` holds all data. Design steps are separate Modal apps the orchestrator calls in order. Make this detachable so the entire pipeline can run without local computer. Also if it crashes allow the ability to restart at a given step.

```
flowchart
  yaml[Campaign YAML]
  rfd3[RFD3 backbones]
  bb[Backbone filters]
  mpnn[SolubleMPNN]
  apo[binder only structure prediction]
  holo[complex structure prediction]
  cv[cross-validation x5 seeds default]
  rank[Rank and label clusters]
  yaml --> rfd3 --> bb --> mpnn --> apo --> holo --> cv --> rank
```

Install and build images from these repositories
RFD3: https://github.com/RosettaCommons/foundry/tree/production/models/rfd3
MPNN: https://github.com/dauparas/LigandMPNN
ESMFold2, Boltz2, AF3: https://github.com/anthropics/uplifting-biomolecular-modeling. This repo has many other structure prediction models we may want to add later


## Layout
file structure: `campaign_name/run_name_01/`

advise on modal apps to run and how to orchestrate

Weights live on a separate Modal volume (RFD3 checkpoint, ESMFold2, AF3, boltz2). No Modal secrets.

## Campaign YAML

Layout example:

general:
  chain_binder: A
  chain_targets: B,C
  name_design_campaign: tpo
  run_name: tpo_01
rfdiffusion3:
  gpu: A100-80GB
  contigs:
    - 120-220,/0,B1-465,/0,C1-465
    - 120-220,/0,B1-465,/0,C1-465
    - 120-220,/0,B1-465,/0,C1-465
    - 120-220,/0,B1-465,/0,C1-465
    - 120-220,/0,B1-465,/0,C1-465
  unindex:
    - A13-25,A53-60,A125,A141-149,A86,A106-111,/0,/0
    - A13-25,A53-60,A125,A141-153,A86,A106-111,/0,/0
    - A13-25,A53-60,A125,A141-149,A106-111,/0,/0
    - A13-25,A53-60,A141-149,A106-111,/0,/0
    - A13-25,A141-149,A106-111,/0,/0
  hotspots: B77,B79,B80,B101,B103,B139,C74,C77,C79,C80,C135,C139
  infer_ori_strategy: hotspots
  length: 1050-1150 # binder chain only. need to add target chain lengths
  is_non_loopy: true
  num_timesteps: 200
  num_batches: 8
  num_designs: 8
  path_input_structure: in/rfdiffusion/tpo_alphafold.pdb
  threshold_interresidue_clashes_sidechain: 3
  threshold_interresidue_clashes_backbone: 0
  threshold_loop_fraction: 0.5
  threshold_radius_gyration_multiplier: 1.1
mpnn:
  gpu: L40S
  mpnn_sampling_temp: 0.1
  num_seqs: 8
  rm_aa: C
  use_solubleMPNN: true
refolding:
  gpu: A100-80GB
  model: esmfold2
  mode: fast
  target_template: false
  n_gpu: 4
  threshold_plddt_binder: 70
  threshold_rmsd_apo_holo: 4.0
  threshold_rmsd_motif: 2.0
  threshold_hotspot_contacts: 0.4
  threshold_ipTM: 0.3
cross-validation:
  gpu: A100-80GB
  model: af3
  mode: exact
  target_template: false
  n_gpu: 4
  num_seeds: 5

## data storage and tracking
Create a summary.csv that stores all designed proteins and each metric computed for all steps
summary.csv is one row per MPNN sequence. Backbones that die before MPNN still get a row with empty sequence columns.
avoid concurrent appends to this file.

Create a summary.json file that tracks number of designs in each step and success and fail. has separate keys for each metric and how many failed each metric.
Example
{
  "campaign": "tpo",
  "run": "tpo_01",
  "n_specs": 5,
  "steps": {
    "rfd3": {"n_in": 5, "n_designs": 320, "n_fail": 0},
    "backbone_filters": {
      "n_in": 320,
      "n_pass": 0,
      "n_fail": 0,
      "metrics": {
        "interresidue_clashes_backbone": {"rule": "<= 0", "n_fail": 0},
        "interresidue_clashes_sidechain": {"rule": "<= 3", "n_fail": 0},
        "loop_fraction": {"rule": "<= 0.5", "n_fail": 0},
        "rg_ratio": {"rule": "<= 1.1", "n_fail": 0},
        "hotspot_contact_fraction": {"rule": ">= 0.4", "n_fail": 0}
      }
    },
    "mpnn": {"n_in": 0, "n_sequences": 0, "n_fail": 0},
    "apo": {
      "n_in": 0,
      "n_pass": 0,
      "n_fail": 0,
      "metrics": {
        "plddt_binder": {"rule": ">= 70", "n_fail": 0}
      }
    },
    "holo": {
      "n_in": 0,
      "n_pass": 0,
      "n_fail": 0,
      "metrics": {
        "plddt_binder": {"rule": ">= 70", "n_fail": 0},
        "rmsd_motif_on_target": {"rule": "<= 2.0", "n_fail": 0},
        "rmsd_binder_apo_holo": {"rule": "<= 4.0", "n_fail": 0},
        "iptm": {"rule": ">= 0.3", "n_fail": 0}
      }
    },
    "cross_validation": {
      "n_in": 0,
      "n_pass": 0,
      "n_fail": 0,
      "num_seeds": 5,
      "metrics": {
        "plddt_binder": {"rule": ">= 70", "n_fail": 0},
        "rmsd_motif_on_target": {"rule": "<= 2.0", "n_fail": 0},
        "rmsd_binder_apo_holo": {"rule": "<= 4.0", "n_fail": 0}
      }
    }
  }
}


## RFD3:

Multiple specs can be provided to write to one RFD3 yaml.

One RFD3 key per index: tpo_spec_00, and so on.
Copy shared fields onto every key: input, select_hotspots, infer_ori_strategy, is_non_loopy, ligand if set, dialect: 2.
Pair contigs[i] with unindex[i].
If unindex is omitted, every spec is contig-only.
Unindex residues must exist in the input and must not overlap that spec’s contig.

Command line options for rfd3.
`num_batches: 8` and `num_designs: 8` become RFD3 `--n-batches 8` and diffusion batch size 8 (64 backbones per spec before filters). Sampler overrides from the Foundry binder examples: `inference_sampler.step_scale=3`, `inference_sampler.gamma_0=0.2`. Also add temperature and number of steps.

in general in the yaml file, `chain_binder: A` and `chain_targets: B,C` are output chain IDs. Extract motif positions from rfd3 json file associated with each cif.gz file. This is the "diffused_index_map" key. In that section the input pdb motif residues will be on the left (key) and the motif residue positions in the output pdb will be on the right (value). 
Example: 
"diffused_index_map": {
  "A13": "A125",
  ...
}
The binder chain is A so only consider positions in the output pdb that start with A. A125 is one of the motif residues in the output pdb and in the original input structure this residue is at position 13. There will also be mapping for the target chains (B,C) but you can ignore them. Store this mapping for MPNN and motif rmsd in later steps.

hotspot positions can change if the chains are reordered or do not start at position 1 in the input file. The output structure will start from 1 and order the chains based on the contig. Use the contig string from the YAML to deterministically calculate the new indices and chain (based on contig position)


## RFD3 filters

look in each respective json file for each design. same name as structure file

Applied to the binder chain before MPNN. Fail reasons go into summary json. Threshold for filters are defined in yaml, defaults are given here. 

Need to calculate Rg since number in json is for the complex. use biotite to calculate this.

- Backbone inter-residue clashes `<= 0`
- Side-chain inter-residue clashes `<= 3` (soft relative to backbone; MPNN replaces side chains)
- Loop fraction `<= 0.5`
- Binder Rg from biotite on chain A: `Rg <= 1.1 * 2.38 * N^0.365`. Store `rg / rg_expected`. Exponent stays `0.365`. multiplier in this example is 1.1 and is defined in yaml. N is number of residues in binder chain

Also record hotspot contacts to hotspots in yaml. Fail if less than threshold_hotspot_contacts of hotspots have zero contacts within 4.5 Å. Interface contacts are added to main csv.

## MPNN:
Only structures generated by RFD3 that pass backbone filters are used. 
use the yaml definitions for input.
SolubleMPNN, temperature `0.1`, `num_sequences: 8`, `omit_AA: C`, fixed residues from `diffused_index_map`. 
No filter at this step

## Refolding metrics
Every sequence from MPNN is refolded.

ipSAE is calculated using this method. https://pypi.org/project/ipsae/ or https://github.com/DunbrackLab/IPSAE

For alignment of designed structure vs refolded structure. CA atoms only. Use biotite.structure.superimpose

- `rmsd_motif_on_target`: align on target chains for refolded and rfd3 designed structures. Calculate rmsd for motif residues only. see mapping from earlier. filter on less than threshold_rmsd_motif
- `rmsd_binder_on_target`: align on target chains for refolded and rfd3 designed structures. Calculate rmsd for binder residues only.
- `rmsd_binder_apo_holo`: align on target chains between apo and holo predictions and calculate rmsd for binder residues only.

## Generate MSA for targets. cache for all predictions
use https://github.com/sokrypton/ColabFold/tree/main/colabfold/mmseqs. define how you are going to implement this to generate MSA files. look at each folding model for what the input msa requirements are. Make sure MSA files generated are compatible with each model used in this pipeline. Store in campaign_name/msa_cache/. Should need to do this once since the target chain sequences do not change.

## Step 1 Refolding - Binder only prediction (apo)
Choose model based on yaml
mode is defined in the anthropics repo. selected by user in the yaml
n_gpu should define the number of simultaneous predictions. one gpu per structure. except in the case of mode: big, which splits the prediction accross multiple gpus. that should be default 2 and not the same setting as n_gpu.

Binder is single-sequence and not templated

After running binder only (apo). Pass only if binder pLDDT >= threshold_plddt_binder

## Step 2 Refolding - Complex prediction (holo)
same model as binder only in step 1
Run passing designs from apo (step 1) with complex (holo). 

Binder is single-sequence and not templated. Always use MSA for target chains. Option to use input structure as template for target chains and not binder. One note is that esmfold2 doesnt allow for templates at the moment.

Pass if binder pLDDT >= threshold_plddt_binder, 
rmsd_motif_on_target <= threshold_rmsd_motif
rmsd_binder_apo_holo <= threshold_rmsd_apo_holo
ipTM >= threshold_ipTM

Record complex ipSAE (`ipSAE_min` of the two directions).

## Cross validation with another model
Choose model based on yaml
Only predict designs that passed the refolding step 2.

mode is defined in the anthropics repo. selected by user in the yaml
Predict binder complex (holo) with another structure prediction model.

5 seeds (or number in yaml). Binder is single-sequence and not templated. Always use MSA for target chains. Use MSA from earlier steps. These are files stored from earlier predictions.
Option to use input structure as template for target chains and not binder. No gating for cross validation run. 
For each seed prediction store ipSAE_min_"target_chain" = min(binder_to_target, target_to_binder). If there is more than one target also store the max_ipSAE_min.
ipsae pip install calculates both binder_to_target, target_to_binder. Dunbrack ipSAE (PAE cutoff 10, distance cutoff 10). 
Across seeds store mean, max.
Also calculate rmsd defined above

Create a column in summary.csv that contains passes_filters (true or false) and add these filters from this step. It will have passed all other filters if it got to this step
Pass if binder pLDDT >= threshold_plddt_binder, 
rmsd_motif_on_target <= threshold_rmsd_motif
rmsd_binder_apo_holo <= threshold_rmsd_apo_holo

## Rank and report

Sort by ipSAE for refolded and coss-validation. 
Cluster backbones with US-align TM-score and write a cluster id on every final structure.


## not in pipeline, test validation
Run structure prediction on the input pdb for each model with and without target templating and different seeds.
write a separate script compared to pipeline that takes input structure, models to use, seeds. then take results and calculate rmsd to original structure of whole complex aligned, target chains only aligned, and binder only aligned. Put that in a csv to user can select which model and parameters to use for pipeline.