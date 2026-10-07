# Sourced from Yehlin Cho ProteinHunter Repository:
import os
import json
import glob
import tempfile
import subprocess
import torch

class LigandMPNNWrapper:
    def __init__(self, run_py="LigandMPNN/run.py", python="python"):
        self.run_py = run_py
        self.python = python

    def run(
        self,
        pdb_path,
        seed=111,
        model_type="protein_mpnn",
        temperature=0.1,
        temperature_per_residue=None,
        chains_to_design=None,
        bias_AA="",
        omit_AA="C",
        extra_args=None,
        fix_unk=True,
        return_logits=False,
    ):
        """
        Unified Ligand/ProteinMPNN runner.

        Args:
            pdb_path (str): Path to PDB or CIF file.
            seed (int): Random seed.
            model_type (str): 'protein_mpnn', 'ligand_mpnn', 'soluble_mpnn', etc.
            temperature (float): Global temperature (sampling).
            temperature_per_residue (dict): Optional per-residue temp override.
            chains_to_design (str or list): e.g. 'A' or ['A','B'].
            bias_AA (str): Bias residues, e.g. "DE".
            omit_AA (str): Omit residues, e.g. "C".
            extra_args (dict): Additional command-line args.
            fix_unk (bool): Replace 'UNK' with 'GLY'.
            return_logits (bool): If True, return (S, logits) tensors.

        Returns:
            list[str] or (list[str], logits): Generated sequences or model outputs.
        """
        extra_args = dict(extra_args or {})

        with tempfile.TemporaryDirectory() as tmpdir:
            out_folder = tmpdir

            # --- Preprocess PDB ---
            pdb_copy = os.path.join(tmpdir, os.path.basename(pdb_path))
            with open(pdb_path, "r") as fin, open(pdb_copy, "w") as fout:
                for line in fin:
                    fout.write(line.replace("UNK", "GLY") if fix_unk else line)

            # --- Handle per-residue temperature ---
            temp_json_path = None
            if temperature_per_residue:
                temp_json_path = os.path.join(tmpdir, "temperature_per_residue.json")
                with open(temp_json_path, "w") as f:
                    json.dump(temperature_per_residue, f)

            # --- Build base command ---
            cmd = [
                self.python, self.run_py,
                "--seed", str(seed),
                "--pdb_path", pdb_copy,
                "--out_folder", out_folder,
                "--model_type", model_type,
                "--temperature", str(temperature)
            ]

            # --- Model checkpoint handling ---
            run_py_path = os.path.abspath(self.run_py)
            BASE_DIR = os.path.dirname(run_py_path)
            MODEL_DIR = os.path.join(BASE_DIR, "model_params")

            if model_type == "protein_mpnn":
                cmd += ["--checkpoint_protein_mpnn", os.path.join(MODEL_DIR, "proteinmpnn_v_48_020.pt")]
            elif model_type == "ligand_mpnn":
                cmd += ["--checkpoint_ligand_mpnn", os.path.join(MODEL_DIR, "ligandmpnn_v_32_010_25.pt")]
            elif model_type == "soluble_mpnn":
                cmd += ["--checkpoint_soluble_mpnn", os.path.join(MODEL_DIR, "solublempnn_v_48_020.pt")]

            # --- Add AA control options ---
            if omit_AA:
                cmd += ["--omit_AA", omit_AA]
            if bias_AA:
                cmd += ["--bias_AA", bias_AA]
            if return_logits:
                cmd += ["--return_logits"]
            if chains_to_design:
                if isinstance(chains_to_design, (list, tuple)):
                    chains_to_design = "".join(chains_to_design)
                cmd += ["--chains_to_design", chains_to_design]
            if temp_json_path:
                cmd += ["--temperature_per_residue", temp_json_path]

            # --- Add any extra CLI arguments ---
            for k, v in extra_args.items():
                cmd += [k, str(v)]

            # --- Run subprocess safely ---
            result = subprocess.run(cmd, capture_output=True, text=True)

            # --- Handle logits output ---
            if return_logits:
                stdout_lines = result.stdout.strip().split("\n")
                json_str = None
                for line in reversed(stdout_lines):
                    if line.strip().startswith("{"):
                        json_str = line.strip()
                        break
                if json_str is None:
                    raise RuntimeError("Could not find JSON output in stdout")

                try:
                    output = json.loads(json_str)
                    S = torch.tensor(output["S"][0])
                    log_probs = torch.tensor(output["log_probs"][0])
                    logits = torch.softmax(log_probs, dim=-1)
                    return S, logits
                except Exception as e:
                    raise RuntimeError(f"Failed to parse logits JSON: {e}")

            # --- Handle normal sequence generation ---
            if result.returncode != 0:
                raise RuntimeError(
                    f"LigandMPNN failed with code {result.returncode}\nSTDERR:\n{result.stderr}"
                )

            fasta_files = glob.glob(os.path.join(out_folder, "seqs", "*.fa"))
            if not fasta_files:
                raise RuntimeError("No FASTA found in output folder.")
            fasta = fasta_files[0]

            seqs = []
            with open(fasta) as f:
                for line in f:
                    if not line.startswith(">"):
                        seqs.append(line.strip())

            return seqs[1:], None
