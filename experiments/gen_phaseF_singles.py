"""Phase F — exhaustive single-energy baseline on W2/W3 (10..80 keV).

Goal: support claim "dual-basis vs Total > single-energy vs Total
matched view count" by training a comprehensive single-energy grid.

Already done (Phase D): single 20/60 keV at 25/50v on W2/W3 = 4 each.
NEW (Phase F): 10/30/40/50/70/80 keV at 25/50v on W2/W3
              = 6 energies x 2 views x 2 walnuts = 24 trainings.

Output:
  pickles: SAX-NeRF/data/res_256/v3_phys/walnut_{N}/walnut_{E}kev_{NV}.pickle
  yaml:    SAX-NeRF/config/Lineformer/res_256/v3_phys{,_25view}/walnut_{N}/walnut_{E}kev_{NV}.yaml
  ckpt:    experiments/res_256/v3_phys{,_25view}/walnut_{N}/walnut_{E}kev_{NV}_phys/

Generates:
  - 1 pickle-gen sbatch per walnut (6 new pickles each x 2 walnuts = 2 sbatches)
  - 12 training sbatches per walnut (6 energies x 2 views x 2 walnuts = 24)
"""
import os
import os.path as osp
import pickle
import sys
import time

REPO = "/ibex/project/c2272/liangbing/cs300_project"
LOGS = f"{REPO}/experiments/logs"

NEW_ENERGIES = [10, 30, 40, 50, 70, 80]
VIEWS = [25, 50]
WALNUTS = ["Walnut_2", "Walnut_3"]


def get_2060_scale(walnut):
    sub = walnut.lower() + "/"
    p = osp.join(REPO, "SAX-NeRF/data/res_256/v3_dual_phys",
                 sub, "walnut_20kev_60kev_50.pickle")
    return float(pickle.load(open(p, "rb"))["norm"]["scale"])


# ----------------------------------------------------------- pickle gen sbatch

CONVERT_SBATCH = """#!/bin/bash
#SBATCH --job-name=m9_phaseF_data_{wshort}
#SBATCH --account=conf-neurips-2026.05.15-elhosemh
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output={LOGS}/%x-%j.out
#SBATCH --error={LOGS}/%x-%j.err

cd {REPO}/data_preprocess
source ~/.bashrc
module load cuda/11.7.1
export TORCH_CUDA_ARCH_LIST="8.0"
export LD_LIBRARY_PATH=/sw/rl9g/cuda/11.7.1/lib64:$LD_LIBRARY_PATH
source activate sax_nerf
set -e

WALNUT={walnut}
SCALE={scale:.6f}

echo "##### Phase F single-energy pickle gen for $WALNUT (scale=$SCALE) #####"
echo "Job: ${{SLURM_JOB_ID}}, Node: $(hostname), Start: $(date)"

{commands}

echo "##### Done $(date) #####"
ls -la {REPO}/SAX-NeRF/data/res_256/v3_phys/{subdir}walnut_{{10,30,40,50,70,80}}kev_*.pickle 2>/dev/null
"""


def gen_convert_sbatch(walnut, ts):
    scale = get_2060_scale(walnut)
    wshort = walnut.replace("Walnut_", "w")
    subdir = walnut.lower() + "/"

    cmds = []
    for E in NEW_ENERGIES:
        for NV in VIEWS:
            cmds.append(f"echo '--- {E}Kev n_train={NV} $(date +%H:%M:%S) ---'")
            cmds.append(f"python convert_walnut_phys.py --walnut {walnut} \\")
            cmds.append(f"    --energy {E}Kev --global_scale $SCALE \\")
            cmds.append(f"    --n_train {NV} --n_val 50 --seed 42")
            cmds.append("")
    body = "\n".join(cmds).rstrip()

    out_path = f"{LOGS}/m9_phaseF_data_{wshort}-{ts}.sbatch"
    with open(out_path, "w") as f:
        f.write(CONVERT_SBATCH.format(
            wshort=wshort, LOGS=LOGS, REPO=REPO,
            walnut=walnut, scale=scale, subdir=subdir, commands=body,
        ))
    return out_path, scale


# ----------------------------------------------------------- training yaml/sbatch

SINGLE_YAML = """exp:
  expname: walnut_{ENAME}_phys
  expdir: {EXPDIR}
  datadir: ./data/res_256/v3_phys/{SUB}walnut_{ENAME}.pickle
network:
  net_type: Lineformer
  num_layers: 4
  hidden_dim: 32
  skips: [2]
  out_dim: 1
  last_activation: sigmoid
  bound: 0.3
  line_size: 2
  dim_head: 4
  heads: 8
  num_blocks: 1
encoder:
  encoding: hashgrid
  input_dim: 3
  num_levels: 16
  level_dim: 2
  base_resolution: 16
  log2_hashmap_size: 19
render:
  n_samples: 256
  n_fine: 0
  perturb: True
  raw_noise_std: 0.
  netchunk: 409600
train:
  epoch: 1500
  n_batch: 1
  n_rays: 1024
  lrate: 0.001
  lrate_gamma: 0.1
  lrate_step: 1500
  resume: False
  window_size: [8, 8]
  window_num: 16
log:
  i_eval: 250
  i_save: 500
"""

SBATCH = """#!/bin/bash
#SBATCH --job-name=m9_phaseF_{TAG}
#SBATCH --account=conf-neurips-2026.05.15-elhosemh
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output={LOGS}/%x-%j.out
#SBATCH --error={LOGS}/%x-%j.err

cd {REPO}/SAX-NeRF
source ~/.bashrc
module load cuda/11.7.1
export TORCH_CUDA_ARCH_LIST="8.0"
export LD_LIBRARY_PATH=/sw/rl9g/cuda/11.7.1/lib64:$LD_LIBRARY_PATH
source activate sax_nerf

echo "##### {HEADER} #####"
echo "Job: ${{SLURM_JOB_ID}}, Node: $(hostname), Start: $(date)"

python train_mlg.py --gpu_id 0 --seed 42 --config {YAML_RELPATH}

STATUS=$?
echo "Exit code: ${{STATUS}}, End: $(date)"
exit $STATUS
"""


def gen_train_jobs(walnut, ts):
    sub = walnut.lower() + "/"
    wshort = walnut.replace("Walnut_", "w")
    paths = []

    for E in NEW_ENERGIES:
        for NV in VIEWS:
            view_dir = "v3_phys" if NV == 50 else "v3_phys_25view"
            ename = f"{E}kev_{NV}"
            expname = f"walnut_{ename}_phys"
            expdir = f"{REPO}/experiments/res_256/{view_dir}/{sub}"

            yaml_dir = osp.join(REPO, "SAX-NeRF/config/Lineformer/res_256",
                                view_dir, walnut.lower())
            os.makedirs(yaml_dir, exist_ok=True)
            yaml_abs = osp.join(yaml_dir, f"walnut_{ename}.yaml")
            yaml_rel = osp.relpath(yaml_abs, osp.join(REPO, "SAX-NeRF"))

            with open(yaml_abs, "w") as f:
                f.write(SINGLE_YAML.format(ENAME=ename, EXPDIR=expdir, SUB=sub))

            tag = f"{wshort}_single_{E}_{NV}v"
            sbatch_path = f"{LOGS}/m9_phaseF_{tag}-{ts}.sbatch"
            with open(sbatch_path, "w") as f:
                f.write(SBATCH.format(
                    TAG=tag, LOGS=LOGS, REPO=REPO,
                    HEADER=f"M9 Phase F Single -- {walnut} {E}Kev @ {NV}v",
                    YAML_RELPATH=yaml_rel,
                ))
            paths.append(sbatch_path)
    return paths


def main():
    ts = int(time.time())

    print("=== Pickle gen sbatches ===")
    convert_paths = []
    for w in WALNUTS:
        p, scale = gen_convert_sbatch(w, ts)
        print(f"  {w} (scale={scale:.4f}): {p}")
        convert_paths.append(p)

    print("\n=== Training sbatches ===")
    train_paths = []
    for w in WALNUTS:
        ps = gen_train_jobs(w, ts)
        print(f"  {w}: {len(ps)} jobs")
        train_paths.extend(ps)

    print(f"\nTotal: {len(convert_paths)} convert + {len(train_paths)} train = {len(convert_paths)+len(train_paths)} sbatches")
    print()
    print("Submit (convert first, then training with dependency):")
    for cp in convert_paths:
        print(f"  CONV=$(sbatch --parsable {cp})")
        print(f"  echo conv: $CONV")
    print(f"  for s in {LOGS}/m9_phaseF_w[23]_single_*-{ts}.sbatch; do sbatch --dependency=afterok:$CONV $s; done")
    print("  # NOTE: above uses LAST CONV; better submit per-walnut training with their respective CONV")


if __name__ == "__main__":
    main()
