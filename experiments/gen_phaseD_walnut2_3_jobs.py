"""Generate Phase D training yaml + sbatch for Walnut_2 / Walnut_3.

Replicates Walnut_1's full 5x4 grid on a new walnut:
  - 7 dual pairs x 2 view (25, 50)  = 14 Path D Softplus trainings
  - 2 single energies (20, 60) x 2 view = 4 single-energy trainings
  - 1 Total ceiling x 2 view = 2 ceiling trainings
  Total = 20 trainings / walnut

Per (low, high) pair, kappa values are recomputed using THIS walnut's actual
shared_scale (extracted from the dual pickle norm dict).

Output:
  SAX-NeRF/config/Lineformer/res_256/v3_xxx{,_25view}/walnut_<N>/<expname>.yaml
  experiments/logs/m9_w<N>_<tag>-<ts>.sbatch

Usage:
  python experiments/gen_phaseD_walnut2_3_jobs.py --walnut Walnut_2
  python experiments/gen_phaseD_walnut2_3_jobs.py --walnut Walnut_2 --walnut Walnut_3
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import pickle
import sys
import time

REPO = "/ibex/project/c2272/liangbing/cs300_project"
sys.path.insert(0, osp.join(REPO, "experiments"))
from compute_pair_calibration import compute  # noqa: E402

# 5x4 grid: 7 dual pairs (matches Walnut_1's existing Phase D entries)
PAIRS = [(10, 70), (10, 80), (20, 50), (20, 60), (20, 70), (20, 80), (30, 80)]
VIEWS = [25, 50]
SINGLE_ENERGIES = [20, 60]


# --------------------------------------------------------------------- yaml
DUAL_BASIS2_YAML = """exp:
  expname: walnut_{LL}kev_{HH}kev_{NV}_basis2_softplus_lh10
  expdir: {EXPDIR}
  datadir: ./data/res_256/v3_dual_phys/{SUB}walnut_{LL}kev_{HH}kev_{NV}.pickle
network:
  net_type: Lineformer_basis2
  num_layers: 4
  hidden_dim: 32
  skips: [2]
  out_dim: 1
  last_activation: softplus
  bound: 0.3
  line_size: 2
  dim_head: 4
  heads: 8
  num_blocks: 1
  kappa_w_low: 1.0
  kappa_w_high: {KW_HIGH:.4f}
  kappa_2_low_init: {K2L:.4f}
  kappa_2_high_init: {K2H:.4f}
  constrain_kappa_2: true
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
loss:
  lambda_recon_low: 1.0
  lambda_recon_high: 10.0
log:
  i_eval: 250
  i_save: 500
"""

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
#SBATCH --job-name=m9_{TAG}
#SBATCH --account=conf-neurips-2026.05.15-elhosemh
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/ibex/project/c2272/liangbing/cs300_project/experiments/logs/%x-%j.out
#SBATCH --error=/ibex/project/c2272/liangbing/cs300_project/experiments/logs/%x-%j.err

cd /ibex/project/c2272/liangbing/cs300_project/SAX-NeRF
source ~/.bashrc
module load cuda/11.7.1
export TORCH_CUDA_ARCH_LIST="8.0"
export LD_LIBRARY_PATH=/sw/rl9g/cuda/11.7.1/lib64:$LD_LIBRARY_PATH
source activate sax_nerf

echo "===================================================================="
echo "{HEADER}"
echo "Job ID: ${{SLURM_JOB_ID}}, Node: $(hostname), Start: $(date)"
echo "===================================================================="

python {TRAIN_SCRIPT} --gpu_id 0 --seed 42 --config {YAML_RELPATH}

STATUS=$?
echo "===================================================================="
echo "Exit code: ${{STATUS}}, End: $(date)"
echo "===================================================================="
exit $STATUS
"""


def get_pair_scale(walnut: str, low: int, high: int) -> float:
    """Read shared_scale from this walnut's dual pickle (uses 50v variant)."""
    sub = walnut.lower() + "/"
    p = osp.join(REPO, "SAX-NeRF/data/res_256/v3_dual_phys",
                 sub, f"walnut_{low}kev_{high}kev_50.pickle")
    with open(p, "rb") as f:
        d = pickle.load(f)
    return float(d["norm"]["scale"])


def write_one(yaml_relpath_in_saxnerf: str, yaml_abs_path: str,
              yaml_body: str, sbatch_path: str, sbatch_body: str):
    os.makedirs(osp.dirname(yaml_abs_path), exist_ok=True)
    with open(yaml_abs_path, "w") as f:
        f.write(yaml_body)
    with open(sbatch_path, "w") as f:
        f.write(sbatch_body)


def gen_for_walnut(walnut: str, ts: int) -> tuple[list, list]:
    """Returns (yaml_paths, sbatch_paths)."""
    sub = walnut.lower() + "/"            # walnut_2/
    wshort = walnut.replace("Walnut_", "w")  # w2
    yamls, sbatches = [], []

    # ---- Group A: Path D Softplus dual (7 pair x 2 view = 14) ----
    for low, high in PAIRS:
        cal = compute(low, high, get_pair_scale(walnut, low, high))
        for nv in VIEWS:
            view_dir = "v3_dual_phys_basis2" if nv == 50 else "v3_dual_phys_basis2_25view"
            expname = f"walnut_{low}kev_{high}kev_{nv}_basis2_softplus_lh10"
            tag = f"{wshort}_b2_sp_p{low}_{high}_{nv}v"
            expdir = f"{REPO}/experiments/res_256/{view_dir}/{sub}"

            yaml_dir = osp.join(REPO, "SAX-NeRF/config/Lineformer/res_256",
                                view_dir, walnut.lower())
            yaml_filename = f"{expname}.yaml"
            yaml_abs = osp.join(yaml_dir, yaml_filename)
            yaml_rel = osp.relpath(yaml_abs, osp.join(REPO, "SAX-NeRF"))

            yaml_body = DUAL_BASIS2_YAML.format(
                LL=low, HH=high, NV=nv,
                EXPDIR=expdir, SUB=sub,
                KW_HIGH=cal["kappa_w_high"],
                K2L=cal["kappa_2_low_init"],
                K2H=cal["kappa_2_high_init"],
            )

            sbatch_path = osp.join(REPO, f"experiments/logs/m9_{tag}-{ts}.sbatch")
            header = (f"M9 Phase D Path D Softplus a+bone -- {walnut} ({low},{high}) keV @ {nv}v\n"
                      f"shared_scale={cal['shared_scale']:.4f}, "
                      f"kw_high={cal['kappa_w_high']:.4f}, "
                      f"k2_init=({cal['kappa_2_low_init']:.4f}, {cal['kappa_2_high_init']:.4f})")
            sbatch_body = SBATCH.format(
                TAG=tag, HEADER=header,
                TRAIN_SCRIPT="train_mlg_basis2.py",
                YAML_RELPATH=yaml_rel,
            )

            write_one(yaml_rel, yaml_abs, yaml_body, sbatch_path, sbatch_body)
            yamls.append(yaml_abs); sbatches.append(sbatch_path)

    # ---- Group B: Single energy (20Kev / 60Kev x 25/50) = 4 ----
    for energy in SINGLE_ENERGIES:
        for nv in VIEWS:
            view_dir = "v3_phys" if nv == 50 else "v3_phys_25view"
            ename = f"{energy}kev_{nv}"           # e.g. 20kev_25
            expname = f"walnut_{ename}_phys"
            tag = f"{wshort}_single_{energy}_{nv}v"
            expdir = f"{REPO}/experiments/res_256/{view_dir}/{sub}"

            yaml_dir = osp.join(REPO, "SAX-NeRF/config/Lineformer/res_256",
                                view_dir, walnut.lower())
            yaml_filename = f"walnut_{ename}.yaml"
            yaml_abs = osp.join(yaml_dir, yaml_filename)
            yaml_rel = osp.relpath(yaml_abs, osp.join(REPO, "SAX-NeRF"))

            yaml_body = SINGLE_YAML.format(ENAME=ename, EXPDIR=expdir, SUB=sub)
            sbatch_path = osp.join(REPO, f"experiments/logs/m9_{tag}-{ts}.sbatch")
            header = f"M9 Phase D Single -- {walnut} {energy}Kev @ {nv}v"
            sbatch_body = SBATCH.format(
                TAG=tag, HEADER=header,
                TRAIN_SCRIPT="train_mlg.py",
                YAML_RELPATH=yaml_rel,
            )
            write_one(yaml_rel, yaml_abs, yaml_body, sbatch_path, sbatch_body)
            yamls.append(yaml_abs); sbatches.append(sbatch_path)

    # ---- Group C: Single Total ceiling (25 / 50) = 2 ----
    for nv in VIEWS:
        view_dir = "v3_phys_total"
        ename = f"total_{nv}"
        expname = f"walnut_{ename}_phys"
        tag = f"{wshort}_total_{nv}v"
        expdir = f"{REPO}/experiments/res_256/{view_dir}/{sub}"

        yaml_dir = osp.join(REPO, "SAX-NeRF/config/Lineformer/res_256",
                            view_dir, walnut.lower())
        yaml_filename = f"walnut_{ename}.yaml"
        yaml_abs = osp.join(yaml_dir, yaml_filename)
        yaml_rel = osp.relpath(yaml_abs, osp.join(REPO, "SAX-NeRF"))

        yaml_body = SINGLE_YAML.format(ENAME=ename, EXPDIR=expdir, SUB=sub)
        sbatch_path = osp.join(REPO, f"experiments/logs/m9_{tag}-{ts}.sbatch")
        header = f"M9 Phase D Total ceiling -- {walnut} @ {nv}v"
        sbatch_body = SBATCH.format(
            TAG=tag, HEADER=header,
            TRAIN_SCRIPT="train_mlg.py",
            YAML_RELPATH=yaml_rel,
        )
        write_one(yaml_rel, yaml_abs, yaml_body, sbatch_path, sbatch_body)
        yamls.append(yaml_abs); sbatches.append(sbatch_path)

    return yamls, sbatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walnut", action="append", required=True,
                    choices=["Walnut_2", "Walnut_3"])
    args = ap.parse_args()

    ts = int(time.time())
    all_yamls, all_sbatches = [], []
    for w in args.walnut:
        print(f"\n=== generating for {w} ===")
        ys, ss = gen_for_walnut(w, ts)
        all_yamls.extend(ys); all_sbatches.extend(ss)
        print(f"  yaml: {len(ys)} files")
        print(f"  sbatch: {len(ss)} files")

    print(f"\nTotal: {len(all_yamls)} yaml + {len(all_sbatches)} sbatch")
    print("\nSubmit all with:")
    print(f"  for s in experiments/logs/m9_w*-{ts}.sbatch; do sbatch $s; done")
    print("Or for SMOKE TEST first (cheapest = 25v single 60Kev):")
    smoke_candidates = [s for s in all_sbatches if "single_60_25v" in s]
    if smoke_candidates:
        print(f"  sbatch {smoke_candidates[0]}")


if __name__ == "__main__":
    main()
