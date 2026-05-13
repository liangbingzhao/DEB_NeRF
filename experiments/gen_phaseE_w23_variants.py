"""Phase E — train Path D variants (B/C/D + Raw + HAP) on W2 + W3 to test if
ANY variant can achieve meaningful α_w/α_2 decomposition where Softplus failed.

5 variants × 2 walnut × 2 view = 20 trainings.
All use the (20,60) pair (matches W1's reference comparison setup).

Variants:
  B: alpha+water_init        (water κ_2 init, alpha-direct param)
  C: frac+bone               (bone init, rho_fraction param)
  D: frac+water              (water init, rho_fraction param)
  Raw: alpha+bone, NO constrain_kappa_2  (κ_2 unconstrained)
  HAP: alpha+bone, HAP material init (different bone material)

Bone init κ_2 = compute_pair_calibration(low,high,walnut_scale)  (per-walnut)
Water init κ_2 = (κ_w_low, κ_w_high) = (1.0, 0.2543)  (walnut-independent)
HAP init κ_2 = (μ_HAP[low], μ_HAP[high]) / walnut_scale  (per-walnut)

Output:
  yaml: SAX-NeRF/config/Lineformer/res_256/v3_dual_phys_basis2{,_25view}/walnut_<N>/
  ckpt: experiments/res_256/v3_dual_phys_basis2{,_25view}/walnut_<N>/<expname>/
"""
import argparse
import os
import os.path as osp
import pickle
import sys
import time

REPO = "/ibex/project/c2272/liangbing/cs300_project"
LOGS = f"{REPO}/experiments/logs"
sys.path.insert(0, osp.join(REPO, "experiments"))
from compute_pair_calibration import compute  # noqa: E402

LOW, HIGH = 20, 60
VIEWS = [25, 50]
WALNUTS = ["Walnut_2", "Walnut_3"]

# HAP attenuation values (cm^-1) from W1 yaml comments — walnut-independent.
MU_HAP = {20: 4.17, 60: 0.45}


def get_pair_scale(walnut, low, high):
    sub = walnut.lower() + "/"
    p = osp.join(REPO, "SAX-NeRF/data/res_256/v3_dual_phys",
                 sub, f"walnut_{low}kev_{high}kev_50.pickle")
    return float(pickle.load(open(p, "rb"))["norm"]["scale"])


# -- yaml templates per variant --

YAML_BASE = """exp:
  expname: {EXPNAME}
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
{EXTRA_NETWORK}encoder:
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


SBATCH = """#!/bin/bash
#SBATCH --job-name=m9_phaseE_{TAG}
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

echo "===================================================================="
echo "{HEADER}"
echo "Job ID: ${{SLURM_JOB_ID}}, Node: $(hostname), Start: $(date)"
echo "===================================================================="

python train_mlg_basis2.py --gpu_id 0 --seed 42 --config {YAML_RELPATH}

STATUS=$?
echo "Exit code: ${{STATUS}}, End: $(date)"
exit $STATUS
"""


def variant_kappa_extra(variant, cal, walnut_scale):
    """Returns (κ_2_low_init, κ_2_high_init, extra_yaml_lines, expname_suffix)."""
    if variant == "B":   # alpha + water_init
        return (1.0, cal["kappa_w_high"],
                "  constrain_kappa_2: true\n  parameterization: alpha_direct\n",
                "water_init")
    if variant == "C":   # frac + bone
        return (cal["kappa_2_low_init"], cal["kappa_2_high_init"],
                "  constrain_kappa_2: true\n  parameterization: rho_fraction\n",
                "frac_bone")
    if variant == "D":   # frac + water
        return (1.0, cal["kappa_w_high"],
                "  constrain_kappa_2: true\n  parameterization: rho_fraction\n",
                "frac_water")
    if variant == "Raw":  # alpha + bone, no constrain
        return (cal["kappa_2_low_init"], cal["kappa_2_high_init"],
                "",  # no constrain_kappa_2
                "raw")  # expname uses bare "_basis2_lh10" for raw -- override below
    if variant == "HAP":  # alpha + bone HAP material
        k2_low = MU_HAP[LOW] / walnut_scale
        k2_high = MU_HAP[HIGH] / walnut_scale
        return (k2_low, k2_high,
                "  constrain_kappa_2: true\n",
                "softplus_hap")
    raise ValueError(variant)


def expname_for(variant, low, high, nv):
    """Per W1 naming convention."""
    if variant == "B":   return f"walnut_{low}kev_{high}kev_{nv}_basis2_water_init_lh10"
    if variant == "C":   return f"walnut_{low}kev_{high}kev_{nv}_basis2_frac_bone_lh10"
    if variant == "D":   return f"walnut_{low}kev_{high}kev_{nv}_basis2_frac_water_lh10"
    if variant == "Raw": return f"walnut_{low}kev_{high}kev_{nv}_basis2_lh10"
    if variant == "HAP": return f"walnut_{low}kev_{high}kev_{nv}_basis2_softplus_hap_lh10"


def main():
    ts = int(time.time())
    yamls, sbatches = [], []

    for walnut in WALNUTS:
        sub = walnut.lower() + "/"
        wshort = walnut.replace("Walnut_", "w")
        scale = get_pair_scale(walnut, LOW, HIGH)
        cal = compute(LOW, HIGH, scale)
        print(f"\n=== {walnut} (20,60) shared_scale={scale:.4f} ===")

        for variant in ["B", "C", "D", "Raw", "HAP"]:
            k2l, k2h, extra, suffix = variant_kappa_extra(variant, cal, scale)
            print(f"  {variant} ({suffix}): κ_2_init=({k2l:.4f}, {k2h:.4f})")
            for nv in VIEWS:
                view_dir = "v3_dual_phys_basis2" if nv == 50 else "v3_dual_phys_basis2_25view"
                expname = expname_for(variant, LOW, HIGH, nv)
                expdir = f"{REPO}/experiments/res_256/{view_dir}/{sub}"

                yaml_dir = osp.join(REPO, "SAX-NeRF/config/Lineformer/res_256",
                                    view_dir, walnut.lower())
                os.makedirs(yaml_dir, exist_ok=True)
                yaml_abs = osp.join(yaml_dir, f"{expname}.yaml")
                yaml_rel = osp.relpath(yaml_abs, osp.join(REPO, "SAX-NeRF"))

                with open(yaml_abs, "w") as f:
                    f.write(YAML_BASE.format(
                        EXPNAME=expname, EXPDIR=expdir, SUB=sub,
                        LL=LOW, HH=HIGH, NV=nv,
                        KW_HIGH=cal["kappa_w_high"], K2L=k2l, K2H=k2h,
                        EXTRA_NETWORK=extra,
                    ))

                tag = f"{wshort}_{suffix}_{nv}v"
                sbatch_path = f"{LOGS}/m9_phaseE_{tag}-{ts}.sbatch"
                with open(sbatch_path, "w") as f:
                    f.write(SBATCH.format(
                        TAG=tag, LOGS=LOGS, REPO=REPO,
                        HEADER=f"M9 Phase E -- {walnut} (20,60) Path D variant={variant} ({suffix}) @ {nv}v",
                        YAML_RELPATH=yaml_rel,
                    ))
                yamls.append(yaml_abs)
                sbatches.append(sbatch_path)

    print(f"\nGenerated {len(yamls)} yaml + {len(sbatches)} sbatch")
    print("\nSubmit all with:")
    print(f"  for s in {LOGS}/m9_phaseE_w[23]_*-{ts}.sbatch; do sbatch $s; done")


if __name__ == "__main__":
    main()
