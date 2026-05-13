"""Phase C data-generation sbatch generator.

Produces one sbatch per walnut that builds the complete 21-pickle set
(matches Walnut_1's existing 5x4 grid + ceiling + ref + single-energy):

  Phase 1: dual (20,60) at 25/50v          -> 2 jobs (gives shared_scale)
  Phase 2: single 20Kev/60Kev at 25/50v    -> 4 jobs (uses shared_scale)
  Phase 3: total ref + total 25/50         -> 3 jobs (minmax, no scale)
  Phase 4: 6 remaining dual pairs at 25/50 -> 12 jobs

Output goes to v3_phys/walnut_{N}/ and v3_dual_phys/walnut_{N}/ subdirs
(option Z subdir layout, walnut_2/3 only, walnut_1 unchanged).

Usage:
  python gen_phaseC_data_jobs.py --walnut Walnut_2
  python gen_phaseC_data_jobs.py --walnut Walnut_2 --walnut Walnut_3
"""
import argparse
import os
import time

REPO = "/ibex/project/c2272/liangbing/cs300_project"
LOGS = f"{REPO}/experiments/logs"

# 6 dual pairs beyond (20,60). Together with (20,60) -> 7 pair x 2 view = 14 dual.
EXTRA_PAIRS = [(10, 70), (10, 80), (20, 50), (20, 70), (20, 80), (30, 80)]

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=m9_pickles_{walnut_short}
#SBATCH --account=conf-neurips-2026.05.15-elhosemh
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/ibex/project/c2272/liangbing/cs300_project/experiments/logs/%x-%j.out
#SBATCH --error=/ibex/project/c2272/liangbing/cs300_project/experiments/logs/%x-%j.err

cd /ibex/project/c2272/liangbing/cs300_project/data_preprocess
source ~/.bashrc
module load cuda/11.7.1
export TORCH_CUDA_ARCH_LIST="8.0"
export LD_LIBRARY_PATH=/sw/rl9g/cuda/11.7.1/lib64:$LD_LIBRARY_PATH
source activate sax_nerf
# Strict mode AFTER env setup (bashrc/module may have unbound vars)
set -e

WALNUT={walnut}
SUBDIR={subdir}    # e.g. walnut_2/

echo "===================================================================="
echo "M9 Phase C - 21-pickle set for $WALNUT"
echo "Job ID: ${{SLURM_JOB_ID}}, Node: $(hostname), Start: $(date)"
echo "===================================================================="

# ---------- Phase 1: dual (20,60) -> shared_scale ----------
echo ""
echo "##### Phase 1: dual (20,60) #####"
for NV in 25 50; do
    echo "--- (20,60) n_train=$NV $(date '+%H:%M:%S') ---"
    python convert_walnut_dual_phys.py --walnut $WALNUT \\
        --low_energy 20Kev --high_energy 60Kev \\
        --n_train $NV --n_val 50 --seed 42
done

# ---------- extract shared_scale from this walnut's (20,60) 50v ----------
SCALE=$(python -c "
import pickle
with open('../SAX-NeRF/data/res_256/v3_dual_phys/${{SUBDIR}}walnut_20kev_60kev_50.pickle','rb') as f:
    d = pickle.load(f)
print(f'{{d[\\"norm\\"][\\"scale\\"]:.6f}}')
")
echo ""
echo "##### $WALNUT (20,60) shared_scale = $SCALE #####"

# ---------- Phase 2: single 20Kev / 60Kev with shared_scale ----------
echo ""
echo "##### Phase 2: single 20/60Kev #####"
for E in 20Kev 60Kev; do
    for NV in 25 50; do
        echo "--- $E n_train=$NV $(date '+%H:%M:%S') ---"
        python convert_walnut_phys.py --walnut $WALNUT \\
            --energy $E --global_scale $SCALE \\
            --n_train $NV --n_val 50 --seed 42
    done
done

# ---------- Phase 3: Total (minmax, no scale) ----------
echo ""
echo "##### Phase 3: Total ref + total 25/50 #####"
echo "--- total_ref $(date '+%H:%M:%S') ---"
python convert_walnut_total.py --walnut $WALNUT --vol_size 256 256 256
for NV in 25 50; do
    echo "--- total_$NV $(date '+%H:%M:%S') ---"
    python convert_walnut_total_with_proj.py --walnut $WALNUT --n_train $NV --n_val 50 --seed 42
done

# ---------- Phase 4: 6 remaining dual pairs ----------
echo ""
echo "##### Phase 4: 6 extra dual pairs #####"
{phase4_block}

echo ""
echo "===================================================================="
echo "Done at $(date)"
echo "Generated pickles:"
ls -la ../SAX-NeRF/data/res_256/v3_phys/${{SUBDIR}}walnut_*.pickle 2>/dev/null
ls -la ../SAX-NeRF/data/res_256/v3_dual_phys/${{SUBDIR}}walnut_*.pickle 2>/dev/null
echo "===================================================================="
"""


def phase4_block():
    lines = []
    for low, high in EXTRA_PAIRS:
        lines.append(f"for NV in 25 50; do")
        lines.append(f"    echo \"--- ({low},{high}) n_train=$NV $(date '+%H:%M:%S') ---\"")
        lines.append(f"    python convert_walnut_dual_phys.py --walnut $WALNUT \\")
        lines.append(f"        --low_energy {low}Kev --high_energy {high}Kev \\")
        lines.append(f"        --n_train $NV --n_val 50 --seed 42")
        lines.append(f"done")
        lines.append("")
    return "\n".join(lines).rstrip()


def make_sbatch(walnut, ts):
    """Walnut: 'Walnut_2' -> walnut_short='W2', subdir='walnut_2/'."""
    walnut_short = walnut.replace("Walnut_", "W")
    subdir = walnut.lower() + "/"
    body = SBATCH_TEMPLATE.format(
        walnut_short=walnut_short,
        walnut=walnut,
        subdir=subdir,
        phase4_block=phase4_block(),
    )
    out_path = f"{LOGS}/m9_pickles_{walnut_short}-{ts}.sbatch"
    with open(out_path, "w") as f:
        f.write(body)
    print(f"Wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walnut", action="append", required=True,
                    choices=["Walnut_2", "Walnut_3"],
                    help="Repeatable: --walnut Walnut_2 --walnut Walnut_3")
    args = ap.parse_args()

    ts = int(time.time())
    paths = [make_sbatch(w, ts) for w in args.walnut]
    print()
    print("Submit with:")
    for p in paths:
        print(f"  sbatch {p}")


if __name__ == "__main__":
    main()
