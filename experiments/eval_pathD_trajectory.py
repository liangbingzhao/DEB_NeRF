"""Path D vs-Total trajectory: PSNR/SSIM at each saved epoch.

Hypothesis (direction B): late-epoch overfitting in 50v makes ρ_total drift away
from Total reference. Earlier epochs may have higher SSIM-vs-Total despite lower
per-energy PSNR.

Output:
  - experiments/eval_M9_total/pathD_trajectory.csv
  - experiments/eval_M9_total/pathD_trajectory.png
"""
import os
import os.path as osp
import sys
import glob
import pickle
import numpy as np

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.join(REPO_ROOT, "SAX-NeRF"))

from src.utils.util import get_psnr_3d, get_ssim_3d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def normalize(v):
    v = v.astype(np.float32)
    return (v - v.min()) / max(v.max() - v.min(), 1e-8)


def load_total_ref():
    p = osp.join(REPO_ROOT, "SAX-NeRF/data/res_256/v3_phys/walnut_total_ref.pickle")
    with open(p, "rb") as f:
        return pickle.load(f)["image"].astype(np.float32)


def parse_kappa(stats_path):
    """Read κ_2 values from stats.txt (logged as 'kappa_2_low: <val>' etc.)"""
    out = {"kappa_2_low": None, "kappa_2_high": None, "psnr_3d_avg": None}
    try:
        with open(stats_path) as f:
            for line in f:
                for k in out:
                    if line.startswith(k + ":"):
                        out[k] = float(line.split(":")[1].strip())
    except Exception:
        pass
    return out


def trajectory_for_run(run_glob):
    """For one Path D run, return list of dicts (epoch, psnr_vs_total, ssim_vs_total, etc.)"""
    run_dirs = sorted(glob.glob(osp.join(REPO_ROOT, run_glob, "*")))
    if not run_dirs:
        return []
    run_dir = run_dirs[-1]  # most recent timestamp
    eval_dirs = sorted(glob.glob(osp.join(run_dir, "eval/epoch_*")))
    total = load_total_ref()
    ref_norm = normalize(total)

    rows = []
    for ed in eval_dirs:
        epoch_str = osp.basename(ed).replace("epoch_", "")
        try:
            epoch = int(epoch_str)
        except ValueError:
            continue
        rho_p = osp.join(ed, "rho_total.npy")
        if not osp.exists(rho_p):
            continue
        rho = np.load(rho_p)
        rho_norm = normalize(rho)
        psnr = float(get_psnr_3d(rho_norm, ref_norm, PIXEL_MAX=1.0))
        ssim = float(get_ssim_3d(rho_norm, ref_norm, PIXEL_MAX=1.0))

        stats = parse_kappa(osp.join(ed, "stats.txt"))
        rows.append({
            "epoch": epoch,
            "psnr_vs_total": psnr,
            "ssim_vs_total": ssim,
            "psnr_3d_avg": stats["psnr_3d_avg"],
            "kappa_2_low": stats["kappa_2_low"],
            "kappa_2_high": stats["kappa_2_high"],
        })
        print(f"  ep{epoch:5d}: vs-Total PSNR={psnr:.3f}, SSIM={ssim:.4f}, "
              f"κ_2=({stats['kappa_2_low']:.4f}, {stats['kappa_2_high']:.4f}), "
              f"avg_psnr={stats['psnr_3d_avg']:.2f}")
    return rows


def main():
    out_dir = osp.join(REPO_ROOT, "experiments/eval_M9_total")
    os.makedirs(out_dir, exist_ok=True)

    runs = [
        ("Path D 25v (raw)",      "experiments/res_256/v3_dual_phys_basis2_25view/walnut_20kev_60kev_25_basis2_lh10"),
        ("Path D 50v (raw)",      "experiments/res_256/v3_dual_phys_basis2/walnut_20kev_60kev_50_basis2_lh10"),
        ("Path D 25v (softplus)", "experiments/res_256/v3_dual_phys_basis2_25view/walnut_20kev_60kev_25_basis2_softplus_lh10"),
        ("Path D 50v (softplus)", "experiments/res_256/v3_dual_phys_basis2/walnut_20kev_60kev_50_basis2_softplus_lh10"),
    ]
    all_traj = {}
    for label, path in runs:
        print(f"\n=== {label} ===")
        all_traj[label] = trajectory_for_run(path)

    traj_25 = all_traj["Path D 25v (raw)"]
    traj_50 = all_traj["Path D 50v (raw)"]
    traj_25_sp = all_traj["Path D 25v (softplus)"]
    traj_50_sp = all_traj["Path D 50v (softplus)"]

    # Save CSV
    csv_p = osp.join(out_dir, "pathD_trajectory.csv")
    with open(csv_p, "w") as f:
        f.write("variant,view,epoch,psnr_vs_total,ssim_vs_total,psnr_3d_avg,kappa_2_low,kappa_2_high\n")
        for label, traj in all_traj.items():
            variant = "softplus" if "softplus" in label else "raw"
            view = "25" if "25v" in label else "50"
            for r in traj:
                f.write(f"{variant},{view},{r['epoch']},{r['psnr_vs_total']:.4f},{r['ssim_vs_total']:.4f},"
                        f"{r['psnr_3d_avg']:.4f},{r['kappa_2_low']:.4f},{r['kappa_2_high']:.4f}\n")
    print(f"\nSaved: {csv_p}")

    # Plot 4 panels: PSNR-vs-Total, SSIM-vs-Total, κ_2_high, psnr_3d_avg vs epoch
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    style_map = {
        "Path D 25v (raw)":      ("o-", "C0"),
        "Path D 50v (raw)":      ("s-", "C1"),
        "Path D 25v (softplus)": ("o--", "C2"),
        "Path D 50v (softplus)": ("s--", "C3"),
    }

    def plot_metric(ax, key, ylabel):
        for label, traj in all_traj.items():
            if not traj:
                continue
            style, color = style_map[label]
            ax.plot([r["epoch"] for r in traj], [r[key] for r in traj],
                    style, label=label, color=color)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plot_metric(axes[0, 0], "psnr_vs_total", "PSNR vs Total")
    axes[0, 0].set_title("vs-Total PSNR (after min-max norm)")

    plot_metric(axes[0, 1], "ssim_vs_total", "SSIM vs Total")
    axes[0, 1].set_title("vs-Total SSIM (after min-max norm)")

    plot_metric(axes[1, 0], "kappa_2_high", "κ_2_high (learned)")
    axes[1, 0].set_title("κ_2_high evolution (init=0.0855; physics requires ≥ 0)")
    axes[1, 0].axhline(0, color="r", linestyle="--", alpha=0.5, label="physics floor")

    plot_metric(axes[1, 1], "psnr_3d_avg", "PSNR_3d_avg")
    axes[1, 1].set_title("Per-energy avg PSNR (training metric)")

    fig.suptitle("Path D vs-Total trajectory: raw vs Softplus(κ_2) parameterization",
                 fontsize=11)
    plt.tight_layout()
    out_png = osp.join(out_dir, "pathD_trajectory.png")
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
