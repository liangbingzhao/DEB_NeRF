"""M9 Total-reference comparison: PSNR/SSIM of all methods' single-field outputs vs Total volume.

For each method, the "general field" candidate is:
  - Single energy (any): image_pred.npy (only one field per training)
  - Path B (singlefield):  rho_pred.npy (the actual single-field deliverable)
  - Dual-head / M4.5-a:    image_pred_low.npy (use 20keV view as the chosen single field)

All fields are min-max normalized to [0,1] before comparison with Total ref (also [0,1]).
SSIM is the more reliable metric (scale-invariant); PSNR also reported.

Usage:
    python eval_vs_total.py
"""
import argparse
import os
import os.path as osp
import re
import sys
import json
import glob

import numpy as np

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.join(REPO_ROOT, "SAX-NeRF"))

from src.utils.util import get_psnr_3d, get_ssim_3d
import pickle


def normalize(vol):
    """min-max normalize 3D volume to [0,1]"""
    v = vol.astype(np.float32)
    vmin, vmax = v.min(), v.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(v)
    return (v - vmin) / (vmax - vmin)


def walnut_subdir(walnut):
    """'Walnut_1' -> '' (legacy paths), 'Walnut_2' -> 'walnut_2/'."""
    return "" if walnut == "Walnut_1" else f"{walnut.lower()}/"


def walnutize_path(path, walnut):
    """Insert walnut_X/ between v3_xxx/ and EXPNAME for non-Walnut_1 walnuts."""
    if walnut == "Walnut_1":
        return path
    sub = f"{walnut.lower()}/"
    return re.sub(
        r"(experiments/res_256/v3_[^/]+/)",
        lambda m: m.group(1) + sub,
        path,
    )


def load_total_ref(walnut="Walnut_1"):
    sub = walnut_subdir(walnut)
    p = osp.join(REPO_ROOT, f"SAX-NeRF/data/res_256/v3_phys/{sub}walnut_total_ref.pickle")
    with open(p, "rb") as f:
        d = pickle.load(f)
    print(f"Loaded Total ref ({walnut}): shape={d['image'].shape}, range=[{d['image'].min():.4f}, {d['image'].max():.4f}]")
    return d["image"].astype(np.float32)


def find_best_eval_dir(parent_glob, metric="psnr_3d_avg"):
    """Scan all epoch_*/stats.txt under parent_glob, return path of best eval dir."""
    candidates = sorted(glob.glob(osp.join(parent_glob, "*/eval/epoch_*/stats.txt")))
    if not candidates:
        return None, None
    best_val = -1e9
    best_dir = None
    for stats_p in candidates:
        try:
            with open(stats_p) as f:
                for line in f:
                    if line.startswith(metric):
                        val = float(line.split(":")[1].strip())
                        if val > best_val:
                            best_val = val
                            best_dir = osp.dirname(stats_p)
                        break
        except Exception as e:
            print(f"  warn: skipping {stats_p}: {e}")
    return best_dir, best_val


def load_field(method_name, eval_dir, field_strategy):
    """Load the single-field volume from eval_dir based on strategy.
    Returns (volume, label_explaining_what_it_is) or (None, None) if missing.
    """
    if field_strategy == "image_pred":
        p = osp.join(eval_dir, "image_pred.npy")
        return (np.load(p), "image_pred (single energy reconstruction)") if osp.exists(p) else (None, None)
    elif field_strategy == "rho_pred":
        p = osp.join(eval_dir, "rho_pred.npy")
        return (np.load(p), "rho_pred (Path B single field)") if osp.exists(p) else (None, None)
    elif field_strategy == "rho_total":
        p = osp.join(eval_dir, "rho_total.npy")
        return (np.load(p), "rho_total = α_w + α_2 (Path D single field)") if osp.exists(p) else (None, None)
    elif field_strategy == "image_pred_low":
        p = osp.join(eval_dir, "image_pred_low.npy")
        return (np.load(p), "image_pred_low (dual-head 20keV view)") if osp.exists(p) else (None, None)
    elif field_strategy == "image_pred_high":
        p = osp.join(eval_dir, "image_pred_high.npy")
        return (np.load(p), "image_pred_high (dual-head 60keV view)") if osp.exists(p) else (None, None)
    elif field_strategy == "image_pred_avg":
        pl = osp.join(eval_dir, "image_pred_low.npy")
        ph = osp.join(eval_dir, "image_pred_high.npy")
        if not (osp.exists(pl) and osp.exists(ph)):
            return None, None
        avg = (np.load(pl) + np.load(ph)) / 2.0
        return avg, "(image_pred_low + image_pred_high)/2 (dual-head averaged)"
    else:
        raise ValueError(f"unknown strategy: {field_strategy}")


def compute_si_psnr(pred, ref, max_val=1.0, eps=1e-12):
    """Scale-Invariant PSNR (image convention).

    Find optimal scalar α minimizing ||ref - α·pred||², then compute PSNR with
    the aligned MSE. Returns (si_psnr, alpha). α value reveals scale mismatch:
      α ≈ 1 → pred is already in ref scale
      α > 1 → pred is smaller than ref, needs amplification
      α < 1 → pred is larger than ref
    """
    pred_flat = pred.flatten().astype(np.float64)
    ref_flat = ref.flatten().astype(np.float64)
    alpha = float(np.dot(ref_flat, pred_flat) / max(float(np.dot(pred_flat, pred_flat)), eps))
    aligned_pred = alpha * pred_flat
    mse = float(np.mean((ref_flat - aligned_pred) ** 2))
    if mse < eps:
        return float("inf"), alpha
    psnr = float(10.0 * np.log10((max_val ** 2) / mse))
    return psnr, alpha


def evaluate(method_name, parent_glob, field_strategy, best_metric, total_ref):
    print(f"\n=== {method_name} ===")
    eval_dir, best_val = find_best_eval_dir(parent_glob, metric=best_metric)
    if eval_dir is None:
        print(f"  no eval dirs found for: {parent_glob}")
        return None
    print(f"  best eval @ {eval_dir} ({best_metric}={best_val:.4f})")
    vol, label = load_field(method_name, eval_dir, field_strategy)
    if vol is None:
        print(f"  field {field_strategy} not found in eval dir")
        return None

    # PROTOCOL (M9 转变 6 — 最终 fair version):
    # 报告三个指标，每个回答不同问题：
    # 1. PSNR (independent minmax) — 旧 protocol，保留兼容（结构 + minmax outlier bias 混合）
    # 2. SSIM (independent minmax) — 同上，但 SSIM 设计上 scale-robust
    # 3. SI-PSNR (raw values, optimal scalar α 校准) — principled fair PSNR
    #    - 对每个 method 找最优 α 让 ||ref - α·pred||² 最小
    #    - PSNR 计算在 α-aligned 后 → 只惩罚结构差异，不惩罚 scale convention 选择
    #    - 报告 α 值揭示 scale mismatch（α 远离 1 = pred 跟 ref 物理 scale 差大）
    v_norm = normalize(vol)
    ref_norm = normalize(total_ref)
    psnr = float(get_psnr_3d(v_norm, ref_norm, PIXEL_MAX=1.0))
    ssim = float(get_ssim_3d(v_norm, ref_norm, PIXEL_MAX=1.0))
    # SI-PSNR on raw values (with ref's natural max as MAX), captures pure structure quality
    pred_raw = vol.astype(np.float32)
    ref_raw = total_ref.astype(np.float32)
    si_psnr, alpha = compute_si_psnr(pred_raw, ref_raw, max_val=float(ref_raw.max() - ref_raw.min()))
    print(f"  field: {label}")
    print(f"  raw range: [{vol.min():.4f}, {vol.max():.4f}]")
    print(f"  vs Total — PSNR={psnr:.3f}, SSIM={ssim:.4f}, SI-PSNR={si_psnr:.3f} (α={alpha:.4f})")
    return {
        "method": method_name,
        "eval_dir": eval_dir,
        "best_metric": best_metric,
        "best_val": best_val,
        "field_strategy": field_strategy,
        "field_label": label,
        "psnr_vs_total": psnr,
        "ssim_vs_total": ssim,
        "si_psnr_vs_total": si_psnr,
        "alpha": alpha,
        "raw_min": float(vol.min()),
        "raw_max": float(vol.max()),
    }


def main():
    parser = argparse.ArgumentParser(description="vs-Total eval per walnut")
    parser.add_argument("--walnut", default="Walnut_1",
                        choices=["Walnut_1", "Walnut_2", "Walnut_3"],
                        help="Walnut id (selects ref pickle, ckpt subdir, output dir)")
    args = parser.parse_args()

    total = load_total_ref(args.walnut)
    out_dir = osp.join(REPO_ROOT, f"experiments/{args.walnut.lower()}/eval_M9_total")
    os.makedirs(out_dir, exist_ok=True)

    # METHODS being compared (all use single-energy or dual-energy projections, NOT Total)
    SPECS = [
        # (method_name, parent_glob, field_strategy, best_metric_in_stats_txt)
        # Single energy baselines (50, 100, 25 view)
        ("Single 50 @20",   "experiments/res_256/v3_phys/walnut_20kev_50_phys",          "image_pred", "psnr_3d"),
        ("Single 50 @60",   "experiments/res_256/v3_phys/walnut_60kev_50_phys",          "image_pred", "psnr_3d"),
        ("Single 100 @20",  "experiments/res_256/v3_phys_100view/walnut_20kev_100_phys", "image_pred", "psnr_3d"),
        ("Single 100 @60",  "experiments/res_256/v3_phys_100view/walnut_60kev_100_phys", "image_pred", "psnr_3d"),
        ("Single 25 @20",   "experiments/res_256/v3_phys_25view/walnut_20kev_25_phys",   "image_pred", "psnr_3d"),
        ("Single 25 @60",   "experiments/res_256/v3_phys_25view/walnut_60kev_25_phys",   "image_pred", "psnr_3d"),

        # Phase F — full single-energy sweep (10/30/40/50/70/80 keV × 25v/50v)
        ("Single 50 @10",   "experiments/res_256/v3_phys/walnut_10kev_50_phys",          "image_pred", "psnr_3d"),
        ("Single 50 @30",   "experiments/res_256/v3_phys/walnut_30kev_50_phys",          "image_pred", "psnr_3d"),
        ("Single 50 @40",   "experiments/res_256/v3_phys/walnut_40kev_50_phys",          "image_pred", "psnr_3d"),
        ("Single 50 @50",   "experiments/res_256/v3_phys/walnut_50kev_50_phys",          "image_pred", "psnr_3d"),
        ("Single 50 @70",   "experiments/res_256/v3_phys/walnut_70kev_50_phys",          "image_pred", "psnr_3d"),
        ("Single 50 @80",   "experiments/res_256/v3_phys/walnut_80kev_50_phys",          "image_pred", "psnr_3d"),
        ("Single 25 @10",   "experiments/res_256/v3_phys_25view/walnut_10kev_25_phys",   "image_pred", "psnr_3d"),
        ("Single 25 @30",   "experiments/res_256/v3_phys_25view/walnut_30kev_25_phys",   "image_pred", "psnr_3d"),
        ("Single 25 @40",   "experiments/res_256/v3_phys_25view/walnut_40kev_25_phys",   "image_pred", "psnr_3d"),
        ("Single 25 @50",   "experiments/res_256/v3_phys_25view/walnut_50kev_25_phys",   "image_pred", "psnr_3d"),
        ("Single 25 @70",   "experiments/res_256/v3_phys_25view/walnut_70kev_25_phys",   "image_pred", "psnr_3d"),
        ("Single 25 @80",   "experiments/res_256/v3_phys_25view/walnut_80kev_25_phys",   "image_pred", "psnr_3d"),

        # Path B (single-field) — rho_pred is the actual single field deliverable
        ("Path B 50 lh1",       "experiments/res_256/v3_dual_phys_singlefield/walnut_20kev_60kev_50_singlefield_lh1",     "rho_pred", "psnr_3d_avg"),
        ("Path B 50 lh10",      "experiments/res_256/v3_dual_phys_singlefield/walnut_20kev_60kev_50_singlefield_lh10",    "rho_pred", "psnr_3d_avg"),
        ("Path B 50 flip60",    "experiments/res_256/v3_dual_phys_singlefield/walnut_20kev_60kev_50_singlefield_flip60",  "rho_pred", "psnr_3d_avg"),
        ("Path B 25 lh1",       "experiments/res_256/v3_dual_phys_singlefield_25view/walnut_20kev_60kev_25_singlefield_lh1",    "rho_pred", "psnr_3d_avg"),
        ("Path B 25 lh10",      "experiments/res_256/v3_dual_phys_singlefield_25view/walnut_20kev_60kev_25_singlefield_lh10",   "rho_pred", "psnr_3d_avg"),
        ("Path B 25 flip60",    "experiments/res_256/v3_dual_phys_singlefield_25view/walnut_20kev_60kev_25_singlefield_flip60", "rho_pred", "psnr_3d_avg"),

        # Path D (2-basis, learnable second basis) — rho_total = α_w + α_2 is the deliverable
        ("Path D 50 basis2 (raw)",       "experiments/res_256/v3_dual_phys_basis2/walnut_20kev_60kev_50_basis2_lh10",                       "rho_total", "psnr_3d_avg"),
        ("Path D 25 basis2 (raw)",       "experiments/res_256/v3_dual_phys_basis2_25view/walnut_20kev_60kev_25_basis2_lh10",                "rho_total", "psnr_3d_avg"),
        ("Path D 50 basis2 (softplus)",  "experiments/res_256/v3_dual_phys_basis2/walnut_20kev_60kev_50_basis2_softplus_lh10",              "rho_total", "psnr_3d_avg"),
        ("Path D 25 basis2 (softplus)",  "experiments/res_256/v3_dual_phys_basis2_25view/walnut_20kev_60kev_25_basis2_softplus_lh10",       "rho_total", "psnr_3d_avg"),
        # HAP init variants (Hydroxyapatite, original WalnutPCCT repo basis material)
        ("Path D 25 basis2 (softplus+HAP)", "experiments/res_256/v3_dual_phys_basis2_25view/walnut_20kev_60kev_25_basis2_softplus_hap_lh10", "rho_total", "psnr_3d_avg"),
        ("Path D 50 basis2 (softplus+HAP)", "experiments/res_256/v3_dual_phys_basis2/walnut_20kev_60kev_50_basis2_softplus_hap_lh10",        "rho_total", "psnr_3d_avg"),
        # Phase D — multi-energy-pair ablation (Path D Softplus α+bone, NIST-derived κ per pair)
        ("Path D 25 (10,80) basis2 (softplus)", "experiments/res_256/v3_dual_phys_basis2_25view/walnut_10kev_80kev_25_basis2_softplus_lh10", "rho_total", "psnr_3d_avg"),
        ("Path D 25 (20,70) basis2 (softplus)", "experiments/res_256/v3_dual_phys_basis2_25view/walnut_20kev_70kev_25_basis2_softplus_lh10", "rho_total", "psnr_3d_avg"),
        ("Path D 25 (10,70) basis2 (softplus)", "experiments/res_256/v3_dual_phys_basis2_25view/walnut_10kev_70kev_25_basis2_softplus_lh10", "rho_total", "psnr_3d_avg"),
        ("Path D 25 (30,80) basis2 (softplus)", "experiments/res_256/v3_dual_phys_basis2_25view/walnut_30kev_80kev_25_basis2_softplus_lh10", "rho_total", "psnr_3d_avg"),
        ("Path D 50 (10,80) basis2 (softplus)", "experiments/res_256/v3_dual_phys_basis2/walnut_10kev_80kev_50_basis2_softplus_lh10",        "rho_total", "psnr_3d_avg"),
        ("Path D 50 (20,70) basis2 (softplus)", "experiments/res_256/v3_dual_phys_basis2/walnut_20kev_70kev_50_basis2_softplus_lh10",        "rho_total", "psnr_3d_avg"),
        ("Path D 50 (10,70) basis2 (softplus)", "experiments/res_256/v3_dual_phys_basis2/walnut_10kev_70kev_50_basis2_softplus_lh10",        "rho_total", "psnr_3d_avg"),
        ("Path D 50 (30,80) basis2 (softplus)", "experiments/res_256/v3_dual_phys_basis2/walnut_30kev_80kev_50_basis2_softplus_lh10",        "rho_total", "psnr_3d_avg"),
        # Phase D extra — fixed-low=20 sweep (20,50) (20,80)
        ("Path D 25 (20,50) basis2 (softplus)", "experiments/res_256/v3_dual_phys_basis2_25view/walnut_20kev_50kev_25_basis2_softplus_lh10", "rho_total", "psnr_3d_avg"),
        ("Path D 25 (20,80) basis2 (softplus)", "experiments/res_256/v3_dual_phys_basis2_25view/walnut_20kev_80kev_25_basis2_softplus_lh10", "rho_total", "psnr_3d_avg"),
        ("Path D 50 (20,50) basis2 (softplus)", "experiments/res_256/v3_dual_phys_basis2/walnut_20kev_50kev_50_basis2_softplus_lh10",        "rho_total", "psnr_3d_avg"),
        ("Path D 50 (20,80) basis2 (softplus)", "experiments/res_256/v3_dual_phys_basis2/walnut_20kev_80kev_50_basis2_softplus_lh10",        "rho_total", "psnr_3d_avg"),
        # Path D 2x2 grid: param × κ_2 init (25v)
        ("Path D 25 alpha+water (B)",    "experiments/res_256/v3_dual_phys_basis2_25view/walnut_20kev_60kev_25_basis2_water_init_lh10",    "rho_total", "psnr_3d_avg"),
        ("Path D 25 frac+bone (C)",      "experiments/res_256/v3_dual_phys_basis2_25view/walnut_20kev_60kev_25_basis2_frac_bone_lh10",     "rho_total", "psnr_3d_avg"),
        ("Path D 25 frac+water (D)",     "experiments/res_256/v3_dual_phys_basis2_25view/walnut_20kev_60kev_25_basis2_frac_water_lh10",    "rho_total", "psnr_3d_avg"),
        ("Path D 50 alpha+water (B)",    "experiments/res_256/v3_dual_phys_basis2/walnut_20kev_60kev_50_basis2_water_init_lh10",           "rho_total", "psnr_3d_avg"),
        ("Path D 50 frac+bone (C)",      "experiments/res_256/v3_dual_phys_basis2/walnut_20kev_60kev_50_basis2_frac_bone_lh10",            "rho_total", "psnr_3d_avg"),
        ("Path D 50 frac+water (D)",     "experiments/res_256/v3_dual_phys_basis2/walnut_20kev_60kev_50_basis2_frac_water_lh10",           "rho_total", "psnr_3d_avg"),

        # Dual-head — try 3 strategies (low view, high view, average)
        ("M4.5-a 50 lh10 (low view)",   "experiments/res_256/v3_dual_phys/walnut_20kev_60kev_50_dual_phys_lambda_high_10",  "image_pred_low",  "psnr_3d_avg"),
        ("M4.5-a 50 lh10 (high view)",  "experiments/res_256/v3_dual_phys/walnut_20kev_60kev_50_dual_phys_lambda_high_10",  "image_pred_high", "psnr_3d_avg"),
        ("M4.5-a 50 lh10 (avg)",        "experiments/res_256/v3_dual_phys/walnut_20kev_60kev_50_dual_phys_lambda_high_10",  "image_pred_avg",  "psnr_3d_avg"),
        ("Dual-head 25 lh10 (low view)","experiments/res_256/v3_dual_phys_25view/walnut_20kev_60kev_25_dualhead_lh10",      "image_pred_low",  "psnr_3d_avg"),
        ("Dual-head 25 lh10 (high view)","experiments/res_256/v3_dual_phys_25view/walnut_20kev_60kev_25_dualhead_lh10",     "image_pred_high", "psnr_3d_avg"),
        ("Dual-head 25 lh10 (avg)",     "experiments/res_256/v3_dual_phys_25view/walnut_20kev_60kev_25_dualhead_lh10",      "image_pred_avg",  "psnr_3d_avg"),
    ]

    # CEILING references (Single SAX-NeRF trained directly on Total — these are upper bounds)
    CEILING = [
        ("Single 25 Total",     "experiments/res_256/v3_phys_total/walnut_total_25_phys",  "image_pred", "psnr_3d"),
        ("Single 50 Total",     "experiments/res_256/v3_phys_total/walnut_total_50_phys",  "image_pred", "psnr_3d"),
    ]

    # Walnut-aware path expansion (Walnut_1: unchanged; Walnut_2/3: insert subdir)
    SPECS = [(n, walnutize_path(p, args.walnut), s, m) for n, p, s, m in SPECS]
    CEILING = [(n, walnutize_path(p, args.walnut), s, m) for n, p, s, m in CEILING]

    results = []
    for name, glob_path, strategy, metric in SPECS:
        full_glob = osp.join(REPO_ROOT, glob_path)
        r = evaluate(name, full_glob, strategy, metric, total)
        if r is not None:
            r["section"] = "method"
            results.append(r)

    ceiling_results = []
    for name, glob_path, strategy, metric in CEILING:
        full_glob = osp.join(REPO_ROOT, glob_path)
        r = evaluate(name, full_glob, strategy, metric, total)
        if r is not None:
            r["section"] = "ceiling"
            ceiling_results.append(r)

    # Save JSON + CSV (all together, with section field)
    all_results = ceiling_results + results
    json_p = osp.join(out_dir, "results.json")
    with open(json_p, "w") as f:
        json.dump(all_results, f, indent=2)
    csv_p = osp.join(out_dir, "comparison_table.csv")
    with open(csv_p, "w") as f:
        f.write("section,method,field_strategy,psnr_vs_total,ssim_vs_total,si_psnr_vs_total,alpha,raw_max,best_val,best_metric,eval_dir\n")
        for r in all_results:
            f.write(f"{r['section']},\"{r['method']}\",{r['field_strategy']},"
                    f"{r['psnr_vs_total']:.4f},{r['ssim_vs_total']:.4f},"
                    f"{r['si_psnr_vs_total']:.4f},{r['alpha']:.4f},{r['raw_max']:.4f},"
                    f"{r['best_val']:.4f},{r['best_metric']},\"{r['eval_dir']}\"\n")
    print(f"\n\nSaved: {json_p}\nSaved: {csv_p}")

    # Print formatted output: Ceiling section + Methods section (sorted by SI-PSNR primary)
    header_fmt = f"{'Method':<32} {'Field':<22} {'PSNR':>7} {'SSIM':>7} {'SI-PSNR':>8} {'α':>7} {'rawMax':>7}"
    row_fmt = lambda r: (f"{r['method']:<32} {r['field_strategy']:<22} "
                         f"{r['psnr_vs_total']:>7.3f} {r['ssim_vs_total']:>7.4f} "
                         f"{r['si_psnr_vs_total']:>8.3f} {r['alpha']:>7.4f} {r['raw_max']:>7.4f}")

    print("\n" + "=" * 100)
    print("CEILING REFERENCE (single SAX-NeRF trained directly on Total — upper bound)")
    print("=" * 100)
    print(header_fmt)
    print("-" * 100)
    for r in ceiling_results:
        print(row_fmt(r))

    methods_sorted = sorted(results, key=lambda r: r["si_psnr_vs_total"], reverse=True)
    print("\n" + "=" * 100)
    print("METHODS UNDER COMPARISON — sorted by SI-PSNR ↓ (principled fair PSNR with optimal scale)")
    print("=" * 100)
    print(header_fmt)
    print("-" * 100)
    for r in methods_sorted:
        print(row_fmt(r))
    print("=" * 100)
    print("Notes: PSNR & SSIM use independent minmax (legacy); SI-PSNR uses raw + optimal α (fair).")
    print("       α = optimal scale factor for pred → ref. α far from 1 means scale mismatch.")


if __name__ == "__main__":
    main()
