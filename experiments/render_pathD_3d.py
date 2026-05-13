"""
M9 Phase A — 3D render of Path D Softplus alpha+bone winner.

Visualizes the basis decomposition (α_w, α_2, ρ_total) learned by the winner
network for both 25v and 50v configurations. Reuses M8's spectral_render
machinery (camera, ray AABB, trilinear sampler, GrayscaleTF) and adds a new
Basis2FalseColorTF that maps α_w → blue and α_2 → red additively (so shell
appears warm, bulk kernel appears cool).

Outputs per view (`experiments/eval_M9_total/render3d/{25v,50v}/`):

  Full volume rotation GIFs:
    rotation_alpha_w_gray.gif
    rotation_alpha_2_gray.gif
    rotation_total_gray.gif
    rotation_basis2_falsecolor.gif

  Shell / kernel decomposition (mask computed from ρ_total):
    rotation_shell_total_gray.gif
    rotation_kernel_total_gray.gif
    rotation_shell_basis2_falsecolor.gif
    rotation_kernel_basis2_falsecolor.gif

  Static panel (mid-z slice + mid-rotation 3D thumb per quantity):
    headline_panel.png
"""
from __future__ import annotations

import argparse
import glob
import os
import os.path as osp
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from spectral_render import (  # noqa: E402
    BOUND, CAMERA_DISTANCE,
    GrayscaleTF, TransferFn,
    create_camera_rays, ray_aabb_intersect, trilinear_sample,
    render_view, percentile_clip,
)


# ──────────────────────────────────────────────────────────────────────
# Path D winner ckpt locations (per walnut)
# ──────────────────────────────────────────────────────────────────────
# Walnut_1 uses hardcoded snapshots (matches Phase A first run reproducibility).
# Walnut_2/3 auto-discover latest epoch under matching expname.
_WINNER_W1 = {
    "50v": osp.join(REPO_ROOT,
        "experiments/res_256/v3_dual_phys_basis2/"
        "walnut_20kev_60kev_50_basis2_softplus_lh10/"
        "2026_05_03_09_44_48/eval/epoch_01250"),
    "25v": osp.join(REPO_ROOT,
        "experiments/res_256/v3_dual_phys_basis2_25view/"
        "walnut_20kev_60kev_25_basis2_softplus_lh10/"
        "2026_05_03_09_44_51/eval/epoch_01500"),
}


_VARIANT_SUFFIX = {
    "softplus":   "softplus_lh10",          # A (winner default, backward compat)
    "water_init": "water_init_lh10",        # B alpha+water
    "frac_bone":  "frac_bone_lh10",         # C
    "frac_water": "frac_water_lh10",        # D
    "raw":        "lh10",                   # Raw (no constrain_kappa_2)
    "hap":        "softplus_hap_lh10",      # HAP material init
}


def get_winner_eval_dir(walnut, view_tag, low=20, high=60, variant="softplus"):
    suffix = _VARIANT_SUFFIX[variant]
    if walnut == "Walnut_1" and (low, high) == (20, 60) and variant == "softplus":
        return _WINNER_W1[view_tag]
    if walnut == "Walnut_1":
        sub = ""
    elif walnut == "Walnut_1_mat":
        sub = "walnut_1_mat/"
    else:
        sub = f"{walnut.lower()}/"
    nv = 50 if view_tag == "50v" else 25
    view_dir = "v3_dual_phys_basis2" if nv == 50 else "v3_dual_phys_basis2_25view"
    expname = f"walnut_{low}kev_{high}kev_{nv}_basis2_{suffix}"
    parent = osp.join(REPO_ROOT, f"experiments/res_256/{view_dir}/{sub}{expname}")
    candidates = sorted(glob.glob(osp.join(parent, "*/eval/epoch_*")))
    return candidates[-1] if candidates else None


# ──────────────────────────────────────────────────────────────────────
# False-color TF: α_w → blue, α_2 → red, opacity from ρ_total
# ──────────────────────────────────────────────────────────────────────
class Basis2FalseColorTF(TransferFn):
    """Additive RGB: B = α_w / α_w_max, R = α_2 / α_2_max, G = blend.

    Designed for Path D's water/bone basis: bulk water-like material (α_w high,
    α_2 low) → blue/cyan; mineral-rich shell (α_2 high) → red/magenta;
    transition zones blend → purple.

    Opacity is taken from ρ_total so the integrated density drives ray
    accumulation (avoids losing the shell when α_2 alone is sparse).
    """

    def __init__(self, alpha_w_max: float, alpha_2_max: float, rho_max: float,
                 alpha_power: float = 1.0, rho_threshold: float = 0.05,
                 r_gain: float = 1.6, b_gain: float = 1.0):
        self.alpha_w_max = alpha_w_max
        self.alpha_2_max = alpha_2_max
        self.rho_max = rho_max
        self.alpha_power = alpha_power
        self.rho_threshold = rho_threshold
        # Gain on R because α_2 is sparse (~3% non-zero); slight boost
        # makes shell highlights visible without saturating the bulk.
        self.r_gain = r_gain
        self.b_gain = b_gain

    def __call__(self, features: torch.Tensor):
        # features: [N, 3] = (α_w, α_2, ρ_total)
        a_w = features[..., 0]
        a_2 = features[..., 1]
        rho = features[..., 2]
        b_n = (a_w * self.b_gain / self.alpha_w_max).clamp(0, 1)
        r_n = (a_2 * self.r_gain / self.alpha_2_max).clamp(0, 1)
        g_n = (b_n + r_n) * 0.5
        rgb = torch.stack([r_n, g_n, b_n], dim=-1)
        rho_n = (rho / self.rho_max).clamp(0, 1)
        opacity = rho_n.pow(self.alpha_power)
        opacity = opacity * (rho > self.rho_threshold).float()
        return rgb, opacity


# ──────────────────────────────────────────────────────────────────────
# Shell / kernel mask (uses ρ_total)
# ──────────────────────────────────────────────────────────────────────
def compute_shell_kernel_masks(rho_total: np.ndarray, air_thresh: float = 0.05,
                               shell_thickness: int = 3,
                               kernel_peel: int | None = None) -> tuple:
    """Returns (shell_mask, kernel_mask) from ρ_total.

    shell  = material within `shell_thickness` voxels of outside-air
    kernel = material strictly more than `kernel_peel` voxels from outside-air

    When `kernel_peel > shell_thickness`, the two masks are no longer
    complementary — there is a transition band that belongs to neither (which
    is exactly what we want when shell_thickness is tuned for thin highlighting
    and kernel_peel is tuned for clean removal of shell residue).
    Defaults to `shell_thickness` when None (legacy behavior).
    """
    from scipy.ndimage import label as cc_label, distance_transform_edt
    if kernel_peel is None:
        kernel_peel = shell_thickness
    material = rho_total > air_thresh
    air = ~material
    labels_air, _ = cc_label(air)
    sizes = np.bincount(labels_air.ravel())
    sizes[0] = 0
    outside_label = int(np.argmax(sizes))
    outside_air = (labels_air == outside_label)
    dist_from_outside = distance_transform_edt(~outside_air)
    shell = material & (dist_from_outside <= float(shell_thickness))
    kernel = material & (dist_from_outside > float(kernel_peel))
    return shell, kernel


# ──────────────────────────────────────────────────────────────────────
# Render helpers
# ──────────────────────────────────────────────────────────────────────
def render_rotation_gif(volume, tf, output_path, gif_frames, gif_size,
                        gif_samples, gif_elev, fov, density_scale, fps=10):
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio  # type: ignore
    frames = []
    t0 = time.time()
    for k in range(gif_frames):
        azim = k * (360.0 / gif_frames)
        img = render_view(volume, tf, azim, gif_elev,
                          gif_size, gif_samples, fov,
                          density_scale=density_scale)
        frames.append((img * 255).clip(0, 255).astype(np.uint8))
        if (k + 1) % 6 == 0 or k == gif_frames - 1:
            elapsed = time.time() - t0
            print(f"      [{k+1}/{gif_frames}] azim={azim:5.0f}°  "
                  f"({elapsed/(k+1):.1f}s/frame, {elapsed:.0f}s total)")
    imageio.mimsave(output_path, frames, duration=1.0 / fps, loop=0)
    print(f"    GIF -> {output_path}")
    return frames


def render_static(volume, tf, azim, elev, image_size, n_samples, fov,
                  density_scale):
    return render_view(volume, tf, azim, elev, image_size, n_samples, fov,
                       density_scale=density_scale)


# ──────────────────────────────────────────────────────────────────────
# Main pipeline (per view)
# ──────────────────────────────────────────────────────────────────────
def process_view(view_tag: str, eval_dir: str, output_dir: str,
                 device: torch.device, args) -> None:
    print(f"\n{'='*70}")
    print(f"[Phase A] {view_tag}: {eval_dir}")
    print(f"{'='*70}")

    os.makedirs(output_dir, exist_ok=True)

    # ── Load three basis volumes
    a_w = np.load(osp.join(eval_dir, "alpha_w.npy")).astype(np.float32)
    a_2 = np.load(osp.join(eval_dir, "alpha_2.npy")).astype(np.float32)
    rho = np.load(osp.join(eval_dir, "rho_total.npy")).astype(np.float32)
    print(f"  α_w:     shape={a_w.shape}, range=[{a_w.min():.4f}, {a_w.max():.4f}]")
    print(f"  α_2:     shape={a_2.shape}, range=[{a_2.min():.4f}, {a_2.max():.4f}]")
    print(f"  ρ_total: shape={rho.shape}, range=[{rho.min():.4f}, {rho.max():.4f}]")

    # ── Color/opacity scales (p99.5 to suppress outliers)
    a_w_p99 = percentile_clip(a_w)
    a_2_p99 = percentile_clip(a_2)
    rho_p99 = percentile_clip(rho)
    print(f"  p99.5 clips: α_w={a_w_p99:.3f}, α_2={a_2_p99:.3f}, ρ={rho_p99:.3f}")

    # ── Shell/kernel masks (from ρ_total)
    shell_mask, kernel_mask = compute_shell_kernel_masks(
        rho, air_thresh=args.air_thresh,
        shell_thickness=args.shell_thickness,
        kernel_peel=args.kernel_peel)
    print(f"  shell  (K={args.shell_thickness}): {int(shell_mask.sum()):,} voxels "
          f"({100*shell_mask.sum()/shell_mask.size:.2f}% of volume)")
    print(f"  kernel (peel={args.kernel_peel}): {int(kernel_mask.sum()):,} voxels "
          f"({100*kernel_mask.sum()/kernel_mask.size:.2f}% of volume)")

    # ── Build masked volumes
    a_w_shell  = (a_w * shell_mask).astype(np.float32)
    a_2_shell  = (a_2 * shell_mask).astype(np.float32)
    rho_shell  = (rho * shell_mask).astype(np.float32)
    a_w_kernel = (a_w * kernel_mask).astype(np.float32)
    a_2_kernel = (a_2 * kernel_mask).astype(np.float32)
    rho_kernel = (rho * kernel_mask).astype(np.float32)

    def to_dev(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(device)

    # Pre-build TFs (shared across modes)
    tf_aw_gray  = GrayscaleTF(a_w_p99, mu_threshold=0.05)
    tf_a2_gray  = GrayscaleTF(a_2_p99, mu_threshold=0.01)
    tf_rho_gray = GrayscaleTF(rho_p99, mu_threshold=0.05)
    tf_falsecolor = Basis2FalseColorTF(a_w_p99, a_2_p99, rho_p99,
                                       rho_threshold=0.05)

    # ── KERNEL-ONLY SWEEP MODE: render kernel GIFs at multiple peel K values,
    #    skip everything else. Useful for tuning kernel mask thickness.
    if args.only_kernel:
        kernel_peels = args.kernel_peels if args.kernel_peels else [args.kernel_peel]
        print(f"\n  [only_kernel] sweeping kernel_peel K values: {kernel_peels}")
        for K in kernel_peels:
            _, k_mask = compute_shell_kernel_masks(
                rho, air_thresh=args.air_thresh,
                shell_thickness=args.shell_thickness,
                kernel_peel=K)
            n_k = int(k_mask.sum())
            print(f"\n  --- kernel_peel K={K}: {n_k:,} voxels "
                  f"({100*n_k/k_mask.size:.2f}% of volume) ---")
            rho_k  = to_dev((rho * k_mask).astype(np.float32))
            fc_k   = to_dev(np.stack([(a_w * k_mask).astype(np.float32),
                                      (a_2 * k_mask).astype(np.float32),
                                      (rho * k_mask).astype(np.float32)], 0))
            for fname, vol_k, tf_k, desc in [
                (f"rotation_kernel_total_gray_K{K:02d}.gif", rho_k,  tf_rho_gray,
                 f"ρ_total kernel only, K={K}"),
                (f"rotation_kernel_basis2_falsecolor_K{K:02d}.gif", fc_k, tf_falsecolor,
                 f"α_w/α_2 distribution on kernel, K={K}"),
            ]:
                print(f"\n  > {fname}  ({desc})")
                out_path = osp.join(output_dir, fname)
                render_rotation_gif(vol_k, tf_k, out_path,
                                    args.gif_frames, args.gif_size,
                                    args.gif_samples, args.gif_elev, args.fov,
                                    args.density_scale, fps=args.fps)
        return

    # ── To torch (single + 3-channel stacks)
    vol = {
        "a_w":         to_dev(a_w),
        "a_2":         to_dev(a_2),
        "rho":         to_dev(rho),
        "rho_shell":   to_dev(rho_shell),
        "rho_kernel":  to_dev(rho_kernel),
        # 3-channel stacks (α_w, α_2, ρ) for false-color
        "fc_full":     to_dev(np.stack([a_w, a_2, rho], 0)),
        "fc_shell":    to_dev(np.stack([a_w_shell, a_2_shell, rho_shell], 0)),
        "fc_kernel":   to_dev(np.stack([a_w_kernel, a_2_kernel, rho_kernel], 0)),
    }

    # ── Define the GIF list
    gif_jobs = [
        ("rotation_alpha_w_gray.gif",                vol["a_w"],        tf_aw_gray,    "α_w (water basis) full"),
        ("rotation_alpha_2_gray.gif",                vol["a_2"],        tf_a2_gray,    "α_2 (mineral basis) full"),
        ("rotation_total_gray.gif",                  vol["rho"],        tf_rho_gray,   "ρ_total full"),
        ("rotation_basis2_falsecolor.gif",           vol["fc_full"],    tf_falsecolor, "α_w blue + α_2 red, full"),
        ("rotation_shell_total_gray.gif",            vol["rho_shell"],  tf_rho_gray,   "ρ_total shell only"),
        ("rotation_kernel_total_gray.gif",           vol["rho_kernel"], tf_rho_gray,   "ρ_total kernel only"),
        ("rotation_shell_basis2_falsecolor.gif",     vol["fc_shell"],   tf_falsecolor, "α_w/α_2 distribution on shell"),
        ("rotation_kernel_basis2_falsecolor.gif",    vol["fc_kernel"],  tf_falsecolor, "α_w/α_2 distribution on kernel"),
    ]

    # ── Render each GIF
    for filename, volume, tf, desc in gif_jobs:
        print(f"\n  > {filename}  ({desc})")
        out_path = osp.join(output_dir, filename)
        render_rotation_gif(volume, tf, out_path,
                            args.gif_frames, args.gif_size,
                            args.gif_samples, args.gif_elev, args.fov,
                            args.density_scale, fps=args.fps)

    # ── Static headline panel: 2 rows × 4 cols
    #   Row 1: mid-z slices (α_w, α_2, ρ_total, falsecolor synthesis)
    #   Row 2: mid-rotation 3D thumbs (α_w, α_2, ρ_total, falsecolor)
    print(f"\n  > headline_panel.png")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    z_mid = a_w.shape[2] // 2
    sl_aw  = a_w[:, :, z_mid]
    sl_a2  = a_2[:, :, z_mid]
    sl_rho = rho[:, :, z_mid]
    # Slice false-color: same logic as the TF, normalized per-pixel
    fc_slice = np.zeros((*sl_aw.shape, 3), dtype=np.float32)
    fc_slice[..., 2] = np.clip(sl_aw / a_w_p99, 0, 1)               # B
    fc_slice[..., 0] = np.clip(sl_a2 * 1.6 / a_2_p99, 0, 1)         # R (boosted)
    fc_slice[..., 1] = (fc_slice[..., 0] + fc_slice[..., 2]) * 0.5  # G

    axes[0, 0].imshow(sl_aw,  cmap="bone", vmin=0, vmax=a_w_p99)
    axes[0, 0].set_title("α_w (water basis) — mid-z slice")
    axes[0, 1].imshow(sl_a2,  cmap="hot",  vmin=0, vmax=a_2_p99)
    axes[0, 1].set_title("α_2 (mineral basis) — mid-z slice")
    axes[0, 2].imshow(sl_rho, cmap="bone", vmin=0, vmax=rho_p99)
    axes[0, 2].set_title("ρ_total — mid-z slice")
    axes[0, 3].imshow(fc_slice)
    axes[0, 3].set_title("False-color (α_w blue + α_2 red) — slice")
    for ax in axes[0]:
        ax.axis("off")

    azim_thumb = args.azim_thumb
    elev_thumb = args.elev_thumb
    img_aw  = render_static(vol["a_w"],     tf_aw_gray,    azim_thumb, elev_thumb,
                            args.thumb_size, args.thumb_samples, args.fov, args.density_scale)
    img_a2  = render_static(vol["a_2"],     tf_a2_gray,    azim_thumb, elev_thumb,
                            args.thumb_size, args.thumb_samples, args.fov, args.density_scale)
    img_rho = render_static(vol["rho"],     tf_rho_gray,   azim_thumb, elev_thumb,
                            args.thumb_size, args.thumb_samples, args.fov, args.density_scale)
    img_fc  = render_static(vol["fc_full"], tf_falsecolor, azim_thumb, elev_thumb,
                            args.thumb_size, args.thumb_samples, args.fov, args.density_scale)
    axes[1, 0].imshow(img_aw)
    axes[1, 0].set_title("α_w — 3D render")
    axes[1, 1].imshow(img_a2)
    axes[1, 1].set_title("α_2 — 3D render")
    axes[1, 2].imshow(img_rho)
    axes[1, 2].set_title("ρ_total — 3D render")
    axes[1, 3].imshow(img_fc)
    axes[1, 3].set_title("False-color — 3D render")
    for ax in axes[1]:
        ax.axis("off")

    fig.suptitle(f"M9 Phase A — Path D Softplus α_w/α_2/ρ_total ({view_tag} winner)",
                 fontsize=14)
    fig.tight_layout()
    panel_path = osp.join(output_dir, "headline_panel.png")
    fig.savefig(panel_path, dpi=130)
    plt.close(fig)
    print(f"    PNG -> {panel_path}")


# ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--walnut", default="Walnut_1",
                   choices=["Walnut_1", "Walnut_1_mat", "Walnut_2", "Walnut_3"],
                   help="Walnut id (selects ckpt + default output_root)")
    p.add_argument("--low_energy", type=int, default=20,
                   help="low energy keV of Path D pair to render (default 20)")
    p.add_argument("--high_energy", type=int, default=60,
                   help="high energy keV of Path D pair to render (default 60)")
    p.add_argument("--variant", default="softplus",
                   choices=["softplus", "water_init", "frac_bone", "frac_water", "raw", "hap"],
                   help="Path D variant (default: softplus alpha+bone, the W1 winner)")
    p.add_argument("--views", nargs="+", default=["25v", "50v"],
                   choices=["25v", "50v"])
    p.add_argument("--output_root", default=None,
                   help="Default: experiments/{walnut}/eval_M9_total/render3d/<low>_<high>[_variant]")
    # GIF quality
    p.add_argument("--gif_size", type=int, default=384,
                   help="GIF frame size in pixels")
    p.add_argument("--gif_samples", type=int, default=192,
                   help="samples per ray for GIF frames")
    p.add_argument("--gif_frames", type=int, default=24)
    p.add_argument("--gif_elev", type=float, default=15.0)
    p.add_argument("--fov", type=float, default=35.0)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--density_scale", type=float, default=800.0)
    # Headline thumbnail quality
    p.add_argument("--thumb_size", type=int, default=384)
    p.add_argument("--thumb_samples", type=int, default=256)
    p.add_argument("--azim_thumb", type=float, default=30.0)
    p.add_argument("--elev_thumb", type=float, default=20.0)
    # Shell/kernel mask
    p.add_argument("--air_thresh", type=float, default=0.05)
    p.add_argument("--shell_thickness", type=int, default=3,
                   help="thickness (voxels) of shell mask: dist_from_outside <= K")
    p.add_argument("--kernel_peel", type=int, default=8,
                   help="peel depth (voxels) for kernel mask: dist_from_outside > K. "
                        "Larger value removes more shell residue from kernel render.")
    p.add_argument("--only_kernel", action="store_true",
                   help="render only kernel GIFs (sweep over --kernel_peels K values), "
                        "skip everything else")
    p.add_argument("--kernel_peels", type=int, nargs="+", default=None,
                   help="list of kernel_peel K values to sweep (only_kernel mode)")
    # Device
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    pair_tag = f"{args.low_energy}_{args.high_energy}"
    full_tag = pair_tag if args.variant == "softplus" else f"{pair_tag}_{args.variant}"
    if args.output_root is None:
        args.output_root = osp.join(
            REPO_ROOT,
            f"experiments/{args.walnut.lower()}/eval_M9_total/render3d/{full_tag}")

    print(f"[Phase A] walnut    = {args.walnut}")
    print(f"[Phase A] pair      = ({args.low_energy}, {args.high_energy}) keV")
    print(f"[Phase A] variant   = {args.variant}")
    print(f"[Phase A] device    = {device}")
    print(f"[Phase A] views     = {args.views}")
    print(f"[Phase A] output    = {args.output_root}")

    for view_tag in args.views:
        eval_dir = get_winner_eval_dir(args.walnut, view_tag,
                                       args.low_energy, args.high_energy,
                                       args.variant)
        if eval_dir is None or not osp.isdir(eval_dir):
            print(f"[Phase A] !! eval dir missing for {args.walnut} {view_tag} "
                  f"({args.low_energy},{args.high_energy}): {eval_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"[Phase A] {view_tag} ckpt = {eval_dir}")
        output_dir = osp.join(args.output_root, view_tag)
        process_view(view_tag, eval_dir, output_dir, device, args)

    print("\n[Phase A] all done.")


if __name__ == "__main__":
    main()
