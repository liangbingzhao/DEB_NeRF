"""Convert Total volume → SAX-NeRF pickle WITH projections.

Total = polychromatic FDK+TV reconstruction. Used as a "general-field" baseline:
train single SAX-NeRF directly on Total to see what reconstruction quality is achievable
when the training target IS the spectrum-integrated walnut.

Pipeline:
  1. Load Total DICOM, downsample to 256³, container removal, air shift
  2. Min-max normalize to [0, 1]
  3. Forward project (TIGRE Ax) at n_train and n_val angles
  4. Save pickle with image + train/val projections

Usage:
  python convert_walnut_total_with_proj.py --n_train 25 --n_val 50 --seed 42
  python convert_walnut_total_with_proj.py --n_train 50 --n_val 50 --seed 42
"""
import os
import sys
import pickle
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_walnut_phys import load_dicom_volume_raw, dicom_dir_for, generate_single_pickle


def main():
    parser = argparse.ArgumentParser(description="Walnut Total + projections")
    parser.add_argument("--walnut", type=str, default="Walnut_1",
                        choices=["Walnut_1", "Walnut_2", "Walnut_3"],
                        help="Walnut id (defaults recon_base + output paths)")
    parser.add_argument("--recon_base", type=str, default=None,
                        help="Override recon dir (default: Reconstructions_mat/{walnut}/FDK_Dose_1_hann_TV_100_20)")
    parser.add_argument("--vol_size", type=int, nargs=3, default=[256, 256, 256])
    parser.add_argument("--n_train", type=int, default=50)
    parser.add_argument("--n_val", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no_remove_container", action="store_true")
    # M1-B physics-aligned mode: NIST calibration at effective energy, matches v3_phys shared scale
    # Default off → original min-max behavior preserved for backwards compatibility
    parser.add_argument("--physics_calib", action="store_true",
                        help="Use v3_phys-style NIST calibration at effective energy (M1-B protocol)")
    parser.add_argument("--mu_water_eff", type=float, default=0.2586,
                        help="μ_water at Total's effective energy (43.5 keV from empirical fit)")
    parser.add_argument("--eff_energy_keV", type=float, default=43.5,
                        help="effective energy label saved in norm dict (informational only)")
    parser.add_argument("--global_scale", type=float, default=3.567219,
                        help="Shared scale matching v3_phys (= dual pickle mu_low.max)")
    args = parser.parse_args()

    if args.recon_base is None:
        args.recon_base = f"Reconstructions_mat/{args.walnut}/FDK_Dose_1_hann_TV_100_20"

    print("=" * 70)
    print(f"Walnut Total → SAX-NeRF pickle (n_train={args.n_train}, n_val={args.n_val})")
    print("=" * 70)

    if args.output is None:
        subdir = "" if args.walnut == "Walnut_1" else f"{args.walnut.lower()}/"
        args.output = (f"../SAX-NeRF/data/res_{args.vol_size[0]}/v3_phys/"
                       f"{subdir}walnut_total_{args.n_train}.pickle")

    print(f"\n[1] Loading Total volume")
    dicom_dir = dicom_dir_for(args.recon_base, "Total")
    assert os.path.isdir(dicom_dir), f"Not found: {dicom_dir}"
    vol = load_dicom_volume_raw(dicom_dir, target_shape=tuple(args.vol_size),
                                remove_container=not args.no_remove_container)

    if args.physics_calib:
        print(f"\n[2] Physics calibration (air shift -> -1000 + NIST μ_water at {args.eff_energy_keV} keV + global scale)")
        air = float(vol.min())
        shift = -1000.0 - air
        hu_adj = vol + shift
        mu = args.mu_water_eff * (hu_adj / 1000.0 + 1.0)
        mu = np.clip(mu, 0.0, None)
        image = np.clip(mu / args.global_scale, 0.0, 1.0).astype(np.float32)
        print(f"  air HU (raw): {air:.2f}, shift: +{shift:.0f} (-> -1000)")
        print(f"  μ_water (eff): {args.mu_water_eff:.4f} cm⁻¹ at {args.eff_energy_keV} keV")
        print(f"  global scale: {args.global_scale:.6f}")
        print(f"  μ range:         [{mu.min():.4f}, {mu.max():.4f}] mean={mu.mean():.4f}")
        print(f"  post-norm range: [{image.min():.4f}, {image.max():.4f}] mean={image.mean():.4f}")
        norm = {
            "air": air, "shift": shift,
            "mu_water": args.mu_water_eff,
            "scale": args.global_scale,
            "eff_energy_keV": args.eff_energy_keV,
            "calib": "phys_air_shift_to_-1000_then_NIST_water_at_eff_energy",
            "source": f"convert_walnut_total_with_proj.py M1-B n_train={args.n_train} (eff_E={args.eff_energy_keV})",
        }
    else:
        print(f"\n[2] Min-max normalize to [0, 1]")
        raw_min, raw_max = float(vol.min()), float(vol.max())
        image = ((vol - raw_min) / (raw_max - raw_min + 1e-12)).astype(np.float32)
        print(f"  raw HU range:    [{raw_min:.2f}, {raw_max:.2f}]")
        print(f"  post-norm range: [{image.min():.4f}, {image.max():.4f}], mean={image.mean():.4f}")
        norm = {
            "raw_min": raw_min,
            "raw_max": raw_max,
            "calib": "minmax_to_0_1",
            "source": f"convert_walnut_total_with_proj.py n_train={args.n_train}",
        }

    print(f"\n[3] Forward projection")
    data = generate_single_pickle(image, "Total",
                                  n_train=args.n_train, n_val=args.n_val,
                                  seed=args.seed)

    data["norm"] = norm

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    fsize = os.path.getsize(args.output) / 1e6
    print(f"\n[4] Saved: {args.output} ({fsize:.1f} MB)")

    pl = data["train"]["projections"]
    pv = data["val"]["projections"]
    print(f"\n[Projections]")
    print(f"  train: shape={pl.shape}, range=[{pl.min():.6f}, {pl.max():.6f}], mean={pl.mean():.6f}")
    print(f"  val:   shape={pv.shape}, range=[{pv.min():.6f}, {pv.max():.6f}], mean={pv.mean():.6f}")
    print("Done.")


if __name__ == "__main__":
    main()
