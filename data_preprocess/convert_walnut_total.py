"""Convert Walnut PCCT Total DICOM → 256³ reference volume (for M9 comparison only).

Total = polychromatic FDK+TV reconstruction (all photons combined, NOT virtual mono).
Used as an "energy-agnostic structural reference" for evaluating Path B's ρ field
and other reconstructions.

Pipeline:
  1. Load 1200 Total DICOM slices (1000×1000)
  2. Apply same air shift + container removal as v3_phys (re-uses load_dicom_volume_raw)
  3. Min-max normalize to [0, 1] (for scale-fair comparison with normalized fields)
  4. Save pickle with just the volume (no projections — this is reference only)

Usage:
  python convert_walnut_total.py --vol_size 256 256 256

Output:
  ../SAX-NeRF/data/res_256/v3_phys/walnut_total_ref.pickle  (~64 MB)
"""
import os
import sys
import pickle
import argparse
import numpy as np

# Reuse load_dicom_volume_raw from convert_walnut_phys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_walnut_phys import load_dicom_volume_raw, dicom_dir_for


def main():
    parser = argparse.ArgumentParser(description="Walnut Total volume → reference pickle")
    parser.add_argument("--walnut", type=str, default="Walnut_1",
                        choices=["Walnut_1", "Walnut_2", "Walnut_3"],
                        help="Walnut id (defaults recon_base + output paths)")
    parser.add_argument("--recon_base", type=str, default=None,
                        help="Override recon dir (default: Reconstructions_mat/{walnut}/FDK_Dose_1_hann_TV_100_20)")
    parser.add_argument("--vol_size", type=int, nargs=3, default=[256, 256, 256])
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
    print("Walnut PCCT Total → 256³ reference volume (for M9 universal comparison)")
    print(f"  vol_size={args.vol_size}")
    print("=" * 70)

    dicom_dir = dicom_dir_for(args.recon_base, "Total")
    assert os.path.isdir(dicom_dir), f"Not found: {dicom_dir}"

    if args.output is None:
        subdir = "" if args.walnut == "Walnut_1" else f"{args.walnut.lower()}/"
        args.output = (f"../SAX-NeRF/data/res_{args.vol_size[0]}/v3_phys/"
                       f"{subdir}walnut_total_ref.pickle")

    print(f"\n[1] Loading Total volume")
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
            "source": f"convert_walnut_total.py M1-B physics-aligned mode (eff_E={args.eff_energy_keV})",
        }
    else:
        print(f"\n[2] Min-max normalize to [0, 1]")
        raw_min, raw_max = float(vol.min()), float(vol.max())
        image = (vol - raw_min) / (raw_max - raw_min + 1e-12)
        image = image.astype(np.float32)
        print(f"  raw HU range:    [{raw_min:.2f}, {raw_max:.2f}]")
        print(f"  post-norm range: [{image.min():.4f}, {image.max():.4f}], mean={image.mean():.4f}")
        norm = {
            "raw_min": raw_min, "raw_max": raw_max,
            "calib": "minmax_to_0_1",
            "source": "convert_walnut_total.py — polychromatic Total reconstruction",
        }

    data = {
        "image": image,
        "energy": "Total",
        "norm": norm,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    fsize = os.path.getsize(args.output) / 1e6
    print(f"\n[3] Saved: {args.output} ({fsize:.1f} MB)")

    n_total = int(image.size)
    n_nonair = int((image > 0.05).sum())
    print(f"\nNon-air voxels (image > 0.05): {n_nonair:,} / {n_total:,} "
          f"({100*n_nonair/n_total:.2f}%)")
    print("Done.")


if __name__ == "__main__":
    main()
