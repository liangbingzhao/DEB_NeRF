"""
Generate phys-calibrated GT volumes at multiple held-out energies for M6 Part B
(virtual mono image / VMI synthesis evaluation).

Reuses the calibration formula from convert_walnut_phys.py but:
  - Skips TIGRE forward projection (we only need the GT image for PSNR comparison
    against synthesized μ, not projections — saves ~5 min/energy).
  - Loops over 5 held-out energies (30, 40, 50, 70, 80 keV).
  - Uses the SAME global_scale as the dual phys pickle's norm.scale, so the
    output `image` is on the same normalized scale as the dual model's outputs.

Output: SAX-NeRF/data/res_256/v3_phys_multi/walnut_{E}kev.pickle
        Each pickle contains {"image": 256³ float32, "energy": "30Kev", "norm": ...}.

Usage:
  python data_preprocess/convert_walnut_v3_phys_multi.py
"""
import os
import os.path as osp
import pickle
import sys

import numpy as np

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.join(REPO, "data_preprocess"))

from convert_walnut_phys import (
    MU_WATER, load_dicom_volume_raw, dicom_dir_for,
)


HELD_OUT_ENERGIES = ["30Kev", "40Kev", "50Kev", "70Kev", "80Kev"]
DUAL_PICKLE = osp.join(REPO, "SAX-NeRF/data/res_256/v3_dual_phys/walnut_20kev_60kev_50.pickle")
RECON_BASE = osp.join(REPO, "data_preprocess/Reconstructions/Walnut_1/FDK_Dose_1_hann_TV_100_20")
OUT_DIR = osp.join(REPO, "SAX-NeRF/data/res_256/v3_phys_multi")
VOL_SIZE = (256, 256, 256)


def get_dual_scale() -> float:
    with open(DUAL_PICKLE, "rb") as f:
        d = pickle.load(f)
    return float(d["norm"]["scale"])


def convert_one(energy: str, global_scale: float) -> dict:
    mu_water = MU_WATER[energy]
    print(f"\n=== {energy} (μ_water={mu_water:.4f}) ===")

    dicom_dir = dicom_dir_for(RECON_BASE, energy)
    if not os.path.isdir(dicom_dir):
        raise FileNotFoundError(f"DICOM dir not found: {dicom_dir}")

    vol = load_dicom_volume_raw(dicom_dir, target_shape=VOL_SIZE, remove_container=True)

    air = float(vol.min())
    shift = -1000.0 - air
    hu_adj = vol + shift
    mu = mu_water * (hu_adj / 1000.0 + 1.0)
    mu = np.clip(mu, 0.0, None)
    image = np.clip(mu / global_scale, 0.0, 1.0).astype(np.float32)
    print(f"  air HU={air:.1f}, shift=+{shift:.1f}, μ_abs range=[{mu.min():.4f}, {mu.max():.4f}], "
          f"image range=[{image.min():.4f}, {image.max():.4f}]")

    return {
        "image": image,
        "energy": energy,
        "norm": {
            "air": air,
            "shift": shift,
            "mu_water": mu_water,
            "scale": global_scale,
            "calib": "phys_air_shift_to_-1000_then_NIST_water",
            "source": "convert_walnut_v3_phys_multi.py (no projections; for VMI synth eval)",
        },
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Reading dual pickle scale from: {DUAL_PICKLE}")
    scale = get_dual_scale()
    print(f"global_scale = {scale:.6f}\n")

    for energy in HELD_OUT_ENERGIES:
        out_path = osp.join(OUT_DIR, f"walnut_{energy.lower()}.pickle")
        if osp.exists(out_path):
            print(f"  SKIP existing: {out_path}")
            continue
        data = convert_one(energy, scale)
        with open(out_path, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        size_mb = osp.getsize(out_path) / 1e6
        print(f"  saved: {out_path} ({size_mb:.1f} MB)")

    print(f"\nDone. {len(HELD_OUT_ENERGIES)} pickles in {OUT_DIR}")


if __name__ == "__main__":
    main()
