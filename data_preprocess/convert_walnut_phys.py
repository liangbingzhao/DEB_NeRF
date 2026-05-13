"""
Convert single-energy Walnut PCCT DICOM → physics-calibrated SAX-NeRF pickle.
Mirror of convert_walnut_dual_phys.py for one energy at a time.

KEY DIFFERENCES from convert_walnut_v3.py
─────────────────────────────────────────
v3 used min-max normalization:    image = (HU - HU.min) / (HU.max - HU.min)
This destroys the absolute-μ scale needed for cross-energy comparability.

v3_phys (this script) does physics calibration:
  2a. Per-energy air shift to standard HU:  HU_adj = HU + (-1000 - air_HU)
  2b. Convert to absolute μ via NIST water: μ = μ_water(E) × (HU_adj/1000 + 1)
  2c. Divide by SHARED scale (passed via --global_scale, taken from dual pickle's
      mu_low.max). This guarantees the single-energy pickle's image values are
      on the SAME scale as the dual pickle's image_low / image_high.

Usage (after running convert_walnut_dual_phys.py first to determine scale):
  python convert_walnut_phys.py \
      --energy 20Kev --vol_size 256 256 256 --global_scale 3.5672 --seed 42
  python convert_walnut_phys.py \
      --energy 60Kev --vol_size 256 256 256 --global_scale 3.5672 --seed 42

The --seed should match the dual pickle's seed so val angles align across pickles.
"""
import os
import sys
import pickle
import argparse
import numpy as np
import pydicom
from scipy.ndimage import zoom
from glob import glob

import tigre
from tigre.utilities.geometry import Geometry


# NIST XCOM mass attenuation coefficients of water (cm²/g)
# Source: https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html
MU_WATER = {
    "10Kev": 5.329,
    "20Kev": 0.8096,
    "30Kev": 0.3756,
    "40Kev": 0.2683,
    "50Kev": 0.2270,
    "60Kev": 0.2059,
    "70Kev": 0.1922,
    "80Kev": 0.1837,
    "90Kev": 0.1779,
    "100Kev": 0.1707,
}


# ──────────────────────────────────────────────
# Geometry (identical to convert_walnut_v3.py)
# ──────────────────────────────────────────────

class ConeGeometry_special(Geometry):
    def __init__(self, data):
        Geometry.__init__(self)
        self.DSD = data["DSD"] / 1000
        self.DSO = data["DSO"] / 1000
        self.nDetector = np.array(data["nDetector"])
        self.dDetector = np.array(data["dDetector"]) / 1000
        self.sDetector = self.nDetector * self.dDetector
        self.nVoxel = np.array(data["nVoxel"][::-1])
        self.dVoxel = np.array(data["dVoxel"][::-1]) / 1000
        self.sVoxel = self.nVoxel * self.dVoxel
        self.offOrigin = np.array(data["offOrigin"][::-1]) / 1000
        self.offDetector = np.array(
            [data["offDetector"][1], data["offDetector"][0], 0]) / 1000
        self.accuracy = data["accuracy"]
        self.mode = data["mode"]
        self.filter = data["filter"]


# ──────────────────────────────────────────────
# Step 1: load DICOM in raw HU (no normalization)
# ──────────────────────────────────────────────
def load_dicom_volume_raw(dicom_dir, target_shape, remove_container=True):
    dcm_files = sorted(glob(os.path.join(dicom_dir, "*.dcm")))
    assert len(dcm_files) > 0, f"No DICOM files found in {dicom_dir}"
    print(f"  Loading {len(dcm_files)} DICOM files from {dicom_dir}")
    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(f)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        slices.append(arr * slope + intercept)
    volume = np.stack(slices, axis=0)
    print(f"  Raw (Z,Y,X): {volume.shape}, range=[{volume.min():.2f}, {volume.max():.2f}]")
    volume = np.transpose(volume, (2, 1, 0))
    air_val = volume.min()
    volume[volume < -600] = air_val
    if remove_container:
        sx, sy, _ = volume.shape
        cx, cy = sx // 2, sy // 2
        Y, X = np.ogrid[:sx, :sy]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        volume[dist > 370, :] = air_val
    if volume.shape != target_shape:
        scale = [t / s for t, s in zip(target_shape, volume.shape)]
        print(f"  Downsampling {volume.shape} -> {target_shape}")
        volume = zoom(volume, scale, order=3, prefilter=False)
    post_air = volume < (air_val + 200)
    volume[post_air] = volume.min()
    print(f"  Pre-norm HU: shape={volume.shape}, range=[{volume.min():.2f}, {volume.max():.2f}]")
    return volume.astype(np.float32)


# ──────────────────────────────────────────────
# Step 2-5: project + pickle
# ──────────────────────────────────────────────
def generate_single_pickle(image, energy, n_train=50, n_val=50, seed=None):
    res = image.shape[0]
    dvoxel = 128.0 / res
    n_det = max(256, res * 2)

    data = {
        "DSD": 1500.0,
        "DSO": 1000.0,
        "nDetector": [n_det, n_det],
        "dDetector": [256.0 / n_det, 256.0 / n_det],
        "nVoxel": [res, res, res],
        "dVoxel": [dvoxel, dvoxel, dvoxel],
        "offOrigin": [0, 0, 0],
        "offDetector": [0, 0],
        "accuracy": 0.5,
        "mode": "cone",
        "filter": None,
        "totalAngle": 360.0,
        "startAngle": 0.0,
        "randomAngle": False,
        "convert": False,
        "rescale_slope": 1.0,
        "rescale_intercept": 0.0,
        "normalize": True,
        "noise": 0,
        "numTrain": n_train,
        "numVal": n_val,
        "energy": energy,
    }
    data["image"] = image.copy()

    geo = ConeGeometry_special(data)
    img_t = np.transpose(image, (2, 1, 0)).copy()

    train_angles = np.linspace(
        0, data["totalAngle"] / 180 * np.pi, n_train + 1
    )[:-1] + data["startAngle"] / 180 * np.pi
    data["train"] = {"angles": train_angles}
    print(f"  TIGRE Ax (train, {n_train} angles)...")
    train_proj = tigre.Ax(img_t, geo, train_angles)[:, ::-1, :]
    data["train"]["projections"] = train_proj

    if seed is not None:
        np.random.seed(seed)
    val_angles = np.sort(np.random.rand(n_val) * np.pi) + data["startAngle"] / 180 * np.pi
    data["val"] = {"angles": val_angles}
    print(f"  TIGRE Ax (val, {n_val} angles)...")
    val_proj = tigre.Ax(img_t, geo, val_angles)[:, ::-1, :]
    data["val"]["projections"] = val_proj

    return data


# ──────────────────────────────────────────────
# Sanity check (single-energy)
# ──────────────────────────────────────────────
def sanity_check_single(data, air_thresh_image=0.05):
    img = data["image"]
    print()
    print("=" * 70)
    print(f"M0.5 sanity: single-energy phys pickle ({data['energy']})")
    print("=" * 70)
    print(f"image: shape={img.shape}, range=[{img.min():.4f}, {img.max():.4f}], mean={img.mean():.4f}")
    print(f"norm dict: {data.get('norm', {})}")
    n_total = int(img.size)
    n_nonair = int((img > air_thresh_image).sum())
    print(f"\nNon-air voxels (image > {air_thresh_image}): "
          f"{n_nonair:,} / {n_total:,} ({100*n_nonair/n_total:.2f}%)")
    pl = data["train"]["projections"]
    pv = data["val"]["projections"]
    print(f"\n[Projections]")
    print(f"  train: shape={pl.shape}, range=[{pl.min():.6f}, {pl.max():.6f}], mean={pl.mean():.6f}")
    print(f"  val:   shape={pv.shape}, range=[{pv.min():.6f}, {pv.max():.6f}], mean={pv.mean():.6f}")
    print("=" * 70)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def dicom_dir_for(recon_base, energy):
    if energy == "Total":
        return os.path.join(recon_base, "Total")
    if "Kev" in energy or "kev" in energy:
        return os.path.join(recon_base, "VirtualMonoImg", energy)
    return os.path.join(recon_base, energy)


def main():
    parser = argparse.ArgumentParser(
        description="Single-energy walnut pickle with PHYSICS calibration (v3_phys)")
    parser.add_argument("--energy", type=str, required=True,
                        help=f"Energy: one of {list(MU_WATER)}")
    parser.add_argument("--global_scale", type=float, required=True,
                        help="SHARED normalization scale (= mu_low.max from dual pickle)")
    parser.add_argument("--walnut", type=str, default="Walnut_1",
                        choices=["Walnut_1", "Walnut_2", "Walnut_3"],
                        help="Walnut id (defaults recon_base + output paths)")
    parser.add_argument("--recon_base", type=str, default=None,
                        help="Override recon dir (default: Reconstructions_mat/{walnut}/FDK_Dose_1_hann_TV_100_20)")
    parser.add_argument("--vol_size", type=int, nargs=3, default=[256, 256, 256])
    parser.add_argument("--n_train", type=int, default=50)
    parser.add_argument("--n_val", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for val angles (default 42 to match dual pickle)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no_remove_container", action="store_true")
    args = parser.parse_args()

    if args.recon_base is None:
        args.recon_base = f"Reconstructions_mat/{args.walnut}/FDK_Dose_1_hann_TV_100_20"

    if args.energy not in MU_WATER:
        raise ValueError(f"energy={args.energy} not in MU_WATER table; available: {list(MU_WATER)}")
    mu_water = MU_WATER[args.energy]

    print("=" * 70)
    print(f"Walnut PCCT -> single-energy SAX-NeRF pickle (v3_phys, PHYSICS-CALIBRATED):")
    print(f"  energy={args.energy} (μ_water={mu_water:.4f} cm²/g)")
    print(f"  vol_size={args.vol_size}")
    print(f"  global_scale={args.global_scale:.6f}")
    print(f"  seed={args.seed}")
    print("=" * 70)

    dicom_dir = dicom_dir_for(args.recon_base, args.energy)
    assert os.path.isdir(dicom_dir), f"Not found: {dicom_dir}"

    if args.output is None:
        tag = args.energy.lower()
        subdir = "" if args.walnut == "Walnut_1" else f"{args.walnut.lower()}/"
        args.output = (f"../SAX-NeRF/data/res_{args.vol_size[0]}/v3_phys/"
                       f"{subdir}walnut_{tag}_{args.n_train}.pickle")

    # Step 1: load DICOM
    print(f"\n[1] Loading volume ({args.energy})")
    vol = load_dicom_volume_raw(dicom_dir, target_shape=tuple(args.vol_size),
                                remove_container=not args.no_remove_container)

    # Step 2: physics calibration
    print(f"\n[2] Physics calibration (air shift -> -1000 + NIST μ_water + global scale)")
    air = float(vol.min())
    shift = -1000.0 - air
    hu_adj = vol + shift
    mu = mu_water * (hu_adj / 1000.0 + 1.0)
    mu = np.clip(mu, 0.0, None)
    print(f"  air HU (raw): {air:.2f}, shift: +{shift:.0f} (-> -1000)")
    print(f"  μ (cm²/g) range=[{mu.min():.4f}, {mu.max():.4f}] mean={mu.mean():.4f}")

    image = np.clip(mu / args.global_scale, 0.0, 1.0).astype(np.float32)
    print(f"  global scale (= dual pickle mu_low.max): {args.global_scale:.4f}")
    print(f"  post-norm: image range=[{image.min():.4f}, {image.max():.4f}]")
    print(f"  (note: for high energies image.max < 1 because of shared scale; this is correct)")

    # Step 3: forward project
    print(f"\n[3] Forward projection")
    data = generate_single_pickle(image, args.energy,
                                  n_train=args.n_train, n_val=args.n_val,
                                  seed=args.seed)

    # Step 4: norm dict
    data["norm"] = {
        "air": air,
        "shift": shift,
        "mu_water": mu_water,
        "scale": args.global_scale,
        "calib": "phys_air_shift_to_-1000_then_NIST_water",
    }

    # Step 5: save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    fsize = os.path.getsize(args.output) / 1e6
    print(f"\n[5] Saved: {args.output} ({fsize:.1f} MB)")

    sanity_check_single(data)
    print(f"\nDone.")


if __name__ == "__main__":
    main()
