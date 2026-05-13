"""
Convert two energies of Walnut PCCT to a single dual-energy SAX-NeRF pickle (v3_phys).

KEY DIFFERENCE from convert_walnut_dual_v3.py
─────────────────────────────────────────────
The previous v3 normalization (subtract air HU, divide by shared_max) accidentally
enforces a WRONG inequality in HU space:
    HU_low − HU_high ≥ air_low_HU − air_high_HU = −299
That is an additive inequality. The actual X-ray physics inequality
    μ_real_low ≥ μ_real_high
expanded via standard CT calibration HU = 1000(μ−μ_water)/μ_water becomes a
MULTIPLICATIVE inequality involving μ_water at each energy:
    HU_low_adj ≥ R × HU_high_adj + 1000(R−1),  R = μ_water_high / μ_water_low ≈ 0.254

Empirical check on raw DICOM (1.41M material voxels in walnut):
   M1's effective additive inequality:  73% violations
   Proper physics inequality:           0.63% violations
The dataset is physically correct; M1's transform was the bug.

NEW PHYSICS-CALIBRATED NORMALIZATION
────────────────────────────────────
  2a. Per-energy air shift to standard HU (-1000):
         HU_adj = HU + (-1000 - air_HU)
  2b. Convert to absolute μ via standard HU formula (NIST water values):
         μ(E) = μ_water(E) × (HU_adj/1000 + 1)
         where μ_water(E) is from NIST XCOM database.
  2c. Global scale: scale = mu_low.max(); image = clip(μ/scale, 0, 1).
         Both energies divided by the SAME scale, preserving absolute μ ratio.

Output pickle structure (compatible with v3_dual + extra norm keys):
  {
    # Geometry / split — same as v3_dual
    "DSD", ..., "energy_low", "energy_high",
    "image_low":  ndarray (R,R,R) in [0, 1],
    "image_high": ndarray (R,R,R) in [0, ~0.13],   # ~mu_water_high/mu_water_low × max ratio
    "train": {...}, "val": {...},
    # New norm dict
    "norm": {
        "air_low": ..., "air_high": ...,
        "mu_water_low": ..., "mu_water_high": ...,
        "scale": ...,                 # mu_low.max() — REUSE for single-energy pickles
        "calib": "phys_air_shift_to_-1000_then_NIST_water",
    }
  }
"""

import os
import pickle
import argparse
import numpy as np
import pydicom
from scipy.ndimage import zoom
from glob import glob

import tigre
from tigre.utilities.geometry import Geometry


# ──────────────────────────────────────────────
# NIST XCOM mass attenuation coefficients of water (cm²/g)
# Source: https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html
# Note: ρ_water = 1.000 g/cm³, so μ_linear (cm⁻¹) = (μ/ρ) numerically.
# ──────────────────────────────────────────────
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
# Step 1: Load DICOM volume in RAW HU (no normalization)
# ──────────────────────────────────────────────

def load_dicom_volume_raw(dicom_dir, target_shape, remove_container=True):
    """Load DICOM, transpose to (X,Y,Z), air-floor, container-remove, downsample,
    post-air-floor. Returns volume in absolute HU values (NOT normalized)."""
    dcm_files = sorted(glob(os.path.join(dicom_dir, "*.dcm")))
    assert len(dcm_files) > 0, f"No DICOM files found in {dicom_dir}"
    print(f"  Loading {len(dcm_files)} DICOM files from {dicom_dir}")

    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(f)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        arr = arr * slope + intercept
        slices.append(arr)

    volume = np.stack(slices, axis=0)
    print(f"  Raw (Z,Y,X): {volume.shape}, range=[{volume.min():.2f}, {volume.max():.2f}]")

    volume = np.transpose(volume, (2, 1, 0))

    air_val = volume.min()
    air_mask = volume < -600
    volume[air_mask] = air_val

    if remove_container:
        sx, sy, _ = volume.shape
        cx, cy = sx // 2, sy // 2
        mask_r = 370
        Y, X = np.ogrid[:sx, :sy]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        volume[dist > mask_r, :] = air_val
        print(f"  Container mask: r={mask_r} @ center=({cx},{cy})")

    if volume.shape != target_shape:
        scale = [t / s for t, s in zip(target_shape, volume.shape)]
        print(f"  Downsampling {volume.shape} -> {target_shape}")
        volume = zoom(volume, scale, order=3, prefilter=False)

    post_air = volume < (air_val + 200)
    volume[post_air] = volume.min()

    print(f"  Pre-norm HU: shape={volume.shape}, range=[{volume.min():.2f}, {volume.max():.2f}]")
    return volume.astype(np.float32)


# ──────────────────────────────────────────────
# Step 2: Forward project both with shared geometry/angles
# ──────────────────────────────────────────────

def generate_dual_pickle(image_low, image_high, energy_low, energy_high,
                         n_train=50, n_val=50, seed=None):
    assert image_low.shape == image_high.shape, "Volume shapes must match"
    res = image_low.shape[0]
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
        "energy_low": energy_low,
        "energy_high": energy_high,
    }
    data["image_low"] = image_low.copy()
    data["image_high"] = image_high.copy()

    geo = ConeGeometry_special(data)
    img_low_t = np.transpose(image_low, (2, 1, 0)).copy()
    img_high_t = np.transpose(image_high, (2, 1, 0)).copy()

    train_angles = np.linspace(
        0, data["totalAngle"] / 180 * np.pi, n_train + 1
    )[:-1] + data["startAngle"] / 180 * np.pi
    data["train"] = {"angles": train_angles}

    print(f"  TIGRE Ax (train, {n_train} angles, low)...")
    train_low = tigre.Ax(img_low_t, geo, train_angles)[:, ::-1, :]
    print(f"  TIGRE Ax (train, {n_train} angles, high)...")
    train_high = tigre.Ax(img_high_t, geo, train_angles)[:, ::-1, :]
    data["train"]["projections_low"] = train_low
    data["train"]["projections_high"] = train_high

    if seed is not None:
        np.random.seed(seed)
    val_angles = np.sort(np.random.rand(n_val) * np.pi) + data["startAngle"] / 180 * np.pi
    data["val"] = {"angles": val_angles}

    print(f"  TIGRE Ax (val, {n_val} angles, low)...")
    val_low = tigre.Ax(img_low_t, geo, val_angles)[:, ::-1, :]
    print(f"  TIGRE Ax (val, {n_val} angles, high)...")
    val_high = tigre.Ax(img_high_t, geo, val_angles)[:, ::-1, :]
    data["val"]["projections_low"] = val_low
    data["val"]["projections_high"] = val_high

    return data


# ──────────────────────────────────────────────
# Step 3: Sanity check (FIXED: percentages over non-air voxels)
# ──────────────────────────────────────────────

def sanity_check_dual_phys(data, air_thresh_image=0.05):
    img_l, img_h = data["image_low"], data["image_high"]
    print()
    print("=" * 70)
    print(f"M1.5 sanity: dual phys pickle {data['energy_low']} + {data['energy_high']}")
    print("=" * 70)
    print(f"image_low:  shape={img_l.shape}, range=[{img_l.min():.4f}, {img_l.max():.4f}], mean={img_l.mean():.4f}")
    print(f"image_high: shape={img_h.shape}, range=[{img_h.min():.4f}, {img_h.max():.4f}], mean={img_h.mean():.4f}")
    print(f"norm dict: {data.get('norm', {})}")

    # Material mask: at least one channel above air threshold
    mat = (img_l > air_thresh_image) | (img_h > air_thresh_image)
    n_total = int(img_l.size)
    n_mat = int(mat.sum())
    print(f"\nMaterial voxels (image > {air_thresh_image} at EITHER channel): "
          f"{n_mat:,} / {n_total:,} ({100*n_mat/n_total:.2f}%)")

    # Physics: pointwise mu_low >= mu_high (in image space, equivalent to μ space because of shared scale)
    diff = img_l - img_h
    n_viol_total = int((diff < 0).sum())
    n_viol_mat = int(((diff < 0) & mat).sum())
    print()
    print(f"[Physics] Pointwise image_low ≥ image_high (≡ μ_low ≥ μ_high under shared-scale calib):")
    print(f"  violations / TOTAL voxels:    {n_viol_total:,} / {n_total:,} ({100*n_viol_total/n_total:.4f}%)  ← old v3 metric")
    if n_mat > 0:
        print(f"  violations / MATERIAL voxels: {n_viol_mat:,} / {n_mat:,} ({100*n_viol_mat/n_mat:.4f}%)  ← meaningful metric")
    if n_mat > 0 and n_viol_mat / n_mat < 0.01:
        print(f"  ✓ < 1% material-voxel violation — physics calibration OK")
    elif n_mat > 0 and n_viol_mat / n_mat < 0.05:
        print(f"  ✓ < 5% material-voxel violation — acceptable (likely recon noise)")
    else:
        print(f"  ✗ HIGH material-voxel violation — check calibration")

    # H1 precheck: ratio histogram on material voxels
    mask = (img_l > air_thresh_image) & (img_h > air_thresh_image)  # both > thresh for ratio
    n_both = int(mask.sum())
    if n_both > 0:
        ratio_hl = img_h[mask] / (img_l[mask] + 1e-6)
        ratio_lh = img_l[mask] / (img_h[mask] + 1e-6)
        print()
        print(f"[H1-precheck] ratios on {n_both:,} both-material voxels:")
        print(f"  μ_high / μ_low percentiles:")
        for p in [5, 25, 50, 75, 95]:
            print(f"    {p:2}%: {np.percentile(ratio_hl, p):.4f}")
        print(f"  μ_low / μ_high percentiles (target ~3.5-4.5 for walnut):")
        for p in [5, 25, 50, 75, 95]:
            print(f"    {p:2}%: {np.percentile(ratio_lh, p):.4f}")
        # ASCII histogram of ratio_lh
        hist, edges = np.histogram(ratio_lh, bins=12, range=(0, 12))
        bar_max = max(int(hist.max()), 1)
        print(f"  histogram μ_low/μ_high (12 bins, range [0,12]):")
        for i in range(len(hist)):
            bar = "#" * int(40 * hist[i] / bar_max)
            print(f"    [{edges[i]:.2f}, {edges[i+1]:.2f}): {hist[i]:8d}  {bar}")

    # Projection sanity
    print()
    print("[Projections]")
    for split in ("train", "val"):
        pl = data[split]["projections_low"]
        ph = data[split]["projections_high"]
        print(f"  {split} low:  shape={pl.shape}, range=[{pl.min():.6f}, {pl.max():.6f}], mean={pl.mean():.6f}")
        print(f"  {split} high: shape={ph.shape}, range=[{ph.min():.6f}, {ph.max():.6f}], mean={ph.mean():.6f}")
        if pl.mean() <= ph.mean():
            print(f"  ⚠ {split}: mean(proj_low) <= mean(proj_high) — physically suspicious")
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
        description="Dual-energy walnut pickle with PHYSICS calibration (v3_phys)")
    parser.add_argument("--low_energy", type=str, default="20Kev")
    parser.add_argument("--high_energy", type=str, default="60Kev")
    parser.add_argument("--walnut", type=str, default="Walnut_1",
                        choices=["Walnut_1", "Walnut_2", "Walnut_3"],
                        help="Walnut id (defaults recon_base + output paths)")
    parser.add_argument("--recon_base", type=str, default=None,
                        help="Override recon dir (default: Reconstructions_mat/{walnut}/FDK_Dose_1_hann_TV_100_20)")
    parser.add_argument("--vol_size", type=int, nargs=3, default=[256, 256, 256])
    parser.add_argument("--n_train", type=int, default=50)
    parser.add_argument("--n_val", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for val angles (default: nondeterministic)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no_remove_container", action="store_true")
    args = parser.parse_args()

    if args.recon_base is None:
        args.recon_base = f"Reconstructions_mat/{args.walnut}/FDK_Dose_1_hann_TV_100_20"

    # Look up NIST water μ values
    if args.low_energy not in MU_WATER:
        raise ValueError(f"low_energy {args.low_energy} not in MU_WATER table; available: {list(MU_WATER)}")
    if args.high_energy not in MU_WATER:
        raise ValueError(f"high_energy {args.high_energy} not in MU_WATER table; available: {list(MU_WATER)}")
    mu_water_low = MU_WATER[args.low_energy]
    mu_water_high = MU_WATER[args.high_energy]

    print("=" * 70)
    print(f"Walnut PCCT -> dual SAX-NeRF pickle (v3_phys, PHYSICS-CALIBRATED):")
    print(f"  low={args.low_energy} (μ_water={mu_water_low:.4f} cm²/g)")
    print(f"  high={args.high_energy} (μ_water={mu_water_high:.4f} cm²/g)")
    print(f"  vol_size={args.vol_size}")
    print(f"  expected ratio μ_water_low/μ_water_high = {mu_water_low/mu_water_high:.3f}")
    print("=" * 70)

    dicom_low = dicom_dir_for(args.recon_base, args.low_energy)
    dicom_high = dicom_dir_for(args.recon_base, args.high_energy)
    assert os.path.isdir(dicom_low), f"Not found: {dicom_low}"
    assert os.path.isdir(dicom_high), f"Not found: {dicom_high}"

    if args.output is None:
        tag_l = args.low_energy.lower()
        tag_h = args.high_energy.lower()
        subdir = "" if args.walnut == "Walnut_1" else f"{args.walnut.lower()}/"
        args.output = (f"../SAX-NeRF/data/res_{args.vol_size[0]}/v3_dual_phys/"
                       f"{subdir}walnut_{tag_l}_{tag_h}_{args.n_train}.pickle")

    # Step 1: load both volumes (raw HU)
    print(f"\n[1a] Loading low-energy volume ({args.low_energy})")
    vol_low = load_dicom_volume_raw(
        dicom_low, target_shape=tuple(args.vol_size),
        remove_container=not args.no_remove_container)

    print(f"\n[1b] Loading high-energy volume ({args.high_energy})")
    vol_high = load_dicom_volume_raw(
        dicom_high, target_shape=tuple(args.vol_size),
        remove_container=not args.no_remove_container)

    # Step 2: PHYSICS-CALIBRATED normalization
    print(f"\n[2] Physics calibration (per-energy air shift -> -1000 + NIST μ_water + global scale)")
    air_low = float(vol_low.min())
    air_high = float(vol_high.min())
    print(f"  air HU (raw):  low={air_low:.2f}, high={air_high:.2f}")

    # 2a. per-energy air shift to standard HU (-1000)
    shift_low = -1000.0 - air_low
    shift_high = -1000.0 - air_high
    hu_low_adj = vol_low + shift_low
    hu_high_adj = vol_high + shift_high
    print(f"  air-shift:     low +{shift_low:.0f} (-> -1000), high +{shift_high:.0f} (-> -1000)")

    # 2b. convert HU_adj to absolute μ via standard CT formula
    mu_low = mu_water_low * (hu_low_adj / 1000.0 + 1.0)
    mu_high = mu_water_high * (hu_high_adj / 1000.0 + 1.0)
    # Floor at 0 (some negative values possible from cubic spline downsampling)
    mu_low = np.clip(mu_low, 0.0, None)
    mu_high = np.clip(mu_high, 0.0, None)
    print(f"  μ_low  (cm²/g) range=[{mu_low.min():.4f}, {mu_low.max():.4f}] mean={mu_low.mean():.4f}")
    print(f"  μ_high (cm²/g) range=[{mu_high.min():.4f}, {mu_high.max():.4f}] mean={mu_high.mean():.4f}")

    # 2c. global scale = mu_low.max(), divide BOTH by same scale
    scale = float(mu_low.max())
    print(f"  global scale (= mu_low.max()): {scale:.4f}  ← REUSE for single-energy pickles")

    image_low = np.clip(mu_low / scale, 0.0, 1.0).astype(np.float32)
    image_high = np.clip(mu_high / scale, 0.0, 1.0).astype(np.float32)
    print(f"  post-norm: image_low [{image_low.min():.4f}, {image_low.max():.4f}],"
          f"  image_high [{image_high.min():.4f}, {image_high.max():.4f}]")

    # Step 3: forward project
    print(f"\n[3] Forward projection (shared geometry, shared angles)")
    data = generate_dual_pickle(
        image_low, image_high, args.low_energy, args.high_energy,
        n_train=args.n_train, n_val=args.n_val, seed=args.seed)

    # Step 4: norm dict (new schema: keep shared_max for backwards compat)
    data["norm"] = {
        "air_low": air_low, "air_high": air_high,
        "shift_low": shift_low, "shift_high": shift_high,
        "mu_water_low": mu_water_low, "mu_water_high": mu_water_high,
        "scale": scale,                           # ← reuse for single-energy
        "shared_max": scale,                      # alias for backward compat with v3_dual loaders
        "calib": "phys_air_shift_to_-1000_then_NIST_water",
    }

    # Step 5: save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    fsize = os.path.getsize(args.output) / 1e6
    print(f"\n[5] Saved: {args.output} ({fsize:.1f} MB)")

    # Step 6: sanity
    sanity_check_dual_phys(data)
    print(f"\nDone. Use scale = {scale:.6f} when converting single-energy pickles for the same dual setup.")


if __name__ == "__main__":
    main()
