# Walnut PCCT Dataset: Physical Understanding

This document systematically records the physical structure of the Walnut PCCT dataset, the acquisition parameters, the reconstruction pipeline, and how the derived VMI (Virtual Monoenergetic Image) data is synthesized. Based on the Zenodo dataset metadata ([15738314](https://zenodo.org/records/15738314)), the original reconstruction code [zezisme/WalnutPCCTReconCodes](https://github.com/zezisme/WalnutPCCTReconCodes), and our local calibration tables and source code (`data_preprocess/CalibrationTable/`).

---

## 1. Detector Physics (Hardware During Acquisition)

### 1.1 X-ray Source

| Parameter | Value |
|---|---|
| Tube voltage | **80 kVp** |
| Tube current | 200 μA |
| Filter | 0.5 mm Al |
| Anode | Not explicitly stated (likely W) |
| Spectrum file | **Not provided** (no S(E) raw data shipped with the dataset) |

### 1.2 Detector (PCCT, Photon Counting)

| Parameter | Value |
|---|---|
| Resolution | 2063 × 505 pixels |
| Pixel size | 100 μm |
| Bit depth | 12-bit (max 4096 photon counts/pixel) |
| Exposure | 70 ms |
| **Energy thresholds** | **15 keV (low)**, **30 keV (high)** |

### 1.3 Physically There Are Only Two Bins

The PCCT detector reads out **two** photon-count streams:
- **Total bin**: all photons ≥ 15 keV (the entire spectrum above the low threshold)
- **High bin**: all photons ≥ 30 keV (above the high threshold)

**The Low bin is computed**:

```
Low = Total − High      (photons in the 15–30 keV range)
```

Source code (`WalnutPCCTReconCodes/functions/ProjDataPrepare.m`):
```matlab
file_list_high = dir([data_path '\High\*.raw']);
file_list_total = dir([data_path '\Total\*.raw']);
% ...
temp_total = reshape(proj, nChannelNum, nSliceNum);
temp_low = temp_total - temp_high;    % Low is constructed
```

**Important consequences**:
- Total = polychromatic — contains all photons from 15–80 keV mixed together
- High + Low = Total (photon conservation) — splits Total into two energy windows
- "Low" is *not* "low-energy reconstruction"; it is "reconstruction from the low-energy photon subset (15–30 keV bin)"

### 1.4 Acquisition Geometry

| Parameter | Value |
|---|---|
| Source-to-object distance (SOD) | 140 mm |
| Source-to-detector distance (SDD) | 325 mm |
| Field of view | 80 mm |
| Projections / rotation | 1440 (0.25° increments) |
| Bed positions | 4 (every 15 mm) |

---

## 2. Reconstruction Pipeline (Raw Projection → 3D Volume)

### 2.1 FDK Reconstruction of Total / High / Low

Source: `WalnutPCCTReconCodes/WalnutDataRecon.m` + `functions/ReconAllEnergy.m` + `ProjDataRecon.m`.

```matlab
recon_para.recon_Bin = [1 1 1];   % [Low, High, Total] — 1 = recon, 0 = skip
% Algorithm options: FDK, SART, MLEM (TIGRE toolbox)
```

Per bin, independently:
1. Read raw projections (counts).
2. Air correction: `-log(max(rawdata, 0)) + log(max(airtable, 0))` (Beer–Lambert).
3. Ring artifact correction (optional).
4. FDK / SART / MLEM reconstruction (TIGRE GPU).
5. Output as HU-format DICOM volume.

**Result**: 3 independent 3D volumes (Total, High, Low), each reconstructed separately.

### 2.2 VMI (Virtual Monoenergetic Image) Synthesis — Where the "10/20/30/.../80 keV" Volumes Come From

**Key fact**: the `walnut_10kev / walnut_20kev / … / walnut_80kev` volumes used in this project are **not physical bins** — they are synthesized from the two physical bins (Low, High) via **material decomposition + VMI synthesis**.

Source: `WalnutPCCTReconCodes/WalnutSpectralRecon.m` + `functions/ImageSpectralRecon.m`.

#### 2.2.1 Material Decomposition

Uses **water + HAP (Hydroxyapatite, Ca₁₀(PO₄)₆(OH)₂)** as two basis materials, performing pixel-wise least squares in image space:

```matlab
% Given each voxel's (μ_low, μ_high):
DM = [Watercali(1,1), HAPcali(1,1);     % 2x2 calibration matrix
      Watercali(1,2), HAPcali(1,2)];
M = DM \ [μ_low; μ_high];               % Solve (M_water, M_HAP)
```

**Output**: per-voxel `(M_water, M_HAP)` — equivalent density (or thickness) of water and HAP.

#### 2.2.2 VMI Synthesis (Project to Any Target Energy)

Use NIST mass-attenuation coefficient tables to project (M_water, M_HAP) to any energy E:

```matlab
% Energy list (dataset ships 9):
recon_para.WalnutVMI_E = 10:10:80;     % 10, 20, 30, 40, 50, 60, 70, 80 keV

% Synthesis (NIST Beer–Lambert)
lineatten_keV = MaterialPair.M1 * H2O_massAttenuationCoeff(Kevs(k))
              + MaterialPair.M2 * HA_massAttenuationCoeff(Kevs(k));
```

**So all 9 keV volumes come from the same (Low, High) physical projections and the same NIST tables**. Information-theoretically, the 9 VMI volumes together carry no more information than the two physical bins.

#### 2.2.3 Mathematical Equivalence with Our "Path D"

Our project's **Path D (basis2)** is doing the *same thing*, but replaces the closed-form least squares with neural-network learning:

| Concept | Original repo | Our Path D |
|---|---|---|
| Basis material 1 | water (M1) | α_w |
| Basis material 2 | HAP (M2) | α_2 |
| κ_basis 1 at energy E | `H2O_massAttenuationCoeff(E)` | `κ_w(E)` |
| κ_basis 2 at energy E | `HA_massAttenuationCoeff(E)` | `κ_2(E)` (**learnable**) |
| Decomposition domain | image space (pixel-wise LS) | voxel space (NN globally) |
| Solver | `DM \ img` (closed form) | NN gradient descent |

→ Path D is a **NeRF-ified version of the classical spectral CT algorithm**. The "bone init" motivation comes directly from this: HAP is the mineral component of bone, and the original repo uses HAP.

---

## 3. Dataset File Layout (Zenodo Structure)

```
Walnut_X/
├── ProjData/                              # Raw projection data
│   ├── couch_{1..4}/
│   │   ├── Total/*.raw                    # Total bin counts (1200 angles)
│   │   └── High/*.raw                     # High bin counts
│   └── ...
├── Reconstructions/Walnut_X/
│   ├── FDK_Dose_1_hann_TV_100_20/
│   │   ├── Total/dicom/*.dcm              # Total FDK reconstruction (polychromatic)
│   │   ├── High/dicom/*.dcm               # High-bin reconstruction (≥ 30 keV photons)
│   │   ├── Low/dicom/*.dcm                # Low = Total − High reconstruction (15–30 keV photons)
│   │   ├── 10kev/dicom/*.dcm              # VMI at 10 keV (synthesized from water+HAP decomposition)
│   │   ├── 20kev/dicom/*.dcm              # VMI at 20 keV
│   │   ├── ...
│   │   └── 80kev/dicom/*.dcm              # VMI at 80 keV
└── ...
```

Our `dicom_dir_for(energy)` in `data_preprocess/convert_walnut_phys.py` loads from these DICOM paths.

---

## 4. HU Calibration Meaning

CT reconstruction volumes are not stored as physical μ (cm⁻¹) but as **HU (Hounsfield Units)**:

```
HU(x, E) = 1000 × (μ(x, E) − μ_water(E)) / μ_water(E)
```

Physical meaning:
- HU = 0 → voxel equivalent to water (μ = μ_water(E))
- HU = -1000 → air (μ = 0)
- HU > 0 → attenuates more than water
- HU = 1000 → μ is twice that of water

**Key invariance**: HU is normalized by water, so for water/air the HU values are stable across energies (HU(water) = 0 at every energy, HU(air) = -1000). **Only non-water materials** (e.g. shell calcium) have HU values that change with energy.

The HU ranges actually observed in this dataset:
- 20 keV VMI: [-2092, 2314] (air HU calibrated to -2092, not -1000; needs +1092 shift)
- 60 keV VMI: [-1793, 406] (needs +793 shift)
- Total: [-1263, 726] (needs +263 shift)

**Why the air HU differs across energies**: the original dataset's reconstruction is not strictly standard CT calibration (where air should always be -1000). Each energy volume has its own calibration drift. Our `convert_walnut_phys.py` standardizes air to -1000 with `air = vol.min(); shift = -1000 - air; HU_adj = HU + shift`.

---

## 5. v3_phys Normalisation (Project's NIST Physical Calibration)

Source: `data_preprocess/convert_walnut_phys.py:246-258`.

```python
# Given a raw HU volume:
air = float(vol.min())                        # the dataset's air HU value
shift = -1000.0 - air                         # correct to the standard -1000
hu_adj = vol + shift                          # now air sits at -1000
mu = mu_water * (hu_adj / 1000.0 + 1.0)      # NIST: μ = μ_water × (HU_adj/1000 + 1)
mu = np.clip(mu, 0.0, None)                  # physical constraint μ ≥ 0
image = np.clip(mu / global_scale, 0.0, 1.0) # normalize to [0, 1]
```

`mu_water(E)` is the NIST water mass-attenuation coefficient × density at energy E:
```python
MU_WATER = {
    "10Kev": 5.329, "20Kev": 0.8096, "30Kev": 0.3756,
    "40Kev": 0.2683, "50Kev": 0.2270, "60Kev": 0.2059,
    "70Kev": 0.1922, "80Kev": 0.1837, "90Kev": 0.1779,
}
# Source: NIST XCOM database
# https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html
```

`global_scale = 3.567219` is `mu_low.max` from the dual-energy pickle — it makes `image_low.max ≈ 1.0` by construction.

**Key design choice**: all energies use the **same global_scale** (not per-energy minmax to [0,1]). Consequences:
- `image_low` (20 keV) max ≈ 1.0 (by design)
- `image_high` (60 keV) max ≈ 0.127 (≈ μ_water(60)/μ_water(20) = 0.2059 / 0.8096)
- The **physical ratio** between energies is preserved (`image_high / image_low ≈ μ_60 / μ_20`)

This is a necessary condition for dual-energy training — independent minmax would destroy the physical ratio and the network could not learn spectral differences.

---

## 6. The Specialness of "Total"

The Total bin is **polychromatic** — 15–80 keV mixed together. It **has no single effective energy**.

Our `convert_walnut_total.py` by default does **independent minmax to [0,1]**:
```python
image = (vol - vol.min()) / (vol.max() - vol.min())
```

**Issue**: this is inconsistent with the v3_phys NIST shared-scale system — "1.0" in independent minmax does not physically mean the same thing as "1.0" in the NIST system.

**Empirical effective-energy fit** (explored 2026-05-04):
- Fit using NIST water table + voxel-wise L2 (1.4 M voxels, R² = 0.79)
- Assumption: each voxel μ(E) ∝ μ_water(E) (water-like approximation, valid for materials close to water)
- Result: **E_eff(Total) ≈ 43.5 keV**

**Caveat**: this depends on the water-like assumption. Shell regions, due to beam hardening, actually have higher effective E (45–55 keV); water-rich regions have lower (35–40 keV). **Total really has no single effective energy** — it is a distribution, not a number.

The `--physics_calib` mode of `convert_walnut_total.py` (added 2026-05-04, off by default) supports calibrating Total to any effective energy via the NIST formula (defaults to 43.5 keV, global_scale = 3.567). Outputs from that mode (`*_aligned.pickle`) are not used in the main evaluation.

---

## 7. Local Calibration Files

`data_preprocess/CalibrationTable/`:

| File | Content |
|---|---|
| `H2O_massAttenuationCoeff.mat` | NIST water μ/ρ at 1, 2, …, 160 keV (cm²/g) |
| `HAP_massAttenuationCoeff.mat` | NIST HAP μ/ρ at 1, 2, …, 160 keV (cm²/g) |
| `Watercali.mat` | Water bin reference values [Low: 1000, High: 1000] (empirical calibration) |
| `HAPcali.mat` | HAP bin reference values [Low: 4701, High: 2977] (empirical calibration) |
| `HU_water_table.mat` | Water raw-signal values per bin: low = 22225, high = 39737, total = 30483 |
| `WalnutMDTable.mat` | Multi-walnut decomposition table per walnut individual |

Reading (`scipy.io.loadmat` cannot read MATLAB v7.3+; need `h5py`):
```python
import h5py
with h5py.File('H2O_massAttenuationCoeff.mat', 'r') as f:
    nist_water = f['H2O_massAttenuationCoeff'][()].flatten()  # 160 keV values
mu_water_at_E = nist_water[E - 1]  # E in keV, 1-indexed
```

---

## 8. NIST Standard vs. Project Values (Key Reference)

| Material | Physical μ at 20 keV | Physical μ at 60 keV | Project κ_norm @ 20 | Project κ_norm @ 60 |
|---|---:|---:|---:|---:|
| **Water** | 0.8096 cm⁻¹ | 0.2059 cm⁻¹ | 1.000 (κ_w_low) | 0.2543 (κ_w_high) |
| **Cortical bone** | ~4.00 cm⁻¹ | ~0.305 cm⁻¹ | **1.121** (Path D bone init) | **0.0855** |
| **HAP** (ρ = 3.16) | **4.17 cm⁻¹** | **0.45 cm⁻¹** | **1.169** (Path D HAP init) | **0.126** |

Conversion: `κ_norm = μ_NIST / 3.567219` (shared v3_phys global_scale).

**Notes**:
- "Cortical bone" is HAP + water + organic matrix (ρ ≈ 1.85, lighter than pure HAP).
- Our Path D default init uses cortical bone, not pure HAP.
- The original repo `WalnutPCCTReconCodes` uses **pure HAP** (ρ = 3.16).
- The two are close at 20 keV (4 % difference) but differ by 47 % at 60 keV (HAP is heavier).

---

## 9. Design Decisions Informed by This Dataset Understanding

The physical bases for several key project decisions:

### 9.1 v3_phys Uses Shared Scale, Not Independent Minmax
- Dual-energy physical ratio must be preserved; otherwise Path B/D cannot learn spectral information.
- See §5.

### 9.2 Path D Uses water + HAP-like as Basis
- Directly mirrors the original repo's spectral-recon math.
- Bone init is HAP-like but slightly lighter.
- **2026-05-04**: added pure-HAP init experiment (jobs 46883678 / 46883679).

### 9.3 Path D Outputs ρ_total = α_w + α_2 Anchored to 20 keV Scale
- `κ_w_low = 1.0` makes `ρ × κ_w_low ≈ image_low`, so ρ ≈ image_low in water regions.
- This is a design choice, not a physical necessity.
- Alternatives (discussed under M2 / M1-B): anchor ρ to 60 keV scale or to effective-Total scale — each has trade-offs.

### 9.4 Limitations of "vs-Total" Evaluation
- Total is polychromatic with no single effective energy.
- Different methods output at different physical scales (20 keV, 60 keV, image_low scale, etc.).
- Raw PSNR vs. Total across scales is unfair.
- Independent minmax introduces saturation bias.
- **Final choice**: keep independent minmax + caveat about saturation bias; use SSIM as the primary metric.
- See `experiments/exp_log.md` M9 protocol discussion in the original project.

---

## 10. Key References

- Walnut PCCT dataset (Zenodo): https://zenodo.org/records/15738314
- Original reconstruction codebase: https://github.com/zezisme/WalnutPCCTReconCodes
- Paper: Zhou, E., Li, W., Xu, W. et al. *A cone-beam photon-counting CT dataset for spectral image reconstruction and deep learning*. Sci Data 12, 1955 (2025). https://doi.org/10.1038/s41597-025-06246-4
- NIST XCOM database (water μ/ρ): https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html
- NIST XCOM (general): https://physics.nist.gov/PhysRefData/XCOM/Text/XCOM.html
