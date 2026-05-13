# Idea: Multi-Energy Sparse CT Reconstruction with Cross-Energy Guidance

---

## 1. Motivation

### 1.1 Limitations of Single-Energy Sparse CT

- Sparse-view CT is an ill-posed inverse problem. With few views, the solution space is huge and reconstructions are prone to artefacts.
- **No cross-check**: there is no intrinsic physical consistency that tells us whether a single-energy reconstruction is real or contains artefacts; the only signal is comparison to ground truth.
- **Scalar output only**: a single-energy SAX-NeRF outputs a single μ(x,y,z). It cannot distinguish materials — the same μ value may correspond to different substances.

### 1.2 Extra Signals from Multi-Energy Data

| Signal | Description |
|---|---|
| **Physical hard constraint** | For any material, μ(E) decreases monotonically with energy E (photoelectric ∝ 1/E³, Compton weakly dependent) → μ_low(x) ≥ μ_high(x) must hold. |
| **Structural consistency** | Low- and high-energy scans see the **same object**; geometric boundaries are identical, only attenuation magnitude differs → gradient directions should align. |
| **Material fingerprint** | The ratio μ_low/μ_high is a **material property**: high-Z materials (e.g. shell containing calcium) have a larger ratio; low-Z materials (e.g. organic kernel) have a smaller ratio. |
| **Cross-check** | If an artefact appears in one energy but not the other, it can be identified and penalised. |

None of these signals exist in the single-energy framework. They are the unique "extra supervision" and "extra output" that multi-energy provides.

---

## 2. Where We Are

### Completed

- Walnut PCCT v3 data preprocessing (keeping negative HU, container removed, min-max normalisation).
- 9 energies × 3 resolutions (128 / 256 / 512), 27 single-energy SAX-NeRF baselines.
- NeRF-style volume rendering: μ → c(R,G,B,α) → alpha compositing — the right half of the Method figure.

### Current 256³ Baselines

| Energy | 256³ 3D PSNR | Role |
|---|---|---|
| 20 keV | **35.72** | Low-energy baseline (high contrast, photoelectric-dominated) |
| 30 keV | 36.06 | Best single-energy at 256³ (reference upper bound) |
| 60 keV | **30.63** | High-energy baseline (low contrast, Compton-dominated) |
| Total  | 33.71 | All-energy baseline |

> Note: the earlier v7b numbers (20 keV PSNR ~42 dB) came from **v2 data** and were inflated by air matching in hollow regions; that batch is deprecated. All later experiments are based on v3 data.

### Not Yet Done (Goals of This Idea)

- Dual-energy joint training (shared encoder + per-energy decoder).
- Cross-energy guidance loss (physical inequality + structural consistency).
- "Non-PSNR" value of multi-energy: material segmentation, spectral rendering, thin-shell continuity.

---

## 3. Planned Framework

### 3.1 Overall Architecture

```
            Low-Energy CT proj          High-Energy CT proj
                    │                            │
                    └──────────┬─────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │   Shared Encoder     │  ← geometry features (energy-independent)
                    │   (hash + transformer)│     trainable
                    └──────────┬───────────┘
                               │ shared spatial feature f(x,y,z)
                    ┌──────────┴───────────┐
                    ▼                      ▼
              ┌──────────┐           ┌──────────┐
              │ Low Dec  │           │ High Dec │  ← independent μ_E(x) heads
              └────┬─────┘           └────┬─────┘
                   │                      │
                   ▼                      ▼
           X-ray Volume Render    X-ray Volume Render
                   │                      │
                   ▼                      ▼
            L_recon_low           L_recon_high
                   │                      │
                   └──────────┬───────────┘
                              ▼
               + L_cross (μ inequality, structural alignment)
                              │
                              ▼
                  Total Loss → backprop to encoder + both decoders
```

**Core assumption**: object geometry ("where") is energy-independent; energy only sets attenuation magnitude. Therefore spatial features are shared and only the density head splits.

### 3.2 Network Modules (Based on SAX-NeRF Lineformer)

| Module | Function | Shared? | Notes |
|---|---|---|---|
| Hash encoder | (x,y,z) → high-dim feature | Yes | Geometric position encoding |
| Transformer/MLP backbone | feature → latent | Yes | Spatial structure extraction |
| Density head (Low) | latent → μ_low | No | Low-energy attenuation |
| Density head (High) | latent → μ_high | No | High-energy attenuation |

**First-stage simplification**: only the last 1–2 MLP layers split into two heads; all earlier layers are shared. Deeper splits are considered only if this is not enough.

### 3.3 Loss

```
L_total = λ_recon_l · L_recon_low
        + λ_recon_h · L_recon_high
        + λ_ineq    · L_inequality        # physical hard constraint
        + λ_struct  · L_structural        # structural consistency
```

- `L_recon_*`: projection MSE (same as SAX-NeRF baseline).
- `L_inequality` (enforces μ_low ≥ μ_high):
  ```
  L_inequality = mean( max(0, μ_high(x) - μ_low(x) + ε) )
  ```
  ε is a margin (e.g. 1e-3) that pushes the network toward strict inequality rather than equality.
- `L_structural` (gradient direction alignment):
  ```
  L_structural = 1 - mean( cos( ∇μ_low(x), ∇μ_high(x) ) )
  ```
  Encourages aligned edge orientation across energies; optionally restricted to voxels with |∇μ_low| > τ to avoid being dominated by uniform regions.

### 3.4 Experiment Setup (Locked)

| Item | Value |
|---|---|
| Resolution | **256³** (only one — no 128/512) |
| Energy pair | **20 keV + 60 keV** |
| Projections | 50 (matches single-energy baseline) |
| Backbone | SAX-NeRF Lineformer |
| Single-energy reference | 20 keV: 35.72 dB; 60 keV: 30.63 dB |

Reasons for 20 + 60 keV: (a) 20 keV is a strong low-energy baseline, 60 keV is a weaker high-energy one — a large energy gap means strong constraints; (b) 30 keV is the best at 256³ but too close to 20 keV; (c) physically, 20 → 60 keV crosses the photoelectric/Compton transition, so the material fingerprint is most separable.

---

## 4. Expected Value

PSNR improvement is a by-product. The real value of multi-energy lies in the three story lines below.

### 4.1 Main Lines

#### Story A — "Free" Material Segmentation ★★★★★

**Physically impossible with single energy.**

- μ_low / μ_high is a material fingerprint:
  - Walnut shell (calcium, higher Z_eff): large ratio (theory ~3–5)
  - Walnut kernel (fat / protein / starch, low Z_eff): small ratio (~1.5–2)
  - Air: μ → 0
- After training, simply compute ratio map = μ_20 / μ_60. No segmentation network, no segmentation GT needed.
- **The network produces material classification without ever seeing a mask label.**

**Why this is the headline**: it lifts SAX-NeRF from "reconstructing a scalar field" to "reconstructing a material distribution" — a qualitative capability jump.

#### Story B — Thin-Shell (1-voxel) Continuity ★★★★

**Targets a known concrete pain point.**

- At 256³, 79 % of the shell is 1 voxel thick and shatters into hundreds of disconnected fragments.
- Dual-energy + L_structural:
  - SNR at partial-volume boundaries effectively boosted by √2 (two independent measurements constrain the same boundary).
  - μ-ratio constraint prevents the network from interpreting an intermediate value as "half air, half shell".
- A single-energy baseline can never do this — it only has one supervision signal.

**Key evaluation note**: PSNR alone, being an average, hides this; we need **structural metrics** (connected-component count, largest-CC fraction, local PSNR in the thin-shell band).

#### Story F — Spectral (False-Color) Volume Rendering ★★★★★

**High demo value, low engineering cost.**

- Encode μ_low, μ_high into RGB channels:
  - R = μ_20keV
  - B = μ_60keV
  - G = (R + B) / 2 or alpha-weighted blend
- Rendered effect:
  - Shell (high ratio) → red
  - Kernel (medium ratio) → purple
  - High-density foreign objects → bright red
- Paired with Story A: "this colour *is* the material type".
- Reuses the existing `generate_volume_render.py`, extended to dual μ.

### 4.2 To Explore (Recorded; Pick After Main Lines)

#### C — Fine Inner Structure of the Kernel
- The kernel has internal fat compartments, protein bands, and small air cavities.
- Low energy (20 keV) is sensitive to density differences; high energy (60 keV) is cleaner and does not amplify artefacts.
- Dual energy may recover details that single-energy 256³ blurs out.
- **Risk**: depends on how much detail the 256³ GT itself retains — 256³ (0.5 mm) is already a 10× downsample from the original 0.05 mm.

#### D — Artefact vs. Real Structure Discrimination
- Real structure: consistent across energies, scaled by physical ratio.
- Artefacts (streaks / rings / FOV boundary): morphology may differ between energies.
- A "disagreement map" can serve as a self-cleaning mask.
- **Risk**: current data is virtual mono (synthetic projection), with few artefacts, so the story is weaker than with real PCCT.

#### E — Data Efficiency (Few-View Reconstruction)
- Hypothesis: dual-energy 25+25 projections ≥ single-energy 50 (same total).
- Cross-energy information provides extra constraint.
- **Value**: dose reduction is a core goal of medical/industrial CT.
- **Cost**: needs new 25-view conversion + a parallel comparison study.

---

## 5. Evaluation Plan

PSNR/SSIM alone cannot support the three stories above. Structural and material metrics are **required**.

### 5.1 Numerical Metrics (per energy, aligned with single-energy baselines)
- 3D PSNR / 3D SSIM
- Projection PSNR / Projection SSIM
- Training curves (loss, 3D PSNR vs. epoch)

### 5.2 Structural Metrics (new, for Story B)

| Metric | Computation |
|---|---|
| Shell connected-component count N_cc | Threshold μ to obtain shell mask; count 3D connected components |
| Largest CC fraction | max_cc_size / total_shell_voxels |
| Thin-shell-band local PSNR | PSNR computed only on the 1-voxel-thick GT region |
| Edge IoU | Predicted shell mask vs. GT shell mask intersection-over-union |

### 5.3 Material Metrics (new, for Story A)

| Metric | Computation |
|---|---|
| Ratio map visualisation | μ_20 / μ_60 imaged directly; check for natural clusters |
| Clustering purity (ARI / Dice) | K-means(k=3) on ratio map, compared to GT shell/kernel/air masks |
| Material histogram separability | Overlap area of the three-class ratio histograms |

### 5.4 Visual Comparisons (for Story F)
- False-color RGB volume rendering (μ_low → R, μ_high → B).
- Per-slice side-by-side: single-energy baseline vs. dual-low / dual-high / ratio map.
- Rotation GIFs (paired with single-energy versions).

### 5.5 Ablations (Verify Each Cross-Loss Contribution)

| Configuration | L_recon | L_inequality | L_structural |
|---|---|---|---|
| Dual-base | ✓ | ✗ | ✗ |
| +Ineq | ✓ | ✓ | ✗ |
| +Struct | ✓ | ✗ | ✓ |
| Full | ✓ | ✓ | ✓ |

Each configuration runs the full 5.1 – 5.4 evaluation.

---

## 6. Falsifiable Hypotheses

| H | Story | Hypothesis | Verification |
|---|---|---|---|
| **H1** | A | μ_20 / μ_60 ratio automatically separates shell / kernel / air | Clustering ARI > 0.8 |
| **H2** | B | Dual-energy + L_structural improves shell connectivity | N_cc reduced ≥ 30 %, largest-CC fraction up |
| **H3** | A, B | High-energy (60 keV) 3D PSNR noticeably improves | 60 keV dual PSNR > 60 keV single baseline by ≥ 1 dB |
| **H4** | — | L_inequality on its own has a regularising effect | Ablation (Dual-base vs. +Ineq) PSNR ≥ +0.3 dB or structural metrics improve |

---

## 7. Relation to Existing Work

- **SAX-NeRF** (baseline, implemented): single-energy + transformer + hash encoding.
- **This idea extends SAX-NeRF**: adds multi-head output and physical cross-energy losses.
- **Classical dual-energy CT**: usually relies on basis-material decomposition with a pre-calibrated spectrum. Our framework does not assume known spectra; it learns μ_E via a neural field.
- **NeRF volume rendering** (implemented): a post-processing step, not a training objective; independent of this framework.

---

## 8. Risks & Open Issues

1. **Shared-encoder capacity**: if the encoder is too small, subtle inter-energy differences may be flattened. Monitor whether the two heads actually produce a well-spread ratio distribution.
2. **L_inequality margin ε**: too small → no regularising effect; too large → μ_high is over-suppressed. Ablate.
3. **"Same object" assumption**: v3 data comes from VirtualMonoImg (dual-energy → material decomposition → virtual monoenergetic post-processing), not direct measurement. Geometric consistency between virtual monoenergetic images should be very tight (same underlying reconstruction), but this is worth noting.
3b. **Non-standard HU calibration of VirtualMonoImg** (discovered in M1): the two energies' air HU values disagree (20 keV = -2092, 60 keV = -1793, gap 299). After shared normalisation (per-energy minus air + shared scale), GT still has ~6 % pointwise soft violations of μ_low ≥ μ_high, concentrated in transition regions. 94 % strict satisfaction is enough — L_inequality is a soft loss and the network is pushed to satisfy it during training. If the inequality loss refuses to converge, spectrum recalibration may be needed.
4. **Multi-energy extension**: if dual-energy works, can we extend to 3 – 8 energies with more heads on the same encoder?
5. **Sensitivity of structural metrics**: shell mask threshold is not unique; sensitivity to the threshold needs checking.

---

## 9. Roadmap Summary

See `plan_milestone.md` in the original project for details. Brief sequence:

1. Data → dual-energy paired dataset (256³, 20 + 60 keV).
2. Network → shared encoder + dual head.
3. Training → dual baseline (no cross loss).
4. Loss → + inequality.
5. Loss → + structural.
6. Evaluation → numerical + structural + material + ablation.
7. Story A landing → material segmentation visualisation.
8. Story F landing → false-color spectral rendering.
9. (Extensions, as needed) → C / D / E.
