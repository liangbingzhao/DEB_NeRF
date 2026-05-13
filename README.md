# DEB-NeRF: Dual-Energy Basis NeRF for Sparse-View PCCT Reconstruction

CS300 final project. **Authors**: Min Zhou, Liangbing Zhao.

This repository contains the source code, configurations, and documentation for our work on **sparse-view photon-counting CT (PCCT) reconstruction using multi-energy neural radiance fields**. It is organized for code inspection and trial runs; the raw dataset and trained model weights are *not* included (see §3 for how to obtain them).

This is a cleaned snapshot — only the **final winning pipeline** is kept. Earlier deprecated single-energy / dual-head explorations have been removed.

---

## 1. Project Summary (One Paragraph)

We extend SAX-NeRF (a transformer-based neural attenuation field) from single-energy to **dual-energy** sparse-view CT on a real photon-counting walnut dataset. Our final winning architecture — **DEB-NeRF / Path D**: a two-basis neural decomposition with a learnable second basis material (Softplus-constrained) and cortical-bone initialization — produces a **single energy-independent 3D scalar field**. On Walnut_1 with **25 per-energy views (50 total projections, matching the Single 50 Total budget)**, it achieves **PSNR 27.92 / SSIM 0.978** against the polychromatic Total reference, **+1.05 dB** over the matched-total-budget Single 50 Total baseline and **+1.35 dB** over the strongest matched-per-energy VMI baseline (Single 25 @ 60 keV). Cross-walnut validation (Walnut_2 / Walnut_3) shows that the optimal basis (pair, init) is sample-dependent — an honest finding that supports re-framing the second basis as a **sparse-view boundary residual** rather than a true material basis.

For full design rationale, hypotheses, and evaluation protocol, see **`docs/idea.md`**.
For dataset physics and the equivalence between Path D and classical spectral CT, see **`docs/pcct_dataset_understand.md`**.

---

## 2. Code Layout

```
cs300_upload/
├── README.md                  ← you are here
├── allocnode.sh               ← Ibex helper: allocate an interactive A100 node
├── run_into_node.sh           ← Ibex helper: ssh into the allocated node
│
├── docs/                      ← documentation (English)
│   ├── idea.md                ← motivation, framework, hypotheses, evaluation plan
│   └── pcct_dataset_understand.md
│                              ← dataset physics, VMI synthesis, Path D equivalence
│
├── SAX-NeRF/                  ← model code (extends upstream SAX-NeRF)
│   ├── src/
│   │   ├── network/
│   │   │   ├── Lineformer.py             ← original single-energy backbone (baseline)
│   │   │   ├── Lineformer_singlefield.py ← Path B: single basis × NIST water curve
│   │   │   └── Lineformer_basis2.py      ← Path D: two-basis decomposition (winner)
│   │   ├── dataset/                      ← TIGRE-based dataloaders (single + dual)
│   │   ├── render/render.py              ← X-ray volume render + render_dual + render_singlefield
│   │   ├── loss/loss.py                  ← physics + structural losses
│   │   ├── trainer_mlg.py                ← single-energy trainer
│   │   └── trainer_mlg_dual.py           ← dual-energy base trainer (Path B / Path D inherit)
│   ├── train_mlg.py                      ← single-energy training entry (baseline)
│   ├── train_mlg_singlefield.py          ← Path B training entry
│   ├── train_mlg_basis2.py               ← Path D training entry (winner)
│   ├── test.py                           ← evaluation harness (PSNR/SSIM 3D + projection)
│   ├── config/Lineformer/res_256/        ← training YAML configs:
│   │     v3_phys/                        ← single-energy phys-calibrated (per walnut, 50v)
│   │     v3_phys_25view/                 ← 25-view single-energy
│   │     v3_phys_total/                  ← Total polychromatic baseline references
│   │     v3_dual_phys_singlefield{,_25view}/ ← Path B configs
│   │     v3_dual_phys_basis2{,_25view}/  ← Path D configs (winner)
│   ├── TIGRE-2.3/                        ← TIGRE source (build with provided MEX/CUDA)
│   ├── requirements.txt
│   └── README.md                         ← upstream SAX-NeRF README
│
├── data_preprocess/                      ← raw → pickle conversion
│   ├── convert_walnut_phys.py            ← single-energy + NIST physical calibration
│   ├── convert_walnut_dual_phys.py       ← dual-energy paired pickle (shared scale)
│   ├── convert_walnut_total.py           ← Total polychromatic reference volume
│   ├── convert_walnut_total_with_proj.py ← Total + projection synthesis (for trainable Total ckpt)
│   └── zezisme_recon/                    ← original MATLAB recon code (external reference;
│                                            source: https://github.com/zezisme/WalnutPCCTReconCodes)
│                                            documents the closed-form spectral CT algorithm
│                                            that Path D mirrors mathematically
│
├── visualize_result/                    ← curated 3D render outputs (one per walnut)
│   ├── walnut_1/{25v_winner, 50v_winner}/         ← Path D Softplus α+bone defaults
│   ├── walnut_2/{30_80_50v, 20_60_water_init_50v}/← cross-walnut pair / init shifts
│   ├── walnut_3/20_70_25v/                        ← Walnut_3 Borda winner
│   └── README.md                                   ← per-subdir selection logic
│       (each subdir: headline_panel.png + 4 rotation GIFs: basis2 false-color,
│        α_w, α_2, ρ_total)
│
└── experiments/                          ← evaluation, generators, analysis
    ├── README.md                         ← experiments quick reference
    ├── exp_log.md                        ← detailed experiment log (Chinese, technical record)
    │
    ├── compute_pair_calibration.py       ← NIST κ_w_high + κ_2 init per (low, high) pair
    │
    ├── eval_vs_total.py                  ← main evaluator (vs-Total polychromatic ranking)
    ├── report_eval_split.py              ← Borda-sorted (PSNR + SSIM) markdown report
    ├── eval_split_report_both.md         ← cross-walnut full ranking (report output)
    │
    ├── render_pathD_3d.py                ← Path D (α_w, α_2, ρ_total) false-color volume render
    │
    ├── gen_phaseC_data_jobs.py           ← per-walnut 21-pickle dataset generation
    ├── gen_phaseD_walnut2_3_jobs.py      ← Path D training across 7-pair × 2-view × walnut
    ├── gen_phaseE_w23_variants.py        ← variant ablation (B/C/D/Raw/HAP) on W2/W3
    └── gen_phaseF_singles.py             ← single-energy baseline grid on W2/W3
```

---

## 3. Data Setup (Not Included Here — ~180 GB)

The raw walnut DICOM data and pre-trained model weights are **not** in this archive. To obtain them:

### 3.1 Download the Walnut PCCT Dataset

From Zenodo: <https://zenodo.org/records/15738314>

Place the extracted folders under `data_preprocess/`:
```
data_preprocess/
├── Walnut_1/                  ← raw projection data + reconstructions
├── Walnut_2/
├── Walnut_3/
├── Reconstructions/Walnut_{1,2,3}/   ← FDK + VMI DICOM volumes
└── CalibrationTable/                 ← NIST + empirical calibration tables (also shipped via the dataset)
```

The dataset paper:
> Zhou, E., Li, W., Xu, W. et al. *A cone-beam photon-counting CT dataset for spectral image reconstruction and deep learning*. Sci Data 12, 1955 (2025). DOI: 10.1038/s41597-025-06246-4

### 3.2 Generate the Per-Walnut Pickle Set

Each walnut needs a 21-pickle set (dual + single + Total at 25 / 50 views, plus 7 dual energy pairs). Submit via:

```bash
python experiments/gen_phaseC_data_jobs.py --walnut Walnut_1
python experiments/gen_phaseC_data_jobs.py --walnut Walnut_2
python experiments/gen_phaseC_data_jobs.py --walnut Walnut_3
# Each invocation submits one SLURM sbatch (~60–90 min on an A100) that produces
# all 21 pickles under SAX-NeRF/data/res_256/v3_*phys*/walnut_<N>/.
```

---

## 4. Environment

```bash
conda env create -f SAX-NeRF/environment.yml   # or use requirements.txt
conda activate sax_nerf
# Required: CUDA 11.7, PyTorch 1.13.1, TIGRE 2.3, A100 GPU.
# The hash encoder CUDA extension is compiled for sm_80 — it must run on A100.
```

Build the TIGRE GPU MEX/CUDA backend once:
```bash
cd SAX-NeRF/TIGRE-2.3/Python && pip install -e .
```

---

## 5. How to Reproduce the Winning Result (Walnut_1, Path D Softplus, 25 per-energy views = 50 total projections)

```bash
# 1. Generate the dataset (assumes the raw data is in data_preprocess/Walnut_1/, Reconstructions/, CalibrationTable/)
python experiments/gen_phaseC_data_jobs.py --walnut Walnut_1

# 2. Train Path D winner (25 per-energy views, ~2.5 h on a single A100)
cd SAX-NeRF
python train_mlg_basis2.py \
    --config config/Lineformer/res_256/v3_dual_phys_basis2_25view/walnut_20kev_60kev_25_basis2_softplus_lh10.yaml

# 3. Evaluate against the polychromatic Total reference
cd ..
python experiments/eval_vs_total.py --walnut Walnut_1
# Output: walnut_1/eval_M9_total/{results.json, comparison_table.csv}
```

To run the same ablation on Walnut_2 / Walnut_3, swap `--walnut Walnut_1` for `Walnut_2` / `Walnut_3` and use the corresponding `walnut_2/` / `walnut_3/` config subdirs.

---

## 6. Key Result — Walnut_1: DEB-NeRF vs. Single-Energy VMI Baselines and the Total Baseline

Metrics are computed against the polychromatic Total reference volume after min-max normalization to [0, 1]. "Per-energy views" is the per-channel view count (so DEB-NeRF with 25 per-energy views consumes 50 total projections, matching the Single 50 Total budget). The bottom block reports DEB-NeRF's gain over the matched-per-energy baseline, the matched-total-budget baseline, and the strongest matched-per-energy single-energy VMI baseline.

| Method | per-energy views | total views | PSNR | SSIM |
|---|---:|---:|---:|---:|
| **DEB-NeRF (20, 60) + softplus (ours)** | **25** | **50** | **27.92** | **0.978** |
| *Total baseline (same backbone trained on Total projections)* | | | | |
| Single 25 Total | 25 | 25 | 25.62 | 0.979 |
| Single 50 Total | 50 | 50 | 26.87 | 0.984 |
| *Single-energy VMI baselines* | | | | |
| Single 25 @ 20 keV | 25 | 25 | 22.52 | 0.961 |
| Single 25 @ 60 keV | 25 | 25 | 26.57 | 0.973 |
| Single 50 @ 20 keV | 50 | 50 | 24.27 | 0.971 |
| Single 50 @ 60 keV | 50 | 50 | 24.77 | 0.977 |
| *Improvement of DEB-NeRF over reference* | | | | |
| Δ vs. Single 25 Total | — | — | **+2.30** | +0.000 |
| Δ vs. Single 50 Total | — | — | **+1.05** | −0.006 |
| Δ vs. Single 25 @ 60 keV | — | — | **+1.35** | +0.005 |

Cross-walnut: on Walnut_2 / Walnut_3, the optimal energy pair shifts to **(30, 80)** and the optimal κ_2 init shifts to **water** — see the Phase D-W23 / Phase E discussion in `experiments/exp_log.md`.

---

## 7. Citation Notes

- **SAX-NeRF**: original Lineformer architecture and training loop adapted from the official SAX-NeRF release.
- **TIGRE 2.3**: GPU-accelerated tomographic projection / backprojection toolbox.
- **Walnut PCCT dataset**: Zhou et al., *Sci Data* 12, 1955 (2025).
- **MATLAB reference recon**: `zezisme/WalnutPCCTReconCodes` — closed-form spectral decomposition; mirrored mathematically by Path D (see `docs/pcct_dataset_understand.md` §2.2.3).

---

## 8. License

Inherits the licenses of the upstream code (SAX-NeRF, TIGRE) and dataset. Project-specific additions are released under the same terms as the host course's deliverables policy.
