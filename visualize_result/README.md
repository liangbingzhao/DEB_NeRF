# Visualize Results: 3D Volume Renders of DEB-NeRF / Path D

Curated 3D render outputs from `experiments/render_pathD_3d.py`, one selection per walnut. All renders are produced from a single trained Path D Softplus α+bone model, post-hoc on the saved volume — see `experiments/render_pathD_3d.py` for the rendering code.

Each subdir contains 5 files:

| File | What you are looking at |
|---|---|
| `headline_panel.png` | Static 8-tile summary (4 mid-z slices + 4 mid-rotation 3D thumbnails) covering α_w (water basis), α_2 (second / mineral basis), ρ_total = α_w + α_2, and false-color (α_w blue + α_2 red). |
| `rotation_basis2_falsecolor.gif` | 24-frame 360° rotation, false-color: water basis in blue, second basis in red — the main headline visual. |
| `rotation_alpha_w_gray.gif` | 360° rotation of the water-basis volume in grayscale ("bone" colormap). |
| `rotation_alpha_2_gray.gif` | 360° rotation of the second-basis volume — note the extreme sparsity (~3 % non-zero voxels) and how it concentrates near boundaries rather than forming coherent material regions. |
| `rotation_total_gray.gif` | 360° rotation of ρ_total = α_w + α_2 in grayscale; the unified energy-independent field. |

## Selection Logic

The headline finding of this project is that the Path D second basis is **not** a true material basis — it is a sparse-view boundary residual. The renders below are picked to make that story visible across three different walnut samples.

| Subdir | Config | Why this one |
|---|---|---|
| `walnut_1/25v_winner/` | (20, 60) Softplus α+bone, 25 per-energy views | **Headline result.** Matched-budget winner: 25 + 25 = 50 total projections, against Single 50 Total baseline (PSNR 27.92 / SSIM 0.978, +1.05 dB over Single 50 Total). |
| `walnut_1/50v_winner/` | (20, 60) Softplus α+bone, 50 per-energy views | **Near-Total baseline result** (Phase A): 50 + 50 = 100 total projections. SSIM 0.9816 vs the Single 50 Total baseline 0.9844 — gap 0.0028. |
| `walnut_2/30_80_50v/` | (30, 80) Softplus α+bone, 50 per-energy views | **Cross-walnut pair shift.** On Walnut_2 the (20, 60) default no longer wins; the energy pair shifts to (30, 80) (K = shell/water ≈ 2.66, vs 4.41 for (20, 60)). |
| `walnut_2/20_60_water_init_50v/` | (20, 60) Softplus α + water_init, 50 per-energy views | **Cross-walnut init shift.** On Walnut_2 the optimal κ_2 init also shifts away from cortical bone: water init (κ_2 = κ_w) beats bone init (ΔSSIM +0.028). This confirms α_2 is not picking up bone-like material — it is filling a sparse residual. |
| `walnut_3/20_70_25v/` | (20, 70) Softplus α+bone, 25 per-energy views | **Walnut_3 cross-walnut Borda winner.** On Walnut_3 at 25 per-energy views, (20, 70) is the top method by combined PSNR + SSIM ranking (PSNR 23.34 / SSIM 0.9546). Another pair shift away from the W1-winning (20, 60). |

## How to View

- `.png` opens in any image viewer or directly on GitHub.
- `.gif` files animate when opened in a browser, an image viewer that supports GIFs, or directly on the GitHub web UI.

## How to Regenerate

To rebuild any of these (assuming the corresponding Path D ckpt and α_w / α_2 / ρ_total volumes are saved under `experiments/walnut_<N>/eval_M9_total/render3d/<config>/<view>/`):

```bash
python experiments/render_pathD_3d.py \
    --walnut walnut_1 \
    --pair 20_60 \
    --view 25v
# Producing the 8 GIFs + 1 headline panel takes ~57 s on a single A100
# (0.1 s per frame; 24 frames per rotation).
```

See `experiments/render_pathD_3d.py` for the full CLI surface, including the `--kernel_peel` knob used in the Walnut_1 kernel-mask K sweep.
