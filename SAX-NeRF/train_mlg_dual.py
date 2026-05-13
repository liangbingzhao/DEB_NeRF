"""Dual-energy training entry point.

Usage:
    python train_mlg_dual.py --gpu_id 0 --config config/Lineformer/res_256/v3_dual/walnut_20kev_60kev_50.yaml

Trains a Lineformer_dual on a paired dual-energy pickle (image_low/high + projections_low/high).
Loss: weighted sum of per-energy reconstruction MSE.
Eval: 8 metrics (proj/3D × PSNR/SSIM × low/high) + projection-domain inequality sanity (V6).

Best checkpoint criterion: PSNR_3d_avg = (PSNR_3d_low + PSNR_3d_high) / 2.

PSNR_3d uses adaptive PIXEL_MAX = image_gt.max() to ensure scale-invariant comparison
with single-energy baselines (which use image scale [0, 1.0] vs dual's [0, 0.499]).
See plan §V5 for the math.
"""

import os
import os.path as osp
import torch
import imageio.v2 as iio
from tqdm import tqdm
import numpy as np
import random
import argparse


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/Lineformer/res_256/v3_dual/walnut_20kev_60kev_50.yaml",
                        help="configs file path")
    parser.add_argument("--gpu_id", default="0", help="gpu to use")
    parser.add_argument("--seed", type=int, default=-1, help="random seed (-1 = no seed)")
    return parser


parser = config_parser()
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

if args.seed >= 0:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"[SEED] Random seed set to {args.seed}")

from src.config.configloading import load_config
from src.render import render_dual, run_network
from src.trainer_mlg_dual import Trainer_dual
from src.utils import get_psnr, get_ssim, get_psnr_3d, get_ssim_3d, cast_to_image
from src.loss.loss import (
    calc_structural_loss,
    calc_structural_loss_magweight,
    calc_structural_loss_diff_tv,
)


cfg = load_config(args.config)
device = torch.device("cuda")


class BasicTrainer_dual(Trainer_dual):
    def __init__(self):
        super().__init__(cfg, device)
        print(f"[Start] exp: {cfg['exp']['expname']} (dual-energy: {cfg.get('exp', {}).get('expname', '')})")

    # ──────────────────────────────────────────────
    # compute_loss: per-energy reconstruction MSE + optional L_inequality
    # ──────────────────────────────────────────────
    def compute_loss(self, data, global_step, idx_epoch):
        rays = data["rays"].reshape(-1, 8)
        projs_low = data["projs_low"].reshape(-1)
        projs_high = data["projs_high"].reshape(-1)

        ret = render_dual(rays, self.net, self.net_fine, **self.conf["render"])
        pred_low = ret["acc_low"]
        pred_high = ret["acc_high"]

        w_low = self.conf.get("loss", {}).get("lambda_recon_low", 1.0)
        w_high = self.conf.get("loss", {}).get("lambda_recon_high", 1.0)
        w_ineq = self.conf.get("loss", {}).get("lambda_inequality", 0.0)
        eps_ineq = self.conf.get("loss", {}).get("epsilon_ineq", 1e-3)

        loss_recon_low = torch.mean((projs_low - pred_low) ** 2)
        loss_recon_high = torch.mean((projs_high - pred_high) ** 2)
        loss_total = w_low * loss_recon_low + w_high * loss_recon_high

        # L_inequality: physics constraint μ_low ≥ μ_high - ε on per-sample raw μ.
        # Skipped when λ=0 to keep behavior identical to M3 baseline.
        if w_ineq > 0.0:
            mu_low = ret["raw"][..., 0]   # [N_rays, N_samples]
            mu_high = ret["raw"][..., 1]
            loss_ineq = torch.mean(torch.relu(mu_high - mu_low + eps_ineq))
            loss_total = loss_total + w_ineq * loss_ineq
            self.writer.add_scalar("train/loss_ineq", loss_ineq.item(), global_step)

        # L_structural: cross-energy structural consistency on auxiliary 3D sub-cube.
        # Variants (struct_loss_type): "cossim" (v1), "magweight" (v2/A1), "difftv" (v3/B3).
        w_struct = self.conf.get("loss", {}).get("lambda_structural", 0.0)
        if w_struct > 0.0:
            g_size = self.conf.get("loss", {}).get("struct_grid_size", 16)
            extent = self.conf.get("loss", {}).get("struct_grid_extent", 0.0375)
            struct_type = self.conf.get("loss", {}).get("struct_loss_type", "cossim")
            net_dev = next(self.net.parameters()).device
            bound = self.net.bound - 1e-6
            half = 0.5 * extent
            center = (torch.rand(3, device=net_dev) * 2 - 1) * (bound - half)
            lin = torch.linspace(-half, half, g_size, device=net_dev)
            gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
            pts_grid = torch.stack([gx, gy, gz], dim=-1) + center
            raw_grid = self.net(pts_grid.reshape(-1, 3)).reshape(g_size, g_size, g_size, 2)
            mu_low_g = raw_grid[..., 0]
            mu_high_g = raw_grid[..., 1]
            if struct_type == "cossim":
                mask_pct = self.conf.get("loss", {}).get("struct_mask_percentile", 0.5)
                loss_struct = calc_structural_loss(mu_low_g, mu_high_g, mask_pct)
            elif struct_type == "magweight":
                loss_struct = calc_structural_loss_magweight(mu_low_g, mu_high_g)
            elif struct_type == "difftv":
                alpha = self.conf.get("loss", {}).get("struct_diff_tv_alpha", 3.80)
                loss_struct = calc_structural_loss_diff_tv(mu_low_g, mu_high_g, alpha)
            else:
                raise ValueError(f"unknown struct_loss_type: {struct_type}")
            loss_total = loss_total + w_struct * loss_struct
            self.writer.add_scalar("train/loss_struct", loss_struct.item(), global_step)

        # Tensorboard scalars
        self.writer.add_scalar("train/loss", loss_total.item(), global_step)
        self.writer.add_scalar("train/loss_recon_low", loss_recon_low.item(), global_step)
        self.writer.add_scalar("train/loss_recon_high", loss_recon_high.item(), global_step)

        return loss_total

    # ──────────────────────────────────────────────
    # eval_step: per-energy 3D + projection metrics
    # ──────────────────────────────────────────────
    def eval_step(self, global_step, idx_epoch):
        # ---- Project both energies along val rays ----
        projs_low_gt = self.eval_dset.projs_low      # [N_val, H, W]
        projs_high_gt = self.eval_dset.projs_high
        rays_flat = self.eval_dset.rays.reshape(-1, 8)
        N, H, W = projs_low_gt.shape

        pred_low_list, pred_high_list = [], []
        for i in tqdm(range(0, rays_flat.shape[0], self.n_rays), desc="eval projections"):
            r = render_dual(rays_flat[i:i + self.n_rays], self.net, self.net_fine, **self.conf["render"])
            pred_low_list.append(r["acc_low"])
            pred_high_list.append(r["acc_high"])
        projs_pred_low = torch.cat(pred_low_list, 0).reshape(N, H, W)
        projs_pred_high = torch.cat(pred_high_list, 0).reshape(N, H, W)

        # ---- Reconstruct full 3D volumes for both energies ----
        net_for_eval = self.net_fine if self.net_fine is not None else self.net
        image_pred_dual = run_network(self.eval_dset.voxels, net_for_eval, self.netchunk)  # [R, R, R, 2]
        image_pred_low = image_pred_dual[..., 0]
        image_pred_high = image_pred_dual[..., 1]
        image_gt_low = self.eval_dset.image_low
        image_gt_high = self.eval_dset.image_high

        # ---- Metrics (PIXEL_MAX adaptive for scale-invariant comparison vs single baselines) ----
        max_low = float(image_gt_low.max().item())
        max_high = float(image_gt_high.max().item())

        psnr_3d_low = get_psnr_3d(image_pred_low, image_gt_low, PIXEL_MAX=max_low)
        psnr_3d_high = get_psnr_3d(image_pred_high, image_gt_high, PIXEL_MAX=max_high)
        ssim_3d_low = get_ssim_3d(image_pred_low, image_gt_low, PIXEL_MAX=max_low)
        ssim_3d_high = get_ssim_3d(image_pred_high, image_gt_high, PIXEL_MAX=max_high)
        proj_psnr_low = get_psnr(projs_pred_low, projs_low_gt)
        proj_psnr_high = get_psnr(projs_pred_high, projs_high_gt)
        proj_ssim_low = get_ssim(projs_pred_low, projs_low_gt)
        proj_ssim_high = get_ssim(projs_pred_high, projs_high_gt)

        # V6: projection-domain inequality sanity (acc_low ≥ acc_high should hold)
        proj_violation_pixels = int((projs_pred_low < projs_pred_high).sum().item())
        proj_total_pixels = int(projs_pred_low.numel())
        proj_violation_ratio = proj_violation_pixels / proj_total_pixels

        # 3D-domain inequality sanity (μ_low ≥ μ_high)
        vol_violation = int((image_pred_low < image_pred_high).sum().item())
        vol_total = int(image_pred_low.numel())
        vol_violation_ratio = vol_violation / vol_total

        # Combined (best ckpt) criterion
        psnr_3d_avg_t = (psnr_3d_low + psnr_3d_high) / 2.0

        loss = {
            "psnr_3d_low": psnr_3d_low,
            "psnr_3d_high": psnr_3d_high,
            "ssim_3d_low": ssim_3d_low,
            "ssim_3d_high": ssim_3d_high,
            "proj_psnr_low": proj_psnr_low,
            "proj_psnr_high": proj_psnr_high,
            "proj_ssim_low": proj_ssim_low,
            "proj_ssim_high": proj_ssim_high,
            "psnr_3d_avg": psnr_3d_avg_t,
            "proj_ineq_violation_ratio": torch.tensor(proj_violation_ratio),
            "vol_ineq_violation_ratio": torch.tensor(vol_violation_ratio),
        }

        # Best ckpt: avg of low/high 3D PSNR
        psnr_3d_avg_v = float(psnr_3d_avg_t.item()) if torch.is_tensor(psnr_3d_avg_t) else float(psnr_3d_avg_t)
        if psnr_3d_avg_v > self.best_psnr_3d:
            torch.save(
                {
                    "epoch": idx_epoch,
                    "network": self.net.state_dict(),
                    "network_fine": self.net_fine.state_dict() if self.n_fine > 0 else None,
                    "optimizer": self.optimizer.state_dict(),
                },
                self.ckpt_best_dir,
            )
            self.best_psnr_3d = psnr_3d_avg_v
            self.logger.info(f"best model update, epoch:{idx_epoch}, best PSNR_3d_avg:{self.best_psnr_3d:.4g}")

        # ---- TensorBoard scalars ----
        for k, v in loss.items():
            v_scalar = v.item() if torch.is_tensor(v) else float(v)
            self.writer.add_scalar(f"eval/{k}", v_scalar, global_step)

        # ---- Save eval outputs ----
        eval_save_dir = osp.join(self.evaldir, f"epoch_{idx_epoch:05d}")
        os.makedirs(eval_save_dir, exist_ok=True)
        np.save(osp.join(eval_save_dir, "image_pred_low.npy"), image_pred_low.cpu().detach().numpy())
        np.save(osp.join(eval_save_dir, "image_pred_high.npy"), image_pred_high.cpu().detach().numpy())
        np.save(osp.join(eval_save_dir, "image_gt_low.npy"), image_gt_low.cpu().detach().numpy())
        np.save(osp.join(eval_save_dir, "image_gt_high.npy"), image_gt_high.cpu().detach().numpy())

        # Slice show: 5 slices per energy, row1=gt, row2=pred
        show_slice = 5
        for tag, gt_img, pred_img in (
            ("low", image_gt_low, image_pred_low),
            ("high", image_gt_high, image_pred_high),
        ):
            show_step = max(1, gt_img.shape[-1] // show_slice)
            show_gt = gt_img[..., ::show_step][..., :show_slice]
            show_pred = pred_img[..., ::show_step][..., :show_slice]
            slabs = []
            for s in range(show_slice):
                slabs.append(torch.concat([show_gt[..., s], show_pred[..., s]], dim=0))
            density_strip = torch.concat(slabs, dim=1)
            self.writer.add_image(f"eval/density_{tag}_row1gt_row2pred", cast_to_image(density_strip),
                                  global_step, dataformats="HWC")
            iio.imwrite(
                osp.join(eval_save_dir, f"slice_show_{tag}_row1gt_row2pred.png"),
                (cast_to_image(density_strip) * 255).astype(np.uint8))

        # Projection PNGs (proj_pred_low/high, proj_gt_low/high)
        for tag, projs_pred, projs_gt in (
            ("low", projs_pred_low, projs_low_gt),
            ("high", projs_pred_high, projs_high_gt),
        ):
            pred_dir = osp.join(self.expdir, f"proj_pred_{tag}")
            gt_dir = osp.join(self.expdir, f"proj_gt_{tag}")
            os.makedirs(pred_dir, exist_ok=True)
            os.makedirs(gt_dir, exist_ok=True)
            for i in range(N):
                iio.imwrite(osp.join(pred_dir, f"proj_pred_{i}.png"),
                            (cast_to_image(projs_pred[i]) * 255).astype(np.uint8))
                iio.imwrite(osp.join(gt_dir, f"proj_gt_{i}.png"),
                            (cast_to_image(projs_gt[i]) * 255).astype(np.uint8))

        # stats.txt: write all metrics + violation ratios
        with open(osp.join(eval_save_dir, "stats.txt"), "w") as f:
            for k, v in loss.items():
                v_scalar = v.item() if torch.is_tensor(v) else float(v)
                f.write(f"{k}: {v_scalar:.6f}\n")
            f.write(f"# notes:\n")
            f.write(f"#   PSNR_3d uses adaptive PIXEL_MAX (low={max_low:.4f}, high={max_high:.4f})\n")
            f.write(f"#   for scale-invariant comparison vs single-energy baselines.\n")
            f.write(f"#   proj_ineq_violation_ratio: fraction of val pixels where pred_low < pred_high\n")
            f.write(f"#   vol_ineq_violation_ratio: fraction of voxels where pred_low < pred_high\n")

        return loss


if __name__ == "__main__":
    trainer = BasicTrainer_dual()
    trainer.start()
