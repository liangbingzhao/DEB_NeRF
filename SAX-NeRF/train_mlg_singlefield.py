"""Single-field training entry (M9 Path B).

One density-like field ρ(x) ∈ [0,1], supervised by both 20 keV and 60 keV projections.
ρ is aligned with the dual pickle's image_low normalization (image_low_max = 1.0):
    pred_image_low(x)  = ρ(x) × kappa_low   (default kappa_low  = 1.0)
    pred_image_high(x) = ρ(x) × kappa_high  (default kappa_high = μ_water_60/μ_water_20 = 0.2543)

Reuses Trainer_dual (in src/trainer_mlg_dual.py) for dataset/optimizer/loop;
overrides compute_loss and eval_step to use render_singlefield. No L_inequality
and no L_structural — physics μ_low ≥ μ_high is automatically satisfied since
kappa_low > kappa_high and ρ ≥ 0.

Usage:
    python train_mlg_singlefield.py --gpu_id 0 \
        --config config/Lineformer/res_256/v3_dual_phys_singlefield/<yaml>
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
    parser.add_argument("--config", required=True, help="configs file path")
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
from src.render import render_singlefield, run_network
from src.trainer_mlg_dual import Trainer_dual
from src.utils import get_psnr, get_ssim, get_psnr_3d, get_ssim_3d, cast_to_image


cfg = load_config(args.config)
device = torch.device("cuda")


class BasicTrainer_singlefield(Trainer_dual):
    def __init__(self):
        super().__init__(cfg, device)
        # ρ ∈ [0,1] aligned with image_low; kappa_high = μ_water_60 / μ_water_20 = 0.2543.
        # Override via yaml "physics" section if needed.
        physics = cfg.get("physics", {})
        self.kappa_low = float(physics.get("kappa_low", 1.0))
        self.kappa_high = float(physics.get("kappa_high", 0.2543))

        # Optional: rescale image_high / projs_high to flip ρ alignment.
        # high_rescale=7.87 (≈ 1/image_high_max) makes ρ align to 60keV (when paired
        # with kappa_low=0.5, kappa_high=1.0). Default 1.0 = no change (20keV alignment).
        self.high_rescale = float(physics.get("high_rescale", 1.0))
        if abs(self.high_rescale - 1.0) > 1e-6:
            self.train_dloader.dataset.projs_high.mul_(self.high_rescale)
            self.train_dloader.dataset.image_high.mul_(self.high_rescale)
            if self.eval_dset is not None:
                self.eval_dset.projs_high.mul_(self.high_rescale)
                self.eval_dset.image_high.mul_(self.high_rescale)
            print(f"[high_rescale] applied factor={self.high_rescale} to projs_high + image_high")

        print(f"[Start] singlefield: expname={cfg['exp']['expname']}, "
              f"kappa_low={self.kappa_low}, kappa_high={self.kappa_high}, "
              f"high_rescale={self.high_rescale}")

    # ──────────────────────────────────────────────
    # compute_loss: per-energy reconstruction MSE on ρ × κ scaling
    # ──────────────────────────────────────────────
    def compute_loss(self, data, global_step, idx_epoch):
        rays = data["rays"].reshape(-1, 8)
        projs_low = data["projs_low"].reshape(-1)
        projs_high = data["projs_high"].reshape(-1)

        ret = render_singlefield(rays, self.net, self.net_fine,
                                 kappa_low=self.kappa_low, kappa_high=self.kappa_high,
                                 **self.conf["render"])
        pred_low = ret["acc_low"]
        pred_high = ret["acc_high"]

        w_low = self.conf.get("loss", {}).get("lambda_recon_low", 1.0)
        w_high = self.conf.get("loss", {}).get("lambda_recon_high", 1.0)

        loss_recon_low = torch.mean((projs_low - pred_low) ** 2)
        loss_recon_high = torch.mean((projs_high - pred_high) ** 2)
        loss_total = w_low * loss_recon_low + w_high * loss_recon_high

        self.writer.add_scalar("train/loss", loss_total.item(), global_step)
        self.writer.add_scalar("train/loss_recon_low", loss_recon_low.item(), global_step)
        self.writer.add_scalar("train/loss_recon_high", loss_recon_high.item(), global_step)
        return loss_total

    # ──────────────────────────────────────────────
    # eval_step: same metrics structure as dual, but image_pred_low/high derived from ρ × κ
    # ──────────────────────────────────────────────
    def eval_step(self, global_step, idx_epoch):
        # ---- Project both energies along val rays ----
        projs_low_gt = self.eval_dset.projs_low
        projs_high_gt = self.eval_dset.projs_high
        rays_flat = self.eval_dset.rays.reshape(-1, 8)
        N, H, W = projs_low_gt.shape

        pred_low_list, pred_high_list = [], []
        for i in tqdm(range(0, rays_flat.shape[0], self.n_rays), desc="eval projections"):
            r = render_singlefield(rays_flat[i:i + self.n_rays], self.net, self.net_fine,
                                    kappa_low=self.kappa_low, kappa_high=self.kappa_high,
                                    **self.conf["render"])
            pred_low_list.append(r["acc_low"])
            pred_high_list.append(r["acc_high"])
        projs_pred_low = torch.cat(pred_low_list, 0).reshape(N, H, W)
        projs_pred_high = torch.cat(pred_high_list, 0).reshape(N, H, W)

        # ---- Reconstruct full 3D ρ field, then scale to per-energy μ ----
        net_for_eval = self.net_fine if self.net_fine is not None else self.net
        rho_pred = run_network(self.eval_dset.voxels, net_for_eval, self.netchunk)  # [R, R, R, 1]
        rho_pred = rho_pred[..., 0]
        image_pred_low = rho_pred * self.kappa_low
        image_pred_high = rho_pred * self.kappa_high
        image_gt_low = self.eval_dset.image_low
        image_gt_high = self.eval_dset.image_high

        # ---- Metrics (PIXEL_MAX adaptive for scale-invariant comparison) ----
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

        # V6: inequality monitor (auto-satisfied since κ_low > κ_high and ρ ≥ 0)
        proj_violation_pixels = int((projs_pred_low < projs_pred_high).sum().item())
        proj_total_pixels = int(projs_pred_low.numel())
        proj_violation_ratio = proj_violation_pixels / proj_total_pixels
        vol_violation = int((image_pred_low < image_pred_high).sum().item())
        vol_total = int(image_pred_low.numel())
        vol_violation_ratio = vol_violation / vol_total

        psnr_3d_avg_t = (psnr_3d_low + psnr_3d_high) / 2.0

        # Stats on the ρ field itself (M9-specific monitor)
        rho_min = float(rho_pred.min().item())
        rho_max = float(rho_pred.max().item())
        rho_mean = float(rho_pred.mean().item())

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
            "rho_min": torch.tensor(rho_min),
            "rho_max": torch.tensor(rho_max),
            "rho_mean": torch.tensor(rho_mean),
        }

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

        for k, v in loss.items():
            v_scalar = v.item() if torch.is_tensor(v) else float(v)
            self.writer.add_scalar(f"eval/{k}", v_scalar, global_step)

        # ---- Save eval outputs ----
        eval_save_dir = osp.join(self.evaldir, f"epoch_{idx_epoch:05d}")
        os.makedirs(eval_save_dir, exist_ok=True)
        np.save(osp.join(eval_save_dir, "rho_pred.npy"), rho_pred.cpu().detach().numpy())
        np.save(osp.join(eval_save_dir, "image_pred_low.npy"), image_pred_low.cpu().detach().numpy())
        np.save(osp.join(eval_save_dir, "image_pred_high.npy"), image_pred_high.cpu().detach().numpy())
        np.save(osp.join(eval_save_dir, "image_gt_low.npy"), image_gt_low.cpu().detach().numpy())
        np.save(osp.join(eval_save_dir, "image_gt_high.npy"), image_gt_high.cpu().detach().numpy())

        # Slice show: ρ field (one canonical view) + per-energy strips
        show_slice = 5
        show_step = max(1, rho_pred.shape[-1] // show_slice)
        rho_show = rho_pred[..., ::show_step][..., :show_slice]
        rho_strip = torch.cat([rho_show[..., s] for s in range(show_slice)], dim=1)
        self.writer.add_image("eval/rho_field", cast_to_image(rho_strip), global_step, dataformats="HWC")
        iio.imwrite(osp.join(eval_save_dir, "slice_show_rho.png"),
                    (cast_to_image(rho_strip) * 255).astype(np.uint8))

        for tag, gt_img, pred_img in (
            ("low", image_gt_low, image_pred_low),
            ("high", image_gt_high, image_pred_high),
        ):
            show_gt = gt_img[..., ::show_step][..., :show_slice]
            show_pred = pred_img[..., ::show_step][..., :show_slice]
            slabs = [torch.cat([show_gt[..., s], show_pred[..., s]], dim=0) for s in range(show_slice)]
            density_strip = torch.cat(slabs, dim=1)
            self.writer.add_image(f"eval/density_{tag}_row1gt_row2pred",
                                   cast_to_image(density_strip), global_step, dataformats="HWC")
            iio.imwrite(osp.join(eval_save_dir, f"slice_show_{tag}_row1gt_row2pred.png"),
                        (cast_to_image(density_strip) * 255).astype(np.uint8))

        # Projection PNGs
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

        with open(osp.join(eval_save_dir, "stats.txt"), "w") as f:
            for k, v in loss.items():
                v_scalar = v.item() if torch.is_tensor(v) else float(v)
                f.write(f"{k}: {v_scalar:.6f}\n")
            f.write(f"# notes:\n")
            f.write(f"#   PSNR_3d uses adaptive PIXEL_MAX (low={max_low:.4f}, high={max_high:.4f})\n")
            f.write(f"#   image_pred_low/high are derived from ρ × κ; ρ is the deliverable single field.\n")
            f.write(f"#   kappa_low={self.kappa_low}, kappa_high={self.kappa_high}\n")
            f.write(f"#   vol_ineq_violation_ratio should be ~0% (auto by construction).\n")

        return loss


if __name__ == "__main__":
    trainer = BasicTrainer_singlefield()
    trainer.start()
