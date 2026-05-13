"""Path D training entry: 2-basis (water + learned) single-field reconstruction.

Network: Lineformer_basis2 — outputs (μ_low, μ_high) computed as:
    μ_low(x)  = α_w(x) × κ_w_low_norm  + α_2(x) × κ_2_low
    μ_high(x) = α_w(x) × κ_w_high_norm + α_2(x) × κ_2_high
Where α_w, α_2 are non-negative (ReLU), κ_w_* are fixed NIST water constants in
normalized scale, and κ_2_* are learnable scalars (init to cortical bone).

Single-field deliverable: ρ_total(x) = α_w(x) + α_2(x), energy-independent.

Reuses Trainer_dual + render_dual since forward returns (N, 2) μ_low/μ_high tuple.
Overrides compute_loss + eval_step to record (α_w, α_2, κ_2 evolution).
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
from src.render import render_dual, run_network
from src.trainer_mlg_dual import Trainer_dual
from src.utils import get_psnr, get_ssim, get_psnr_3d, get_ssim_3d, cast_to_image


cfg = load_config(args.config)
device = torch.device("cuda")


class BasicTrainer_basis2(Trainer_dual):
    def __init__(self):
        super().__init__(cfg, device)
        # Log initial κ_2 values (effective, after positivity constraint if enabled)
        k2_low0, k2_high0 = self.net.get_kappa_2()
        print(f"[Start] basis2: expname={cfg['exp']['expname']}")
        print(f"  constrain_kappa_2: {self.net.constrain_kappa_2}")
        print(f"  Fixed (water): κ_w_low={float(self.net.kappa_w_low):.4f}, "
              f"κ_w_high={float(self.net.kappa_w_high):.4f}")
        print(f"  Learnable (effective init): κ_2_low={float(k2_low0):.4f}, "
              f"κ_2_high={float(k2_high0):.4f}")

    def compute_loss(self, data, global_step, idx_epoch):
        rays = data["rays"].reshape(-1, 8)
        projs_low = data["projs_low"].reshape(-1)
        projs_high = data["projs_high"].reshape(-1)

        ret = render_dual(rays, self.net, self.net_fine, **self.conf["render"])
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

        # Record current learnable κ_2 values
        k2_low_eff, k2_high_eff = self.net.get_kappa_2()
        self.writer.add_scalar("train/kappa_2_low",  float(k2_low_eff),  global_step)
        self.writer.add_scalar("train/kappa_2_high", float(k2_high_eff), global_step)
        return loss_total

    def eval_step(self, global_step, idx_epoch):
        # ---- Project both energies along val rays ----
        projs_low_gt = self.eval_dset.projs_low
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

        # ---- Reconstruct full 3D volumes for both energies + α_w, α_2 separately ----
        net_for_eval = self.net_fine if self.net_fine is not None else self.net
        # Per-energy μ volumes (via forward())
        image_pred_dual = run_network(self.eval_dset.voxels, net_for_eval, self.netchunk)  # [R,R,R,2]
        image_pred_low = image_pred_dual[..., 0]
        image_pred_high = image_pred_dual[..., 1]

        # α_w, α_2 (and ρ_total, f_w if rho_fraction) via get_decomposition in chunks
        voxels = self.eval_dset.voxels
        voxel_shape = voxels.shape[:-1]
        flat = voxels.reshape(-1, 3)
        decomp_chunks = []
        with torch.no_grad():
            for i in range(0, flat.shape[0], self.netchunk):
                decomp_chunks.append(net_for_eval.get_decomposition(flat[i:i + self.netchunk]))
        alpha_w = torch.cat([d["alpha_w"] for d in decomp_chunks], 0).reshape(voxel_shape + (1,))[..., 0]
        alpha_2 = torch.cat([d["alpha_2"] for d in decomp_chunks], 0).reshape(voxel_shape + (1,))[..., 0]
        # rho_fraction extra fields:
        f_w_field = None
        rho_param_field = None  # the network's direct ρ_total output (rho_fraction mode only)
        if "f_w" in decomp_chunks[0]:
            f_w_field = torch.cat([d["f_w"] for d in decomp_chunks], 0).reshape(voxel_shape + (1,))[..., 0]
            rho_param_field = torch.cat([d["rho_total"] for d in decomp_chunks], 0).reshape(voxel_shape + (1,))[..., 0]
        # ρ_total deliverable: always = α_w + α_2 (which equals rho_param_field in rho_fraction mode)
        rho_total = alpha_w + alpha_2

        image_gt_low = self.eval_dset.image_low
        image_gt_high = self.eval_dset.image_high

        # ---- Metrics (PIXEL_MAX adaptive) ----
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

        proj_violation_pixels = int((projs_pred_low < projs_pred_high).sum().item())
        proj_total_pixels = int(projs_pred_low.numel())
        proj_violation_ratio = proj_violation_pixels / proj_total_pixels
        vol_violation = int((image_pred_low < image_pred_high).sum().item())
        vol_total = int(image_pred_low.numel())
        vol_violation_ratio = vol_violation / vol_total

        psnr_3d_avg_t = (psnr_3d_low + psnr_3d_high) / 2.0

        # Path D specific monitor: track κ_2 learning + α stats
        # (use effective κ_2 — after softplus constraint if enabled)
        k2_low_t, k2_high_t = net_for_eval.get_kappa_2()
        k2_low = float(k2_low_t.item())
        k2_high = float(k2_high_t.item())
        aw_mean = float(alpha_w.mean().item())
        a2_mean = float(alpha_2.mean().item())
        rho_mean = float(rho_total.mean().item())
        rho_max = float(rho_total.max().item())

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
            "kappa_2_low": torch.tensor(k2_low),
            "kappa_2_high": torch.tensor(k2_high),
            "alpha_w_mean": torch.tensor(aw_mean),
            "alpha_2_mean": torch.tensor(a2_mean),
            "rho_total_mean": torch.tensor(rho_mean),
            "rho_total_max": torch.tensor(rho_max),
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

        eval_save_dir = osp.join(self.evaldir, f"epoch_{idx_epoch:05d}")
        os.makedirs(eval_save_dir, exist_ok=True)
        # Path D specific: save α_w, α_2, ρ_total
        np.save(osp.join(eval_save_dir, "alpha_w.npy"), alpha_w.cpu().detach().numpy())
        np.save(osp.join(eval_save_dir, "alpha_2.npy"), alpha_2.cpu().detach().numpy())
        np.save(osp.join(eval_save_dir, "rho_total.npy"), rho_total.cpu().detach().numpy())
        # rho_fraction extras
        if f_w_field is not None:
            np.save(osp.join(eval_save_dir, "f_w.npy"), f_w_field.cpu().detach().numpy())
            np.save(osp.join(eval_save_dir, "rho_param.npy"), rho_param_field.cpu().detach().numpy())
        np.save(osp.join(eval_save_dir, "image_pred_low.npy"), image_pred_low.cpu().detach().numpy())
        np.save(osp.join(eval_save_dir, "image_pred_high.npy"), image_pred_high.cpu().detach().numpy())
        np.save(osp.join(eval_save_dir, "image_gt_low.npy"), image_gt_low.cpu().detach().numpy())
        np.save(osp.join(eval_save_dir, "image_gt_high.npy"), image_gt_high.cpu().detach().numpy())

        # Slice show: ρ_total + α_w + α_2 + per-energy
        show_slice = 5
        for tag, img in (
            ("rho_total", rho_total),
            ("alpha_w", alpha_w),
            ("alpha_2", alpha_2),
        ):
            show_step = max(1, img.shape[-1] // show_slice)
            show = img[..., ::show_step][..., :show_slice]
            strip = torch.cat([show[..., s] for s in range(show_slice)], dim=1)
            self.writer.add_image(f"eval/{tag}", cast_to_image(strip), global_step, dataformats="HWC")
            iio.imwrite(osp.join(eval_save_dir, f"slice_show_{tag}.png"),
                        (cast_to_image(strip) * 255).astype(np.uint8))

        for tag, gt_img, pred_img in (
            ("low", image_gt_low, image_pred_low),
            ("high", image_gt_high, image_pred_high),
        ):
            show_step = max(1, gt_img.shape[-1] // show_slice)
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
            f.write(f"#   PSNR_3d uses adaptive PIXEL_MAX (low={max_low:.4f}, high={max_high:.4f}).\n")
            f.write(f"#   Single-field deliverable: rho_total = alpha_w + alpha_2 (energy-independent).\n")
            f.write(f"#   κ_w_low={float(net_for_eval.kappa_w_low):.4f} (fixed water).\n")
            f.write(f"#   κ_w_high={float(net_for_eval.kappa_w_high):.4f} (fixed water).\n")
            f.write(f"#   κ_2_low={k2_low:.4f} (learnable, init=1.121 = bone-like).\n")
            f.write(f"#   κ_2_high={k2_high:.4f} (learnable, init=0.0855 = bone-like).\n")

        return loss


if __name__ == "__main__":
    trainer = BasicTrainer_basis2()
    trainer.start()
