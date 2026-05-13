import torch
import torch.nn.functional as F


def calc_mse_loss(loss, x, y):
    """
    Calculate mse loss.
    """
    # Compute loss
    loss_mse = torch.mean((x-y)**2)
    loss["loss"] += loss_mse
    loss["loss_mse"] = loss_mse
    return loss

def calc_mse_loss_raw(loss, x, y, k = 1):
    """
    Calculate mse loss for raw.
    """
    # Compute loss for raw
    loss_mse_raw = torch.mean((x-y)**2)
    loss["loss"] += k * loss_mse_raw
    loss["loss_mse_raw"] = loss_mse_raw
    return loss

def calc_tv_loss(loss, x, k):
    """
    Calculate total variation loss.
    Args:
        x (n1, n2, n3, 1): 3d density field.
        k: relative weight
    """
    n1, n2, n3 = x.shape
    tv_1 = torch.abs(x[1:,1:,1:]-x[:-1,1:,1:]).sum()
    tv_2 = torch.abs(x[1:,1:,1:]-x[1:,:-1,1:]).sum()
    tv_3 = torch.abs(x[1:,1:,1:]-x[1:,1:,:-1]).sum()
    tv = (tv_1+tv_2+tv_3) / (n1*n2*n3)
    loss["loss"] += tv * k
    loss["loss_tv"] = tv * k
    return loss


def _grad3d(x):
    """Forward finite-diff gradient on a 3D voxel grid. Returns shape [G-1, G-1, G-1, 3]."""
    gx = x[1:, 1:, 1:] - x[:-1, 1:, 1:]
    gy = x[1:, 1:, 1:] - x[1:, :-1, 1:]
    gz = x[1:, 1:, 1:] - x[1:, 1:, :-1]
    return torch.stack([gx, gy, gz], dim=-1)


def calc_structural_loss(mu_low, mu_high, mask_percentile=0.5, eps=1e-8):
    """M5 v1 (cossim): gradient direction alignment via cosine similarity on edge voxels.

    L = 1 - mean(cos_sim(grad mu_low, grad mu_high)) on |grad mu_low| > tau.
    Known issue: trivial minimum (∇=0 globally) is not penalized — network can
    smooth edges to satisfy alignment objective. Kept for ablation.
    """
    g_low = _grad3d(mu_low)
    g_high = _grad3d(mu_high)
    cos = F.cosine_similarity(g_low, g_high, dim=-1, eps=eps)

    if mask_percentile > 0.0:
        mag = torch.linalg.norm(g_low, dim=-1)
        tau = torch.quantile(mag.flatten(), mask_percentile)
        mask = mag > tau
        if mask.any():
            cos = cos[mask]

    return 1.0 - cos.mean()


def calc_structural_loss_magweight(mu_low, mu_high, eps=1e-8):
    """M5 v2 (magweight): direction alignment weighted by per-voxel min gradient magnitude.

    L = mean[(1 - cos_sim) * min(|grad mu_low|, |grad mu_high|)]

    Trivial minimum (∇=0) yields 0 loss but no benefit — recon objective fights
    smoothing. Penalty is only paid where BOTH energies have signal AND directions disagree.
    """
    g_low = _grad3d(mu_low)
    g_high = _grad3d(mu_high)
    cos = F.cosine_similarity(g_low, g_high, dim=-1, eps=eps)
    mag_min = torch.minimum(
        torch.linalg.norm(g_low, dim=-1),
        torch.linalg.norm(g_high, dim=-1),
    )
    return torch.mean((1.0 - cos) * mag_min)


def calc_structural_loss_diff_tv(mu_low, mu_high, alpha=3.80):
    """M5 v3 (difftv): TV of difference image (mu_low - alpha * mu_high).

    Physics: if ratio mu_low/mu_high is constant alpha, the difference image
    is smooth and its gradients vanish. alpha=3.80 = M3.5 GT ratio median
    (NIST water at 20/60 keV is 3.93; data-driven value used).
    """
    diff = mu_low - alpha * mu_high
    gx = diff[1:, 1:, 1:] - diff[:-1, 1:, 1:]
    gy = diff[1:, 1:, 1:] - diff[1:, :-1, 1:]
    gz = diff[1:, 1:, 1:] - diff[1:, 1:, :-1]
    return torch.mean(torch.abs(gx) + torch.abs(gy) + torch.abs(gz))
