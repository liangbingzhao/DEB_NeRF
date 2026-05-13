"""Dual-energy TIGRE dataset.

Loads a paired pickle produced by data_preprocess/convert_walnut_dual_v3.py with fields:
    image_low, image_high           - GT volumes (R,R,R), shared normalization
    train.angles                    - shared train angles
    train.projections_low/_high     - paired train projections
    val.angles                      - shared val angles
    val.projections_low/_high       - paired val projections
    + standard geometry fields (DSD, DSO, nVoxel, ...)

Returns per __getitem__:
    train: {"rays", "projs_low", "projs_high"}
    val:   {"rays", "projs_low", "projs_high"}
"""

import torch
import pickle
import numpy as np
from torch.utils.data import Dataset

from .tigre_mlg import ConeGeometry, ray_window_partition, proj_window_partition


class TIGREDataset_MLG_dual(Dataset):
    """Dual-energy variant of TIGREDataset_MLG.

    Differences from single-energy version:
    - Loads image_low/image_high (instead of single image)
    - Loads projections_low/projections_high (instead of single projections)
    - Window selection uses sum(low + high) so windows with ANY signal in either
      energy are kept (more robust than picking based on one energy alone)
    - Returns paired projections per ray
    """

    def __init__(self, path, n_rays=1024, type="train", window_size=[32, 32], window_num=4,
                 device="cuda", biased_sampling=False):
        super().__init__()

        with open(path, "rb") as handle:
            data = pickle.load(handle)

        # Sanity: confirm this is a dual-energy pickle
        for k in ("image_low", "image_high"):
            assert k in data, f"Dual pickle missing required field '{k}'"
        for split in ("train", "val"):
            for k in (f"projections_low", f"projections_high", "angles"):
                assert k in data[split], f"Dual pickle missing field '{split}.{k}'"

        self.geo = ConeGeometry(data)
        self.window_size = window_size
        self.window_num = window_num
        self.biased_sampling = biased_sampling
        self.type = type
        self.n_rays = n_rays
        self.near, self.far = self.get_near_far(self.geo)

        if type == "train":
            self.projs_low = torch.tensor(data["train"]["projections_low"], dtype=torch.float32, device=device)
            self.projs_high = torch.tensor(data["train"]["projections_high"], dtype=torch.float32, device=device)
            angles = data["train"]["angles"]
            rays = self.get_rays(angles, self.geo, device)
            self.rays = torch.cat([
                rays,
                torch.ones_like(rays[..., :1]) * self.near,
                torch.ones_like(rays[..., :1]) * self.far
            ], dim=-1)
            self.n_samples = data["numTrain"]
            coords = torch.stack(torch.meshgrid(
                torch.linspace(0, self.geo.nDetector[1] - 1, self.geo.nDetector[1], device=device),
                torch.linspace(0, self.geo.nDetector[0] - 1, self.geo.nDetector[0], device=device),
                indexing="ij"), -1)
            self.coords = torch.reshape(coords, [-1, 2])
            self.image_low = torch.tensor(data["image_low"], dtype=torch.float32, device=device)
            self.image_high = torch.tensor(data["image_high"], dtype=torch.float32, device=device)
            self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=device)

        elif type == "val":
            self.projs_low = torch.tensor(data["val"]["projections_low"], dtype=torch.float32, device=device)
            self.projs_high = torch.tensor(data["val"]["projections_high"], dtype=torch.float32, device=device)
            angles = data["val"]["angles"]
            rays = self.get_rays(angles, self.geo, device)
            self.rays = torch.cat([
                rays,
                torch.ones_like(rays[..., :1]) * self.near,
                torch.ones_like(rays[..., :1]) * self.far
            ], dim=-1)
            self.n_samples = data["numVal"]
            self.image_low = torch.tensor(data["image_low"], dtype=torch.float32, device=device)
            self.image_high = torch.tensor(data["image_high"], dtype=torch.float32, device=device)
            self.voxels = torch.tensor(self.get_voxels(self.geo), dtype=torch.float32, device=device)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if self.type == "train":
            rays = self.rays[index]                # [H, W, 8]
            projs_low = self.projs_low[index]      # [H, W]
            projs_high = self.projs_high[index]    # [H, W]
            # Combined mask: a window is "valid" if both energies have signal across it
            projs_combined = projs_low + projs_high

            rays_window = ray_window_partition(rays, self.window_size)             # [Nw, h, w, 8]
            projs_low_window = proj_window_partition(projs_low, self.window_size)  # [Nw, h, w]
            projs_high_window = proj_window_partition(projs_high, self.window_size)
            projs_combined_window = proj_window_partition(projs_combined, self.window_size)

            # Valid windows: every pixel positive in COMBINED projection (i.e., at least one energy has signal)
            window_pixel_count = self.window_size[0] * self.window_size[1]
            projs_window_valid_indx = (
                (projs_combined_window > 0).sum(dim=-1).sum(dim=-1) == window_pixel_count
            )

            # Window selection (with optional biased sampling on combined projection mean)
            if self.biased_sampling:
                window_means = projs_combined_window.mean(dim=-1).mean(dim=-1).cpu().numpy()
                window_weights = window_means / (window_means.sum() + 1e-8)
                select_inds_window = np.random.choice(
                    projs_window_valid_indx.shape[0], size=[self.window_num],
                    replace=False, p=window_weights)
            else:
                select_inds_window = np.random.choice(
                    projs_window_valid_indx.shape[0], size=[self.window_num], replace=False)

            # Selected windows
            rays_window_select = rays_window[select_inds_window]            # [Nw_sel, h, w, 8]
            projs_low_window_select = projs_low_window[select_inds_window]
            projs_high_window_select = projs_high_window[select_inds_window]

            selected_rays_window = rays_window_select.reshape(-1, 8)
            selected_projs_low_window = projs_low_window_select.flatten()
            selected_projs_high_window = projs_high_window_select.flatten()

            # Scattered rays from non-selected windows (combined-mask filtered)
            total_inds = list(range(projs_combined_window.shape[0]))
            else_inds = [x for x in total_inds if x not in select_inds_window]
            rays_window_else = rays_window[else_inds]
            projs_low_window_else = projs_low_window[else_inds]
            projs_high_window_else = projs_high_window[else_inds]
            projs_combined_window_else = projs_combined_window[else_inds]

            # Pixel-level validity by combined mask
            else_inds_pixel_valid = projs_combined_window_else > 0

            rays_else_valid = rays_window_else[else_inds_pixel_valid]
            projs_low_else_valid = projs_low_window_else[else_inds_pixel_valid]
            projs_high_else_valid = projs_high_window_else[else_inds_pixel_valid]

            else_valid_select_index = np.random.choice(
                projs_low_else_valid.shape[0], size=[self.n_rays], replace=False)

            selected_rays_else = rays_else_valid[else_valid_select_index]
            selected_projs_low_else = projs_low_else_valid[else_valid_select_index]
            selected_projs_high_else = projs_high_else_valid[else_valid_select_index]

            selected_rays = torch.concat([selected_rays_window, selected_rays_else], dim=0)
            selected_projs_low = torch.concat([selected_projs_low_window, selected_projs_low_else], dim=0)
            selected_projs_high = torch.concat([selected_projs_high_window, selected_projs_high_else], dim=0)

            return {
                "projs_low": selected_projs_low,
                "projs_high": selected_projs_high,
                "rays": selected_rays,
            }
        elif self.type == "val":
            return {
                "projs_low": self.projs_low[index],
                "projs_high": self.projs_high[index],
                "rays": self.rays[index],
            }

    # The geometry helpers below mirror TIGREDataset_MLG. Keeping them here as static methods
    # (vs. inheriting) so this dataset doesn't depend on TIGREDataset_MLG's __init__ signature.

    def get_voxels(self, geo: ConeGeometry):
        n1, n2, n3 = geo.nVoxel
        s1, s2, s3 = geo.sVoxel / 2 - geo.dVoxel / 2
        xyz = np.meshgrid(
            np.linspace(-s1, s1, n1),
            np.linspace(-s2, s2, n2),
            np.linspace(-s3, s3, n3), indexing="ij")
        return np.asarray(xyz).transpose([1, 2, 3, 0])

    def get_rays(self, angles, geo: ConeGeometry, device):
        W, H = geo.nDetector
        DSD = geo.DSD
        rays = []
        for angle in angles:
            pose = torch.Tensor(self.angle2pose(geo.DSO, angle)).to(device)
            if geo.mode == "cone":
                i, j = torch.meshgrid(
                    torch.linspace(0, W - 1, W, device=device),
                    torch.linspace(0, H - 1, H, device=device), indexing="ij")
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = torch.stack([uu / DSD, vv / DSD, torch.ones_like(uu)], -1)
                rays_d = torch.sum(torch.matmul(pose[:3, :3], dirs[..., None]).to(device), -1)
                rays_o = pose[:3, -1].expand(rays_d.shape)
            elif geo.mode == "parallel":
                i, j = torch.meshgrid(
                    torch.linspace(0, W - 1, W, device=device),
                    torch.linspace(0, H - 1, H, device=device), indexing="ij")
                uu = (i.t() + 0.5 - W / 2) * geo.dDetector[0] + geo.offDetector[0]
                vv = (j.t() + 0.5 - H / 2) * geo.dDetector[1] + geo.offDetector[1]
                dirs = torch.stack([torch.zeros_like(uu), torch.zeros_like(uu), torch.ones_like(uu)], -1)
                rays_d = torch.sum(torch.matmul(pose[:3, :3], dirs[..., None]).to(device), -1)
                rays_o = torch.sum(
                    torch.matmul(pose[:3, :3], torch.stack([uu, vv, torch.zeros_like(uu)], -1)[..., None]).to(device),
                    -1) + pose[:3, -1].expand(rays_d.shape)
            else:
                raise NotImplementedError("Unknown CT scanner type!")
            rays.append(torch.concat([rays_o, rays_d], dim=-1))
        return torch.stack(rays, dim=0)

    def angle2pose(self, DSO, angle):
        phi1 = -np.pi / 2
        R1 = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(phi1), -np.sin(phi1)],
                       [0.0, np.sin(phi1), np.cos(phi1)]])
        phi2 = np.pi / 2
        R2 = np.array([[np.cos(phi2), -np.sin(phi2), 0.0],
                       [np.sin(phi2), np.cos(phi2), 0.0],
                       [0.0, 0.0, 1.0]])
        R3 = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                       [np.sin(angle), np.cos(angle), 0.0],
                       [0.0, 0.0, 1.0]])
        rot = np.dot(np.dot(R3, R2), R1)
        trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
        T = np.eye(4)
        T[:-1, :-1] = rot
        T[:-1, -1] = trans
        return T

    def get_near_far(self, geo: ConeGeometry, tolerance=0.005):
        dist1 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
        dist2 = np.linalg.norm([geo.offOrigin[0] - geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
        dist3 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] - geo.sVoxel[1] / 2])
        dist4 = np.linalg.norm([geo.offOrigin[0] + geo.sVoxel[0] / 2, geo.offOrigin[1] + geo.sVoxel[1] / 2])
        dist_max = np.max([dist1, dist2, dist3, dist4])
        near = np.max([0, geo.DSO - dist_max - tolerance])
        far = np.min([geo.DSO * 2, geo.DSO + dist_max + tolerance])
        return near, far
