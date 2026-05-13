"""Dual-energy variant of Trainer.

Difference from src/trainer_mlg.py: instantiates TIGREDataset_MLG_dual instead of
TIGREDataset_MLG. The training loop, optimizer, scheduler, and checkpointing logic
are identical. Subclasses (e.g., BasicTrainer_dual in train_mlg_dual.py) override
compute_loss() and eval_step() to handle paired projections.
"""

import os
import os.path as osp
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from shutil import copyfile

from .dataset import TIGREDataset_MLG_dual as Dataset_MLG_dual
from .utils import gen_log, time2file_name
import datetime

from .network import get_network
from .encoder import get_encoder


class Trainer_dual:
    def __init__(self, cfg, device="cuda"):
        self.global_step = 0
        self.conf = cfg
        self.n_fine = cfg["render"]["n_fine"]
        self.epochs = cfg["train"]["epoch"]
        self.i_eval = cfg["log"]["i_eval"]
        self.i_save = cfg["log"]["i_save"]
        self.netchunk = cfg["render"]["netchunk"]
        self.n_rays = cfg["train"]["n_rays"]

        # Log directory
        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)
        self.expdir = osp.join(cfg["exp"]["expdir"], cfg["exp"]["expname"], date_time)
        self.ckptdir = osp.join(self.expdir, "ckpt.tar")
        self.ckptdir_backup = osp.join(self.expdir, "ckpt_backup.tar")
        self.ckpt_best_dir = osp.join(self.expdir, "ckpt_best.tar")
        self.best_psnr_3d = 0  # tracks PSNR_3d_avg = (low + high) / 2
        self.evaldir = osp.join(self.expdir, "eval")
        os.makedirs(self.evaldir, exist_ok=True)
        self.logger = gen_log(self.expdir)

        # Dataset (dual)
        biased = cfg["train"].get("biased_sampling", False)
        train_dset = Dataset_MLG_dual(
            cfg["exp"]["datadir"], cfg["train"]["n_rays"], "train",
            cfg["train"]["window_size"], cfg["train"]["window_num"],
            device, biased_sampling=biased)
        self.eval_dset = Dataset_MLG_dual(
            cfg["exp"]["datadir"], cfg["train"]["n_rays"], "val",
            cfg["train"]["window_size"], cfg["train"]["window_num"],
            device) if self.i_eval > 0 else None
        self.train_dloader = torch.utils.data.DataLoader(train_dset, batch_size=cfg["train"]["n_batch"])
        self.voxels = self.eval_dset.voxels if self.i_eval > 0 else None

        # Network (Lineformer_dual is selected via cfg.network.net_type)
        network = get_network(cfg["network"]["net_type"])
        cfg["network"].pop("net_type", None)
        encoder = get_encoder(**cfg["encoder"])
        self.net = network(encoder, **cfg["network"]).to(device)
        grad_vars = list(self.net.parameters())
        self.net_fine = None
        if self.n_fine > 0:
            self.net_fine = network(encoder, **cfg["network"]).to(device)
            grad_vars += list(self.net_fine.parameters())

        # Optimizer + LR scheduler
        self.optimizer = torch.optim.Adam(params=grad_vars, lr=cfg["train"]["lrate"], betas=(0.9, 0.999))
        scheduler_type = cfg["train"].get("scheduler", "step")
        if scheduler_type == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer, T_max=self.epochs, eta_min=cfg["train"].get("lrate_min", 1e-5))
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self.optimizer, step_size=cfg["train"]["lrate_step"], gamma=cfg["train"]["lrate_gamma"])

        # Checkpoint resume
        self.epoch_start = 0
        if cfg["train"]["resume"] and osp.exists(self.ckptdir):
            print(f"Load checkpoints from {self.ckptdir}.")
            ckpt = torch.load(self.ckptdir)
            self.epoch_start = ckpt["epoch"] + 1
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.global_step = self.epoch_start * len(self.train_dloader)
            self.net.load_state_dict(ckpt["network"])
            if self.n_fine > 0:
                self.net_fine.load_state_dict(ckpt["network_fine"])

        # TensorBoard
        self.writer = SummaryWriter(self.expdir)
        self.writer.add_text("parameters", self.args2string(cfg), global_step=0)

    def args2string(self, hp):
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))

    def start(self):
        """Main loop. Identical structure to single-energy trainer."""
        self.logger.info(self.conf)

        def fmt_loss_str(losses):
            parts = []
            for k in losses:
                v = losses[k]
                if torch.is_tensor(v):
                    v = v.item()
                parts.append(f", {k}: {v:.4g}")
            return "".join(parts)

        iter_per_epoch = len(self.train_dloader)
        pbar = tqdm(total=iter_per_epoch * self.epochs, leave=True)
        if self.epoch_start > 0:
            pbar.update(self.epoch_start * iter_per_epoch)

        for idx_epoch in range(self.epoch_start, self.epochs + 1):
            # Eval
            if (idx_epoch % self.i_eval == 0 or idx_epoch == self.epochs) and self.i_eval > 0:
                self.net.eval()
                with torch.no_grad():
                    loss_test = self.eval_step(global_step=self.global_step, idx_epoch=idx_epoch)
                self.net.train()
                tqdm.write(f"[EVAL] epoch: {idx_epoch}/{self.epochs}{fmt_loss_str(loss_test)}")
                self.logger.info(f"[EVAL] epoch: {idx_epoch}/{self.epochs}{fmt_loss_str(loss_test)}")

            # Train
            for data in self.train_dloader:
                self.global_step += 1
                self.net.train()
                loss_train = self.train_step(data, global_step=self.global_step, idx_epoch=idx_epoch)
                pbar.set_description(
                    f"epoch={idx_epoch}/{self.epochs}, loss={loss_train:.4g}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.4g}")
                pbar.update(1)

            if idx_epoch % 10 == 0:
                self.logger.info(
                    f"epoch={idx_epoch}/{self.epochs}, loss={loss_train:.4g}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.4g}")

            # Save checkpoint
            if (idx_epoch % self.i_save == 0 or idx_epoch == self.epochs) and self.i_save > 0 and idx_epoch > 0:
                if osp.exists(self.ckptdir):
                    copyfile(self.ckptdir, self.ckptdir_backup)
                tqdm.write(f"[SAVE] epoch: {idx_epoch}/{self.epochs}, path: {self.ckptdir}")
                self.logger.info(f"[SAVE] epoch: {idx_epoch}/{self.epochs}, path: {self.ckptdir}")
                torch.save(
                    {
                        "epoch": idx_epoch,
                        "network": self.net.state_dict(),
                        "network_fine": self.net_fine.state_dict() if self.n_fine > 0 else None,
                        "optimizer": self.optimizer.state_dict(),
                    },
                    self.ckptdir,
                )

            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
            self.lr_scheduler.step()

        tqdm.write(f"Training complete! See logs in {self.expdir}")

    def train_step(self, data, global_step, idx_epoch):
        self.optimizer.zero_grad()
        loss = self.compute_loss(data, global_step, idx_epoch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # Subclasses must override these:
    def compute_loss(self, data, global_step, idx_epoch):
        raise NotImplementedError("Subclass must implement compute_loss")

    def eval_step(self, global_step, idx_epoch):
        raise NotImplementedError("Subclass must implement eval_step")
