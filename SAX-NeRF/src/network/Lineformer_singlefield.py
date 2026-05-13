"""Single-field variant of Lineformer for dual-energy supervision (Path B).

Architecture (M9 design, see plan):
    (x,y,z) → hash encoder → shared transformer/MLP backbone → latent → head_rho → ρ(x)

The single output ρ(x) is an energy-independent density-like field.
Per-energy μ values are computed at render time via fixed NIST water-curve scaling:
    μ_low(x)  = ρ(x) × κ_low   (κ_low  = (μ/ρ)_water(20 keV) = 0.8096)
    μ_high(x) = ρ(x) × κ_high  (κ_high = (μ/ρ)_water(60 keV) = 0.2059)

Backbone is identical to Lineformer_dual; only the head is single-channel.
"""

import torch
import torch.nn as nn

from .Lineformer import Line_Attention_Blcok


class Lineformer_singlefield(nn.Module):
    def __init__(self, encoder, bound=0.2, num_layers=8, hidden_dim=256, skips=[4], out_dim=1,
                 last_activation="sigmoid", line_size=16, dim_head=32, heads=8, num_blocks=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.bound = bound
        self.encoder = encoder
        self.in_dim = encoder.output_dim

        # Shared backbone — identical to Lineformer_dual (drops final Linear+Sigmoid of single-energy)
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_dim, hidden_dim)] +
            [Line_Attention_Blcok(dim=hidden_dim, line_size=line_size, dim_head=dim_head,
                                   heads=heads, num_blocks=num_blocks)
             if i not in skips
             else nn.Linear(hidden_dim + self.in_dim, hidden_dim)
             for i in range(1, num_layers - 1, 1)]
        )
        self.activations = nn.ModuleList([nn.LeakyReLU() for _ in range(num_layers - 1)])

        # Single density head — one channel output (energy-independent ρ)
        self.head_rho = nn.Linear(hidden_dim, out_dim)

        if last_activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif last_activation == "relu":
            self.act = nn.LeakyReLU()
        else:
            raise NotImplementedError(f"Unknown last activation: {last_activation}")

    def forward(self, x):
        """
        input: (N, 3)
        output: (N, 1) — ρ values (energy-independent density field)
        """
        x = self.encoder(x, self.bound)
        input_pts = x[..., :self.in_dim]

        for i in range(len(self.layers)):
            layer = self.layers[i]
            activation = self.activations[i]

            if i in self.skips:
                x = torch.cat([input_pts, x], -1)

            x = layer(x)
            x = activation(x)

        rho = self.act(self.head_rho(x))  # (N, out_dim) where out_dim=1
        return rho
