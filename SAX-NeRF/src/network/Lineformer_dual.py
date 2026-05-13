"""Dual-energy variant of Lineformer: shared backbone + two density heads.

Architecture (see docs/idea.md §3.1):
    (x,y,z) → hash encoder → shared transformer/MLP backbone → latent
                                                                  ├── head_low  → μ_low(x)
                                                                  └── head_high → μ_high(x)

Output is concatenated (N, 2): channel 0 is μ_low, channel 1 is μ_high.
Both heads have independent Linear weights and independent Sigmoid activations.
The shared backbone is identical to single-energy Lineformer (transformer-attention + MLP).
"""

import torch
import torch.nn as nn

from .Lineformer import Line_Attention_Blcok


class Lineformer_dual(nn.Module):
    def __init__(self, encoder, bound=0.2, num_layers=8, hidden_dim=256, skips=[4], out_dim=1,
                 last_activation="sigmoid", line_size=16, dim_head=32, heads=8, num_blocks=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.bound = bound
        self.encoder = encoder
        self.in_dim = encoder.output_dim

        # Shared backbone: identical to single-energy Lineformer EXCEPT we drop the final Linear+Sigmoid
        # (replaced by the two parallel heads below).
        # Layers 0 .. num_layers-2 (inclusive) form the shared backbone.
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_dim, hidden_dim)] +
            [Line_Attention_Blcok(dim=hidden_dim, line_size=line_size, dim_head=dim_head,
                                   heads=heads, num_blocks=num_blocks)
             if i not in skips
             else nn.Linear(hidden_dim + self.in_dim, hidden_dim)
             for i in range(1, num_layers - 1, 1)]
        )
        # Activations for the backbone (one less than single-energy because last Linear was removed)
        self.activations = nn.ModuleList([nn.LeakyReLU() for _ in range(num_layers - 1)])

        # Two parallel density heads — independent weights and biases
        self.head_low = nn.Linear(hidden_dim, out_dim)
        self.head_high = nn.Linear(hidden_dim, out_dim)

        # Final activation per head (independent Sigmoid modules so future per-head changes are easy)
        if last_activation == "sigmoid":
            self.act_low = nn.Sigmoid()
            self.act_high = nn.Sigmoid()
        elif last_activation == "relu":
            self.act_low = nn.LeakyReLU()
            self.act_high = nn.LeakyReLU()
        else:
            raise NotImplementedError(f"Unknown last activation: {last_activation}")

    def forward(self, x):
        """
        input: (N, 3)
        output: (N, 2) where [..., 0] = μ_low and [..., 1] = μ_high
        """
        x = self.encoder(x, self.bound)
        input_pts = x[..., :self.in_dim]

        # Shared backbone forward (note: we iterate over self.layers which has num_layers-1 entries now)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            activation = self.activations[i]

            if i in self.skips:
                x = torch.cat([input_pts, x], -1)

            x = layer(x)
            x = activation(x)

        # Per-energy heads on the shared latent
        mu_low = self.act_low(self.head_low(x))    # (N, out_dim)
        mu_high = self.act_high(self.head_high(x))  # (N, out_dim)

        return torch.cat([mu_low, mu_high], dim=-1)  # (N, 2 * out_dim); for out_dim=1 this is (N, 2)
