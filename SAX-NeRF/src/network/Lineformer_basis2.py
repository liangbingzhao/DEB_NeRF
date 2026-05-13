"""Path D: 2-basis Lineformer with learnable second-basis attenuation coefficients.

Architecture (M9 Path D):
    (x,y,z) → encoder → backbone → (α_w(x), α_2(x)) ReLU outputs (non-negative)
    + global learnable scalars (κ_2_low, κ_2_high)
    Render output:
        μ_low(x)  = α_w(x) × κ_w_low  + α_2(x) × κ_2_low
        μ_high(x) = α_w(x) × κ_w_high + α_2(x) × κ_2_high

Where:
    κ_w_low, κ_w_high are FIXED NIST water constants in normalized scale
        (κ_w_low_norm  = 1.0,    matching image_low normalization)
        (κ_w_high_norm = 0.2543, = μ_water_60/μ_water_20)
    κ_2_low, κ_2_high are LEARNABLE (initialized to cortical bone in normalized scale)

Physics:
    α_w ≥ 0, α_2 ≥ 0 (via ReLU)
    κ_w_low > κ_w_high, init κ_2_low > κ_2_high → μ_low ≥ μ_high auto-satisfied
    "Single field" deliverable: ρ_total(x) = α_w(x) + α_2(x), energy-independent

forward() returns (N, 2) where [..., 0] = μ_low and [..., 1] = μ_high — directly
compatible with render_dual / raw2outputs_dual.

Use get_alphas(x) for analysis to retrieve (α_w, α_2) directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .Lineformer import Line_Attention_Blcok


class Lineformer_basis2(nn.Module):
    def __init__(self, encoder, bound=0.2, num_layers=8, hidden_dim=256, skips=[4], out_dim=1,
                 last_activation="relu", line_size=16, dim_head=32, heads=8, num_blocks=1,
                 kappa_w_low=1.0, kappa_w_high=0.2543,
                 kappa_2_low_init=1.121, kappa_2_high_init=0.0855,
                 constrain_kappa_2=False,
                 parameterization="alpha_direct"):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.bound = bound
        self.encoder = encoder
        self.in_dim = encoder.output_dim

        # Shared backbone (same as Lineformer_singlefield)
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_dim, hidden_dim)] +
            [Line_Attention_Blcok(dim=hidden_dim, line_size=line_size, dim_head=dim_head,
                                   heads=heads, num_blocks=num_blocks)
             if i not in skips
             else nn.Linear(hidden_dim + self.in_dim, hidden_dim)
             for i in range(1, num_layers - 1, 1)]
        )
        self.activations = nn.ModuleList([nn.LeakyReLU() for _ in range(num_layers - 1)])

        # Two-channel basis head (α_w, α_2)
        self.head_basis = nn.Linear(hidden_dim, 2)

        # Fixed water constants (registered as buffers, not learnable)
        self.register_buffer("kappa_w_low", torch.tensor(float(kappa_w_low)))
        self.register_buffer("kappa_w_high", torch.tensor(float(kappa_w_high)))

        # Optional positivity constraint on κ_2 via Softplus reparameterization
        # When True: stored param is "raw"; effective κ_2 = softplus(raw) > 0 always
        # Use inverse softplus on init so effective starts at bone-like values
        self.constrain_kappa_2 = bool(constrain_kappa_2)
        if self.constrain_kappa_2:
            def inv_softplus(y):
                # softplus(x) = log(1 + exp(x)) = y → x = log(exp(y) - 1)
                return float(torch.log(torch.expm1(torch.tensor(float(y)))).item())
            self.kappa_2_low  = nn.Parameter(torch.tensor(inv_softplus(kappa_2_low_init)))
            self.kappa_2_high = nn.Parameter(torch.tensor(inv_softplus(kappa_2_high_init)))
        else:
            self.kappa_2_low  = nn.Parameter(torch.tensor(float(kappa_2_low_init)))
            self.kappa_2_high = nn.Parameter(torch.tensor(float(kappa_2_high_init)))

        # parameterization mode:
        #   "alpha_direct": head outputs (α_w, α_2) directly via softplus/relu (current Path D)
        #   "rho_fraction": head outputs (ρ_total via softplus, f_w via sigmoid);
        #                   α_w = ρ_total × f_w, α_2 = ρ_total × (1 - f_w)
        self.parameterization = parameterization
        assert parameterization in ("alpha_direct", "rho_fraction"), f"unknown: {parameterization}"

        if parameterization == "alpha_direct":
            if last_activation == "relu":
                self.act = nn.ReLU()
            elif last_activation == "softplus":
                self.act = nn.Softplus()
            else:
                raise NotImplementedError(f"alpha_direct requires non-negative activation, got: {last_activation}")
        else:
            # rho_fraction uses softplus + sigmoid internally; last_activation is ignored
            self.act = None

    def get_kappa_2(self):
        """Return effective κ_2 values (after positivity constraint if enabled).

        Always non-negative when constrain_kappa_2=True (via softplus on the underlying
        raw parameter). When False, returns raw param values which may go negative
        during training (this is what produced the late-epoch fitting hack).
        """
        if self.constrain_kappa_2:
            return F.softplus(self.kappa_2_low), F.softplus(self.kappa_2_high)
        return self.kappa_2_low, self.kappa_2_high

    def _backbone(self, x):
        x = self.encoder(x, self.bound)
        input_pts = x[..., :self.in_dim]
        for i in range(len(self.layers)):
            layer = self.layers[i]
            activation = self.activations[i]
            if i in self.skips:
                x = torch.cat([input_pts, x], -1)
            x = layer(x)
            x = activation(x)
        return x

    def _alphas_from_raw(self, raw):
        """Internal: convert head_basis raw output (N, 2) to (α_w, α_2)
        and (if rho_fraction) ρ_total, f_w. Returns dict.
        """
        if self.parameterization == "rho_fraction":
            rho_total = F.softplus(raw[..., 0:1])              # (N, 1), ≥ 0
            f_w       = torch.sigmoid(raw[..., 1:2])           # (N, 1), ∈ [0, 1]
            alpha_w   = rho_total * f_w
            alpha_2   = rho_total * (1.0 - f_w)
            return {"alpha_w": alpha_w, "alpha_2": alpha_2,
                    "rho_total": rho_total, "f_w": f_w}
        else:
            alpha = self.act(raw)
            return {"alpha_w": alpha[..., 0:1], "alpha_2": alpha[..., 1:2]}

    def get_alphas(self, x):
        """Return (α_w, α_2) tuple. Backward-compat API."""
        feat = self._backbone(x)
        d = self._alphas_from_raw(self.head_basis(feat))
        return d["alpha_w"], d["alpha_2"]

    def get_decomposition(self, x):
        """Return dict with α_w, α_2, and (if rho_fraction) ρ_total, f_w."""
        feat = self._backbone(x)
        return self._alphas_from_raw(self.head_basis(feat))

    def forward(self, x):
        """
        input: (N, 3)
        output: (N, 2) — channel 0 = μ_low, channel 1 = μ_high (compat with render_dual)
        """
        feat = self._backbone(x)
        d = self._alphas_from_raw(self.head_basis(feat))
        alpha_w, alpha_2 = d["alpha_w"], d["alpha_2"]
        k2_low, k2_high = self.get_kappa_2()
        mu_low  = alpha_w * self.kappa_w_low  + alpha_2 * k2_low
        mu_high = alpha_w * self.kappa_w_high + alpha_2 * k2_high
        return torch.cat([mu_low, mu_high], dim=-1)
