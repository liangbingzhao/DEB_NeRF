"""
M9 Phase D — Compute Path D yaml calibration constants for any (low, high) keV pair.

NIST source (cortical bone & water):
  https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/bone.html
  https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html
Material: "Bone, Cortical (ICRU-44)" — same as doc / winner derivation.

Outputs (printed):
  κ_w_low         : 1.0 by convention (defines α_w units)
  κ_w_high        : MU_WATER(high) / MU_WATER(low)
  shared_scale    : empirical (from convert_walnut_dual_phys); ESTIMATE if not given
                    ≈ 4.41 × μ_water(low), where K=4.41 from (20,60) baseline empirical
  κ_2_low_init    : μ_bone(low) / shared_scale
  κ_2_high_init   : μ_bone(high) / shared_scale

For 70 keV / 90 keV (not in NIST table), values are log-log interpolated from
adjacent NIST anchors.

Sanity check for (20, 60):
  κ_w_high     expected 0.2543 ✓
  κ_2_low_init  expected 1.121  ✓ (matches winner)
  κ_2_high_init expected 0.0883 (NIST authoritative; winner used 0.0855 from
                                 a slightly different transcription, see exp_log.md)

Usage:
  python experiments/compute_pair_calibration.py --low 20 --high 60
  python experiments/compute_pair_calibration.py --low 30 --high 80 --shared_scale 1.65
  python experiments/compute_pair_calibration.py --all  # baseline + 4 new pairs
"""
from __future__ import annotations

import argparse
import math


# NIST XCOM "Water" mass attenuation coefficients (cm²/g).
# Source: https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html
# Verified against MU_WATER in data_preprocess/convert_walnut_phys.py.
MU_WATER_NIST = {
    10: 5.329, 15: 1.673, 20: 0.8096, 30: 0.3756, 40: 0.2683,
    50: 0.2270, 60: 0.2059, 80: 0.1837, 100: 0.1707,
    # 70, 90 not in NIST table — log-log interpolate below
}

# NIST XCOM "Bone, Cortical (ICRU-44)" mass attenuation coefficients (cm²/g).
# Source: https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/bone.html
# These are the AUTHORITATIVE values fetched 2026-05-05.
# (Note: experiments/exp_log.md derived κ_2_high_init = 0.305/3.567 = 0.0855 using a
#  slightly different transcription; NIST gives 0.3148 → 0.0883. The 3% discrepancy
#  affects only the (20,60) winner init by an amount softplus learns away.)
MU_BONE_NIST = {
    10: 28.51, 15: 9.032, 20: 4.001, 30: 1.331, 40: 0.6655,
    50: 0.4242, 60: 0.3148, 80: 0.2229, 100: 0.1855,
    # 70, 90 not in NIST table — log-log interpolate below
}

# Empirical walnut-shell-to-water linear-attenuation ratio at low energy.
# Derived from (20, 60) baseline: shared_scale=3.567 / μ_water(20)=0.8096 = 4.41.
# Used only for shared_scale ESTIMATE; actual value comes from convert_walnut_dual_phys.
SHELL_TO_WATER_K = 4.41


def loglog_interp(table: dict, target_keV: int) -> float:
    """Log-log interpolate μ at target_keV from nearest two anchors in table.

    log(μ_E) = log(μ_lo) + t · (log(μ_hi) - log(μ_lo))
    where t = log(E/E_lo) / log(E_hi/E_lo).
    """
    keVs = sorted(table)
    if target_keV in table:
        return table[target_keV]
    # Find bracketing anchors
    lo = max((k for k in keVs if k < target_keV), default=None)
    hi = min((k for k in keVs if k > target_keV), default=None)
    if lo is None or hi is None:
        raise ValueError(f"keV={target_keV} outside NIST table range "
                         f"[{min(keVs)}, {max(keVs)}]")
    t = math.log(target_keV / lo) / math.log(hi / lo)
    log_mu = math.log(table[lo]) + t * (math.log(table[hi]) - math.log(table[lo]))
    return math.exp(log_mu)


def mu_water(keV: int) -> tuple[float, str]:
    """Returns (μ_water in cm²/g, source tag)."""
    if keV in MU_WATER_NIST:
        return MU_WATER_NIST[keV], "NIST"
    return loglog_interp(MU_WATER_NIST, keV), "log-interp"


def mu_bone(keV: int) -> tuple[float, str]:
    """Returns (μ_bone (cortical, ICRU-44) in cm²/g, source tag)."""
    if keV in MU_BONE_NIST:
        return MU_BONE_NIST[keV], "NIST"
    return loglog_interp(MU_BONE_NIST, keV), "log-interp"


def compute(low: int, high: int, shared_scale: float | None = None) -> dict:
    if low >= high:
        raise ValueError(f"low ({low}) must be < high ({high}) keV")
    mu_w_l, src_w_l = mu_water(low)
    mu_w_h, src_w_h = mu_water(high)
    mu_b_l, src_b_l = mu_bone(low)
    mu_b_h, src_b_h = mu_bone(high)

    if shared_scale is None:
        scale = SHELL_TO_WATER_K * mu_w_l
        scale_src = f"ESTIMATE = {SHELL_TO_WATER_K} × μ_water({low})"
    else:
        scale = shared_scale
        scale_src = "user-provided (from convert_walnut_dual_phys output)"

    return {
        "low": low, "high": high,
        "mu_w_low": mu_w_l, "mu_w_high": mu_w_h,
        "mu_b_low": mu_b_l, "mu_b_high": mu_b_h,
        "src_w_low": src_w_l, "src_w_high": src_w_h,
        "src_b_low": src_b_l, "src_b_high": src_b_h,
        "shared_scale": scale, "scale_src": scale_src,
        "kappa_w_low": 1.0,
        "kappa_w_high": mu_w_h / mu_w_l,
        "kappa_2_low_init": mu_b_l / scale,
        "kappa_2_high_init": mu_b_h / scale,
    }


def print_report(d: dict):
    low, high = d["low"], d["high"]
    print("=" * 72)
    print(f"Path D yaml calibration for pair (low={low} keV, high={high} keV)")
    print("=" * 72)
    print(f"  μ_water({low})  = {d['mu_w_low']:.4f} cm²/g  ({d['src_w_low']})")
    print(f"  μ_water({high})  = {d['mu_w_high']:.4f} cm²/g  ({d['src_w_high']})")
    print(f"  μ_bone({low})   = {d['mu_b_low']:.4f} cm²/g  ({d['src_b_low']})")
    print(f"  μ_bone({high})   = {d['mu_b_high']:.4f} cm²/g  ({d['src_b_high']})")
    print()
    print(f"  shared_scale  = {d['shared_scale']:.4f}  ({d['scale_src']})")
    print()
    print(f"  ─────────── yaml fields for v3_dual_phys_basis2 ───────────")
    print(f"  network:")
    print(f"    kappa_w_low:  {d['kappa_w_low']}")
    print(f"    kappa_w_high: {d['kappa_w_high']:.4f}        # μ_water({high}) / μ_water({low})")
    print(f"    kappa_2_low_init:  {d['kappa_2_low_init']:.4f}     # μ_bone({low}) / shared_scale")
    print(f"    kappa_2_high_init: {d['kappa_2_high_init']:.4f}     # μ_bone({high}) / shared_scale")
    print(f"    constrain_kappa_2: true       # softplus on κ_2")
    print(f"    last_activation: softplus")

    # Verify against winner for (20, 60)
    if low == 20 and high == 60:
        winner_kw = 0.2543
        winner_k2l = 1.121
        winner_k2h = 0.0855
        print()
        print(f"  [verify against (20,60) winner yaml]")
        print(f"    κ_w_high:     computed={d['kappa_w_high']:.4f}, "
              f"winner={winner_kw} → {'MATCH' if abs(d['kappa_w_high']-winner_kw)<0.001 else 'DIFF'}")
        print(f"    κ_2_low_init: computed={d['kappa_2_low_init']:.4f}, "
              f"winner={winner_k2l} → {'MATCH' if abs(d['kappa_2_low_init']-winner_k2l)<0.005 else 'DIFF'}")
        diff_h = abs(d['kappa_2_high_init'] - winner_k2h)
        print(f"    κ_2_high_init: computed={d['kappa_2_high_init']:.4f}, "
              f"winner={winner_k2h} → DIFF={diff_h:.4f} (~3%, NIST authoritative)")
    print("=" * 72)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--low", type=int, default=None)
    p.add_argument("--high", type=int, default=None)
    p.add_argument("--shared_scale", type=float, default=None)
    p.add_argument("--all", action="store_true",
                   help="run for baseline (20,60) + 4 new pairs")
    args = p.parse_args()

    if args.all:
        pairs = [(20, 60), (10, 80), (20, 70), (10, 70), (30, 80)]
        for low, high in pairs:
            print_report(compute(low, high))
            print()
    else:
        if args.low is None or args.high is None:
            raise SystemExit("must give --low/--high or --all")
        print_report(compute(args.low, args.high, args.shared_scale))


if __name__ == "__main__":
    main()
