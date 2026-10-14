#!/usr/bin/env python3
"""
Phase 4 v5b — Ring-Frozen Re-calibration (지시 24 / U-2)
=========================================================
v5 결과: ring_strength=0.1012 (v1/v2 합의 0.009의 11.2×) → D_calib REVOKED.
v5b 설계: ring_strength=0.009 / ring_n=8.6 를 동결 추가, fwhm이 이동하는지 검증.

Tier 1 (동결): N=300, σ_g=0.0408, p=0.001
Ring  (동결): ring_strength=0.009, ring_n=8.6   ← 추가 (v1/v2 합의값)
Tier 2 (자유): fwhm, alpha, in_low, in_high      ← ring 제외

Seeds: 955000, 956000, 957000 (no perturbed anchor runs)

출력:
  phase4/theta_real_v5b.json  — v5/v5b 비교표 포함
  phase4/rep_v5b_0/calib_trace.csv  (etc.)

NOTE: D_calib 재발행 없음 — 결과는 §6 판정표 입력으로만 사용됨.

Run:
    cd /mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation
    PYTHONUNBUFFERED=1 /mnt/gstore/home/xiangyue/HBM_paper/envs/hbm_cudnn/bin/python \\
        /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/run_phase4_v5b.py \\
        2>&1 | tee /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/run_phase4_v5b.log
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
sys.path.insert(0, str(ROOT / 'src'))

import torch
import torch.nn.functional as F
from pipeline import DegradationPipeline, softplus_inverse, sigmoid_box_inverse, _N_LO, _N_HI
from losses import CalibrationLoss

# ── Config ────────────────────────────────────────────────────────────
TEMPLATE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_clean')
REAL_DIR     = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2')
PHASE4_DIR   = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4')

N_TEMPLATES = 50
N_REAL      = 50
N_STEPS     = 2000
BATCH_SIZE  = 8
N_REPS      = 3

LR_DEFAULT = 0.01

PRENORM_A = 0.501834
PRENORM_B = 0.173320

# Tier 1 frozen (전작 실측)
TIER1_N  = 300.0
TIER1_SG = 0.0408
TIER1_P  = 0.001

# Ring frozen to v1/v2 consensus (HANDOFF v6.1 §2 / 지시 24)
RING_FROZEN_STRENGTH = 0.009
RING_FROZEN_N        = 8.6

# Tier 2 constraints
IN_LOW_MIN,  IN_LOW_MAX  = 0.00, 0.30
IN_HIGH_MIN, IN_HIGH_MAX = 0.70, 1.00

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
print(f"Pre-normalization: x' = {PRENORM_A:.6f}·x + {PRENORM_B:.6f}")
print(f"Tier 1 FROZEN: N={TIER1_N}, σ_g={TIER1_SG}, p={TIER1_P}")
print(f"Ring  FROZEN: ring_strength={RING_FROZEN_STRENGTH}, ring_n={RING_FROZEN_N}  (v1/v2 합의)")
print(f"Tier 2 FREE: fwhm, alpha, in_low, in_high")

THETA_MANUAL = dict(
    fwhm=2.5, N=TIER1_N, sigma_g=TIER1_SG, p=TIER1_P, alpha=0.35,
    in_low=0.05, in_high=0.90, out_low=0.10, out_high=0.90,
    ring_strength=0.12, ring_n=8.0,
)

# v5 results for comparison table
V5_MEAN = dict(
    fwhm=1.4248, alpha=0.4973, in_low=0.0039, in_high=0.7021,
    ring_strength=0.1012, ring_n=7.987,
)

# Tier 2 params (ring excluded)
PARAMS_TIER2 = ['fwhm', 'alpha', 'in_low', 'in_high']


# ── Helpers ───────────────────────────────────────────────────────────
def set_pipeline_v5b(pipe: DegradationPipeline) -> None:
    """Set pipeline to v5b init: Tier 1 + ring frozen, Tier 2 free."""
    with torch.no_grad():
        # Tier 1: frozen
        N_clamped = float(np.clip(TIER1_N, _N_LO + 1, _N_HI - 1))
        pipe.N_raw.fill_(sigmoid_box_inverse(N_clamped))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(TIER1_SG, 1e-4)))
        pipe.p.fill_(TIER1_P)
        # Ring: frozen to v1/v2 consensus
        pipe.ring_strength_raw.fill_(softplus_inverse(RING_FROZEN_STRENGTH))
        pipe.ring_n_raw.fill_(softplus_inverse(RING_FROZEN_N))
        # Tier 2: free, initialized from THETA_MANUAL
        pipe.fwhm_raw.fill_(softplus_inverse(THETA_MANUAL['fwhm']))
        pipe.alpha_raw.fill_(softplus_inverse(THETA_MANUAL['alpha']))
        pipe.in_low.fill_(np.clip(THETA_MANUAL['in_low'], IN_LOW_MIN, IN_LOW_MAX))
        pipe.in_high.fill_(np.clip(THETA_MANUAL['in_high'], IN_HIGH_MIN, IN_HIGH_MAX))
        pipe.out_low.fill_(THETA_MANUAL['out_low'])
        pipe.out_high.fill_(THETA_MANUAL['out_high'])

    # Freeze Tier 1 + ring
    pipe.N_raw.requires_grad_(False)
    pipe.sigma_g_raw.requires_grad_(False)
    pipe.p.requires_grad_(False)
    pipe.ring_strength_raw.requires_grad_(False)
    pipe.ring_n_raw.requires_grad_(False)


def project_constraints_v5b(pipe: DegradationPipeline) -> None:
    with torch.no_grad():
        pipe.in_low.clamp_(IN_LOW_MIN, IN_LOW_MAX)
        pipe.in_high.clamp_(IN_HIGH_MIN, IN_HIGH_MAX)


def get_theta_physical(pipe: DegradationPipeline) -> dict:
    with torch.no_grad():
        return dict(
            N=float(_N_LO + (_N_HI - _N_LO) * torch.sigmoid(pipe.N_raw)),
            sigma_g=float(F.softplus(pipe.sigma_g_raw).item()),
            p=float(pipe.p.item()),
            fwhm=float(F.softplus(pipe.fwhm_raw).item()),
            alpha=float(F.softplus(pipe.alpha_raw).item()),
            in_low=float(pipe.in_low.item()),
            in_high=float(pipe.in_high.item()),
            out_low=float(pipe.out_low.item()),
            out_high=float(pipe.out_high.item()),
            ring_strength=float(F.softplus(pipe.ring_strength_raw).item()),
            ring_n=float(F.softplus(pipe.ring_n_raw).item()),
        )


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Load images ───────────────────────────────────────────────────────
print(f"\nLoading {N_TEMPLATES} clean templates …")
template_paths = sorted(TEMPLATE_DIR.glob('*.png'))[:N_TEMPLATES]
templates_np_raw = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
                    for p in template_paths]
H, W = templates_np_raw[0].shape

templates_np = [PRENORM_A * img + PRENORM_B for img in templates_np_raw]
templates_t  = [torch.from_numpy(t).to(DEVICE) for t in templates_np]
print(f"  raw:    mean={np.mean([t.mean() for t in templates_np_raw]):.4f}")
print(f"  mapped: mean={np.mean([t.mean() for t in templates_np]):.4f}")

print(f"Loading {N_REAL} real (dirty2) images …")
real_paths = sorted(REAL_DIR.glob('*.png'))[:N_REAL]
real_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
           for p in real_paths]
print(f"  real stats: mean={np.mean([i.mean() for i in real_np]):.4f}  "
      f"std={np.mean([i.std() for i in real_np]):.4f}")

# ── CalibrationLoss ───────────────────────────────────────────────────
print(f"\nBuilding CalibrationLoss from {N_REAL} real images …")
loss_fn = CalibrationLoss(real_np).to(DEVICE)
print("  Done.")


# ── Training loop ─────────────────────────────────────────────────────
def run_one_calibration_v5b(pipe, loss_fn, templates, rep_seed, out_dir, run_label):
    set_pipeline_v5b(pipe)
    _seed_all(rep_seed)

    # Only Tier 2 params in optimizer (ring excluded)
    tier2_raws   = [pipe.fwhm_raw, pipe.alpha_raw]
    tier2_direct = [pipe.in_low, pipe.in_high, pipe.out_low, pipe.out_high]
    optimizer = torch.optim.Adam(
        [{'params': tier2_raws,   'lr': LR_DEFAULT},
         {'params': tier2_direct, 'lr': LR_DEFAULT}]
    )

    trace = []
    t0 = time.time()

    for step in range(N_STEPS):
        optimizer.zero_grad()
        x = templates[step % len(templates)]
        batch_loss = torch.tensor(0.0, device=DEVICE)
        batch_parts = {'L_vm': 0.0, 'L_nps': 0.0, 'L_rad': 0.0, 'L_hist': 0.0}

        for _ in range(BATCH_SIZE):
            eps = {
                'eps1': torch.randn(H, W, device=DEVICE),
                'eps2': torch.randn(H, W, device=DEVICE),
                'eps3': torch.rand(H, W, device=DEVICE),
            }
            syn = pipe(x, eps)
            L, parts = loss_fn(syn)
            batch_loss = batch_loss + L
            for k in batch_parts:
                batch_parts[k] += parts[k].item()

        (batch_loss / BATCH_SIZE).backward()
        optimizer.step()
        project_constraints_v5b(pipe)

        loss_val = (batch_loss / BATCH_SIZE).item()
        phys = get_theta_physical(pipe)
        row = dict(step=step, L_total=loss_val, elapsed=time.time()-t0)
        row.update({k: v/BATCH_SIZE for k, v in batch_parts.items()})
        row.update(phys)
        trace.append(row)

        if (step+1) % 200 == 0:
            print(f"  [{run_label}] step {step+1:4d}  L={loss_val:.5f}  "
                  f"fwhm={phys['fwhm']:.3f}  alpha={phys['alpha']:.3f}  "
                  f"in_low={phys['in_low']:.4f}  in_high={phys['in_high']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'calib_trace.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
        writer.writeheader()
        writer.writerows(trace)
    print(f"  [{run_label}] done: {N_STEPS} steps  elapsed={time.time()-t0:.0f}s")
    return get_theta_physical(pipe), trace


# ── Main ──────────────────────────────────────────────────────────────
pipe_opt = DegradationPipeline().to(DEVICE)
rep_results = []
t_total = time.time()

print(f"\n{'='*65}")
print("Phase 4 v5b: Ring-Frozen Re-calibration (지시 24 / U-2)")
print(f"  Tier 1 frozen: N={TIER1_N}, σ_g={TIER1_SG}")
print(f"  Ring  frozen: ring_strength={RING_FROZEN_STRENGTH}, ring_n={RING_FROZEN_N}")
print(f"  Tier 2 free: fwhm, alpha, in_low, in_high")
print(f"{'='*65}")

MAIN_SEEDS = [955000, 956000, 957000]
for rep, rep_seed in enumerate(MAIN_SEEDS):
    rep_dir = PHASE4_DIR / f'rep_v5b_{rep}'
    print(f"\n[Rep {rep}] seed={rep_seed}  fwhm_init=2.5 (THETA_MANUAL)")
    theta_hat, trace = run_one_calibration_v5b(
        pipe_opt, loss_fn, templates_t,
        rep_seed=rep_seed,
        out_dir=rep_dir,
        run_label=f"v5b_r{rep}",
    )
    rep_results.append(theta_hat)
    print(f"  θ̂_v5b rep {rep}: fwhm={theta_hat['fwhm']:.4f}  alpha={theta_hat['alpha']:.4f}  "
          f"in_low={theta_hat['in_low']:.4f}  in_high={theta_hat['in_high']:.4f}")

print(f"\nAll reps done. Total: {(time.time()-t_total)/60:.1f} min")

# ── Aggregate ─────────────────────────────────────────────────────────
theta_mean = {p: float(np.mean([r[p] for r in rep_results])) for p in PARAMS_TIER2}
theta_std  = {p: float(np.std( [r[p] for r in rep_results])) for p in PARAMS_TIER2}
theta_mean['N']           = TIER1_N
theta_mean['sigma_g']     = TIER1_SG
theta_mean['p']           = TIER1_P
theta_mean['ring_strength'] = RING_FROZEN_STRENGTH
theta_mean['ring_n']        = RING_FROZEN_N
theta_std['N']            = 0.0
theta_std['sigma_g']      = 0.0
theta_std['p']            = 0.0
theta_std['ring_strength'] = 0.0
theta_std['ring_n']        = 0.0

# ── v5 / v5b comparison ───────────────────────────────────────────────
print(f"\n{'='*65}")
print("v5 vs v5b comparison (ring frozen effect on fwhm):")
print(f"{'Param':<16} {'θ_manual':>10} {'v5_mean':>10} {'v5b_mean':>10} {'Δ':>8}")
print('-'*56)
compare_params = ['fwhm', 'alpha', 'in_low', 'in_high', 'ring_strength', 'ring_n']
v5b_full_mean = theta_mean.copy()
v5b_full_mean.update({p: theta_mean.get(p, float('nan')) for p in compare_params})

for p in compare_params:
    man    = THETA_MANUAL.get(p, float('nan'))
    v5_val = V5_MEAN.get(p, float('nan'))
    v5b_val = theta_mean.get(p, float('nan'))
    delta  = v5b_val - v5_val
    print(f"  {p:<14} {man:>10.4f} {v5_val:>10.4f} {v5b_val:>10.4f} {delta:>+8.4f}")

fwhm_v5   = V5_MEAN['fwhm']
fwhm_v5b  = theta_mean['fwhm']
fwhm_shift = abs(fwhm_v5b - fwhm_v5)
fwhm_shift_pct = 100 * fwhm_shift / (fwhm_v5 + 1e-12)
print(f"\n  fwhm shift v5→v5b: {fwhm_v5:.4f} → {fwhm_v5b:.4f}  (Δ={fwhm_v5b-fwhm_v5:+.4f}, {fwhm_shift_pct:.1f}%)")

# Branch for §6 judgment
fwhm_v5b_interior = 0.3 <= fwhm_v5b <= 8.0
fwhm_v5b_std_pct  = 100 * theta_std.get('fwhm', 0.0) / (fwhm_v5b + 1e-12)
large_shift = fwhm_shift_pct > 20.0  # >20% shift = ring/fwhm entangled
near_manual = abs(fwhm_v5b - 2.5) < 0.3  # within 12% of 2.5

print(f"\n  v5b fwhm interior: {'YES' if fwhm_v5b_interior else 'NO'}")
print(f"  v5b rep std: {theta_std.get('fwhm',0.0):.4f} ({fwhm_v5b_std_pct:.1f}%)")
print(f"  fwhm shift > 20%: {'YES (ring/fwhm entangled)' if large_shift else 'NO (fwhm stable)'}")
print(f"  v5b fwhm near manual (2.5±0.3): {'YES' if near_manual else 'NO'}")

# ── Save ──────────────────────────────────────────────────────────────
all_params_list = ['N', 'sigma_g', 'p', 'fwhm', 'alpha', 'in_low', 'in_high',
                   'ring_strength', 'ring_n']
theta_mean_full = {p: theta_mean.get(p, float('nan')) for p in all_params_list}
theta_std_full  = {p: theta_std.get(p, 0.0) for p in all_params_list}

comparison_table = []
for p in compare_params:
    comparison_table.append({
        'param':    p,
        'manual':   THETA_MANUAL.get(p, None),
        'v5_mean':  V5_MEAN.get(p, None),
        'v5b_mean': theta_mean.get(p, None),
        'v5b_std':  theta_std.get(p, None),
        'delta_v5_v5b': float(theta_mean.get(p, 0.0) - V5_MEAN.get(p, 0.0)),
    })

output = {
    "version": "v5b",
    "purpose": "지시 24 (U-2): ring_strength/ring_n 동결 후 fwhm 이동 검증. §6 판정표 입력.",
    "handoff_ref": "HANDOFF_v6.1 §2 / 지시 24",
    "tier1_frozen": {"N": TIER1_N, "sigma_g": TIER1_SG, "p": TIER1_P},
    "ring_frozen":  {"ring_strength": RING_FROZEN_STRENGTH, "ring_n": RING_FROZEN_N,
                     "source": "v1/v2 합의"},
    "tier2_free":   PARAMS_TIER2,
    "prenormalization": {"a": PRENORM_A, "b": PRENORM_B},
    "theta_real_mean": theta_mean_full,
    "theta_real_std":  theta_std_full,
    "theta_hat_reps":  rep_results,
    "theta_manual":    THETA_MANUAL,
    "v5_reference":    V5_MEAN,
    "comparison_table": comparison_table,
    "fwhm_shift_analysis": {
        "fwhm_v5":         fwhm_v5,
        "fwhm_v5b":        fwhm_v5b,
        "delta":           float(fwhm_v5b - fwhm_v5),
        "delta_pct":       float(fwhm_v5b - fwhm_v5) / (fwhm_v5 + 1e-12) * 100,
        "shift_abs_pct":   float(fwhm_shift_pct),
        "large_shift_flag": bool(large_shift),
        "near_manual_flag": bool(near_manual),
        "v5b_interior":    bool(fwhm_v5b_interior),
        "v5b_std_pct":     float(fwhm_v5b_std_pct),
        "interpretation": (
            "fwhm shifted >20% when ring frozen → ring/fwhm entangled; v5b fwhm more reliable."
            if large_shift else
            "fwhm stable (<20% shift) when ring frozen → fwhm estimate robust to ring constraint."
        ),
    },
    "d_calib_note": "v5b does NOT re-issue d_calib_config.json. §6 judgment table applies.",
    "config": {
        "template_dir": str(TEMPLATE_DIR), "real_dir": str(REAL_DIR),
        "n_steps": N_STEPS, "batch_size": BATCH_SIZE,
        "n_reps": N_REPS, "seeds": MAIN_SEEDS,
        "lr_default": LR_DEFAULT, "device": DEVICE,
    }
}

out_path = PHASE4_DIR / 'theta_real_v5b.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_path}")

print(f"\n{'='*65}")
print("Phase 4 v5b complete")
print(f"  Total: {(time.time()-t_total)/60:.1f} min")
print(f"  fwhm v5={fwhm_v5:.4f}  v5b={fwhm_v5b:.4f}  shift={fwhm_v5b-fwhm_v5:+.4f} ({fwhm_shift_pct:.1f}%)")
print(f"  → {'Large shift: ring/fwhm entangled' if large_shift else 'Stable: fwhm robust to ring constraint'}")
print(f"\nNext: apply §6 judgment table (지시 23/24/25 완료 후)")
