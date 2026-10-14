#!/usr/bin/env python3
"""
Phase 4 v5c-b — out_* + Ring Frozen Re-calibration (HANDOFF v6.2)
==================================================================
v5c-b 설계: v5c(out_* 동결) 위에 ring_strength=0.009, ring_n=8.6 추가 동결.
유일 변인: ring 동결 여부. fwhm 이동 여부로 ring/fwhm 얽힘 재판정.

Tier 1 (동결): N=300, σ_g=0.0408, p=0.001
out_* (동결): out_low=0.10, out_high=0.90
Ring  (동결): ring_strength=0.009, ring_n=8.6  (v1/v2 합의)
Tier 2 (자유): fwhm, alpha, in_low, in_high

Seeds: 975000, 976000, 977000 (3 reps)

출력:
  phase4/theta_real_v5cb.json  — v5c/v5c-b 비교표 포함
  phase4/rep_v5cb_{0,1,2}/calib_trace.csv

Run:
    cd /mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        /mnt/gstore/home/xiangyue/HBM_paper/envs/hbm_cudnn/bin/python \\
        /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/run_phase4_v5cb.py \\
        2>&1 | tee /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/run_phase4_v5cb.log
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

# Tier 1 frozen
TIER1_N  = 300.0
TIER1_SG = 0.0408
TIER1_P  = 0.001

# out_* frozen (v6.2)
OUT_LOW_FROZEN  = 0.10
OUT_HIGH_FROZEN = 0.90

# Ring frozen (v1/v2 합의)
RING_FROZEN_STRENGTH = 0.009
RING_FROZEN_N        = 8.6

# Tier 2 constraints
IN_LOW_MIN,  IN_LOW_MAX  = 0.00, 0.30
IN_HIGH_MIN, IN_HIGH_MAX = 0.70, 1.00

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
print(f"Pre-normalization: x' = {PRENORM_A:.6f}·x + {PRENORM_B:.6f}")
print(f"Tier 1 FROZEN: N={TIER1_N}, σ_g={TIER1_SG}, p={TIER1_P}")
print(f"out_* FROZEN:  out_low={OUT_LOW_FROZEN}, out_high={OUT_HIGH_FROZEN}")
print(f"Ring  FROZEN:  ring_strength={RING_FROZEN_STRENGTH}, ring_n={RING_FROZEN_N}")
print(f"Tier 2 FREE:   fwhm, alpha, in_low, in_high")

THETA_MANUAL = dict(
    fwhm=2.5, N=TIER1_N, sigma_g=TIER1_SG, p=TIER1_P, alpha=0.35,
    in_low=0.05, in_high=0.90, out_low=OUT_LOW_FROZEN, out_high=OUT_HIGH_FROZEN,
    ring_strength=0.12, ring_n=8.0,
)

# v5c results for comparison
V5C_MEAN = dict(
    fwhm=1.3563, alpha=0.3171, in_low=0.1457, in_high=0.7000,
    ring_strength=0.0284, ring_n=7.9314,
)

PARAMS_TIER2 = ['fwhm', 'alpha', 'in_low', 'in_high']
MAIN_SEEDS = [975000, 976000, 977000]


# ── Helpers ───────────────────────────────────────────────────────────
def set_pipeline_v5cb(pipe: DegradationPipeline) -> None:
    with torch.no_grad():
        N_clamped = float(np.clip(TIER1_N, _N_LO + 1, _N_HI - 1))
        pipe.N_raw.fill_(sigmoid_box_inverse(N_clamped))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(TIER1_SG, 1e-4)))
        pipe.p.fill_(TIER1_P)
        pipe.out_low.fill_(OUT_LOW_FROZEN)
        pipe.out_high.fill_(OUT_HIGH_FROZEN)
        pipe.ring_strength_raw.fill_(softplus_inverse(RING_FROZEN_STRENGTH))
        pipe.ring_n_raw.fill_(softplus_inverse(RING_FROZEN_N))
        pipe.fwhm_raw.fill_(softplus_inverse(THETA_MANUAL['fwhm']))
        pipe.alpha_raw.fill_(softplus_inverse(THETA_MANUAL['alpha']))
        pipe.in_low.fill_(np.clip(THETA_MANUAL['in_low'], IN_LOW_MIN, IN_LOW_MAX))
        pipe.in_high.fill_(np.clip(THETA_MANUAL['in_high'], IN_HIGH_MIN, IN_HIGH_MAX))

    pipe.N_raw.requires_grad_(False)
    pipe.sigma_g_raw.requires_grad_(False)
    pipe.p.requires_grad_(False)
    pipe.out_low.requires_grad_(False)
    pipe.out_high.requires_grad_(False)
    pipe.ring_strength_raw.requires_grad_(False)
    pipe.ring_n_raw.requires_grad_(False)


def project_constraints_v5cb(pipe: DegradationPipeline) -> None:
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
print(f"  mapped: mean={np.mean([t.mean() for t in templates_np]):.4f}")

print(f"Loading {N_REAL} real (dirty2) images …")
real_paths = sorted(REAL_DIR.glob('*.png'))[:N_REAL]
real_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
           for p in real_paths]

print(f"\nBuilding CalibrationLoss …")
loss_fn = CalibrationLoss(real_np).to(DEVICE)
print("  Done.")


# ── Training loop ─────────────────────────────────────────────────────
def run_one_calibration_v5cb(pipe, loss_fn, templates, rep_seed, out_dir, run_label):
    set_pipeline_v5cb(pipe)
    _seed_all(rep_seed)

    tier2_raws   = [pipe.fwhm_raw, pipe.alpha_raw]
    tier2_direct = [pipe.in_low, pipe.in_high]
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
        project_constraints_v5cb(pipe)

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
print("Phase 4 v5c-b: out_*+Ring Frozen Re-calibration (HANDOFF v6.2)")
print(f"  Tier 1 frozen: N={TIER1_N}, σ_g={TIER1_SG}")
print(f"  out_* frozen:  out_low={OUT_LOW_FROZEN}, out_high={OUT_HIGH_FROZEN}")
print(f"  Ring  frozen:  ring_strength={RING_FROZEN_STRENGTH}, ring_n={RING_FROZEN_N}")
print(f"  Tier 2 free:   fwhm, alpha, in_low, in_high")
print(f"{'='*65}")

for rep, rep_seed in enumerate(MAIN_SEEDS):
    rep_dir = PHASE4_DIR / f'rep_v5cb_{rep}'
    print(f"\n[Rep {rep}] seed={rep_seed}  fwhm_init=2.5 (THETA_MANUAL)")
    theta_hat, _ = run_one_calibration_v5cb(
        pipe_opt, loss_fn, templates_t,
        rep_seed=rep_seed, out_dir=rep_dir,
        run_label=f"v5cb_r{rep}",
    )
    rep_results.append(theta_hat)
    print(f"  θ̂_v5cb rep {rep}: fwhm={theta_hat['fwhm']:.4f}  "
          f"alpha={theta_hat['alpha']:.4f}  "
          f"in_low={theta_hat['in_low']:.4f}  in_high={theta_hat['in_high']:.4f}")

print(f"\nAll reps done. Total: {(time.time()-t_total)/60:.1f} min")

# ── Aggregate ─────────────────────────────────────────────────────────
theta_mean = {p: float(np.mean([r[p] for r in rep_results])) for p in PARAMS_TIER2}
theta_std  = {p: float(np.std( [r[p] for r in rep_results])) for p in PARAMS_TIER2}
theta_mean.update({'N': TIER1_N, 'sigma_g': TIER1_SG, 'p': TIER1_P,
                   'out_low': OUT_LOW_FROZEN, 'out_high': OUT_HIGH_FROZEN,
                   'ring_strength': RING_FROZEN_STRENGTH, 'ring_n': RING_FROZEN_N})
theta_std.update({'N': 0.0, 'sigma_g': 0.0, 'p': 0.0,
                  'out_low': 0.0, 'out_high': 0.0,
                  'ring_strength': 0.0, 'ring_n': 0.0})

# ── v5c / v5c-b comparison ────────────────────────────────────────────
print(f"\n{'='*65}")
print("v5c vs v5c-b comparison (ring frozen effect on fwhm):")
print(f"{'Param':<16} {'θ_manual':>10} {'v5c_mean':>10} {'v5cb_mean':>10} {'Δ':>8}")
print('-'*56)
compare_params = ['fwhm', 'alpha', 'in_low', 'in_high']
for p in compare_params:
    man     = THETA_MANUAL.get(p, float('nan'))
    v5c_val = V5C_MEAN.get(p, float('nan'))
    vcb_val = theta_mean.get(p, float('nan'))
    delta   = vcb_val - v5c_val
    print(f"  {p:<14} {man:>10.4f} {v5c_val:>10.4f} {vcb_val:>10.4f} {delta:>+8.4f}")

fwhm_v5c  = V5C_MEAN['fwhm']
fwhm_v5cb = theta_mean['fwhm']
fwhm_shift_pct = 100 * (fwhm_v5cb - fwhm_v5c) / (fwhm_v5c + 1e-12)
large_shift = abs(fwhm_shift_pct) > 20.0

print(f"\n  fwhm shift v5c→v5c-b: {fwhm_v5c:.4f} → {fwhm_v5cb:.4f}  "
      f"(Δ={fwhm_v5cb-fwhm_v5c:+.4f}, {fwhm_shift_pct:+.1f}%)")
print(f"  fwhm std: {theta_std.get('fwhm',0):.4f}")
print(f"  Large shift (>20%): {'YES — ring/fwhm entangled' if large_shift else 'NO — fwhm robust'}")

# ── Save ──────────────────────────────────────────────────────────────
all_params_list = ['N', 'sigma_g', 'p', 'fwhm', 'alpha', 'in_low', 'in_high',
                   'out_low', 'out_high', 'ring_strength', 'ring_n']
theta_mean_full = {p: theta_mean.get(p, float('nan')) for p in all_params_list}
theta_std_full  = {p: theta_std.get(p, 0.0) for p in all_params_list}

comparison_table = []
for p in compare_params:
    comparison_table.append({
        'param':       p,
        'manual':      THETA_MANUAL.get(p, None),
        'v5c_mean':    V5C_MEAN.get(p, None),
        'v5cb_mean':   theta_mean.get(p, None),
        'v5cb_std':    theta_std.get(p, None),
        'delta_v5c_v5cb': float(theta_mean.get(p, 0.0) - V5C_MEAN.get(p, 0.0)),
    })

output = {
    "version": "v5cb",
    "purpose": "HANDOFF v6.2: v5c + ring 동결. 유일 변인 = ring 동결. fwhm 얽힘 재판정.",
    "handoff_ref": "HANDOFF_v6.2_out_param_ruling.md",
    "tier1_frozen":   {"N": TIER1_N, "sigma_g": TIER1_SG, "p": TIER1_P},
    "out_frozen":     {"out_low": OUT_LOW_FROZEN, "out_high": OUT_HIGH_FROZEN},
    "ring_frozen":    {"ring_strength": RING_FROZEN_STRENGTH, "ring_n": RING_FROZEN_N,
                       "source": "v1/v2 합의"},
    "tier2_free":     PARAMS_TIER2,
    "prenormalization": {"a": PRENORM_A, "b": PRENORM_B},
    "theta_real_mean": theta_mean_full,
    "theta_real_std":  theta_std_full,
    "theta_hat_reps":  rep_results,
    "v5c_reference":   V5C_MEAN,
    "comparison_table": comparison_table,
    "fwhm_shift_analysis": {
        "fwhm_v5c":        fwhm_v5c,
        "fwhm_v5cb":       fwhm_v5cb,
        "delta":           float(fwhm_v5cb - fwhm_v5c),
        "delta_pct":       float(fwhm_shift_pct),
        "large_shift_flag": bool(large_shift),
        "interpretation": (
            "fwhm shifted >20% when ring frozen → ring/fwhm entangled (v5c-b fwhm more reliable)."
            if large_shift else
            "fwhm stable (<20% shift) when ring frozen → fwhm robust to ring constraint."
        ),
    },
    "config": {
        "template_dir": str(TEMPLATE_DIR), "real_dir": str(REAL_DIR),
        "n_steps": N_STEPS, "batch_size": BATCH_SIZE,
        "n_reps": N_REPS, "seeds": MAIN_SEEDS,
        "lr_default": LR_DEFAULT, "device": DEVICE,
    }
}

out_path = PHASE4_DIR / 'theta_real_v5cb.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_path}")

print(f"\n{'='*65}")
print("Phase 4 v5c-b complete")
print(f"  Total: {(time.time()-t_total)/60:.1f} min")
print(f"  fwhm v5c={fwhm_v5c:.4f}  v5cb={fwhm_v5cb:.4f}  "
      f"shift={fwhm_v5cb-fwhm_v5c:+.4f} ({fwhm_shift_pct:+.1f}%)")
print(f"  → {'ring/fwhm ENTANGLED' if large_shift else 'fwhm INDEPENDENT (robust to ring)'}")
