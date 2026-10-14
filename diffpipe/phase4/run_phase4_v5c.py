#!/usr/bin/env python3
"""
Phase 4 v5c — out_* Frozen Re-calibration (HANDOFF v6.2)
=========================================================
v6.2 판정: out_low/out_high를 캘리브레이션 대상에서 영구 제외 (0.10/0.90 동결).
v5의 fwhm=1.4248은 out_*(0.25/1.41)과 결합 피팅됐으므로 재실행 필요.

Tier 1 (동결): N=300, σ_g=0.0408, p=0.001
out_* (동결): out_low=0.10, out_high=0.90   ← v6.2 추가
Tier 2 (자유): fwhm, alpha, in_low, in_high, ring_strength, ring_n

Seeds:
  주 3 rep:     970000, 971000, 972000
  perturbed:    973000 (fwhm_init=1.5), 974000 (fwhm_init=3.5)

출력:
  phase4/theta_real_v5c.json  — v5/v5c 비교표 포함
  phase4/rep_v5c_{0,1,2,p0,p1}/calib_trace.csv

Run:
    cd /mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        /mnt/gstore/home/xiangyue/HBM_paper/envs/hbm_cudnn/bin/python \\
        /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/run_phase4_v5c.py \\
        2>&1 | tee /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/run_phase4_v5c.log
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

# out_* frozen (v6.2 판정: 영구 제외)
OUT_LOW_FROZEN  = 0.10
OUT_HIGH_FROZEN = 0.90

# Tier 2 constraints
IN_LOW_MIN,  IN_LOW_MAX  = 0.00, 0.30
IN_HIGH_MIN, IN_HIGH_MAX = 0.70, 1.00

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
print(f"Pre-normalization: x' = {PRENORM_A:.6f}·x + {PRENORM_B:.6f}")
print(f"Tier 1 FROZEN: N={TIER1_N}, σ_g={TIER1_SG}, p={TIER1_P}")
print(f"out_* FROZEN: out_low={OUT_LOW_FROZEN}, out_high={OUT_HIGH_FROZEN}  (v6.2 판정)")
print(f"Tier 2 FREE: fwhm, alpha, in_low, in_high, ring_strength, ring_n")

THETA_MANUAL = dict(
    fwhm=2.5, N=TIER1_N, sigma_g=TIER1_SG, p=TIER1_P, alpha=0.35,
    in_low=0.05, in_high=0.90, out_low=OUT_LOW_FROZEN, out_high=OUT_HIGH_FROZEN,
    ring_strength=0.12, ring_n=8.0,
)

# v5 results for comparison (out_* 미동결 버전 — deprecated)
V5_MEAN = dict(
    fwhm=1.4248, alpha=0.4973, in_low=0.0039, in_high=0.7021,
    out_low=0.2515, out_high=1.4133,
    ring_strength=0.1012, ring_n=7.987,
)

# Tier 2 params (out_* 제외, ring 포함)
PARAMS_TIER2 = ['fwhm', 'alpha', 'in_low', 'in_high', 'ring_strength', 'ring_n']

# Run configurations: (seed, fwhm_init, label, rep_dir_name)
MAIN_SEEDS = [
    (970000, 2.5,  'v5c_r0', 'rep_v5c_0'),
    (971000, 2.5,  'v5c_r1', 'rep_v5c_1'),
    (972000, 2.5,  'v5c_r2', 'rep_v5c_2'),
    (973000, 1.5,  'v5c_p0', 'rep_v5c_p0'),  # perturbed fwhm_init=1.5
    (974000, 3.5,  'v5c_p1', 'rep_v5c_p1'),  # perturbed fwhm_init=3.5
]


# ── Helpers ───────────────────────────────────────────────────────────
def set_pipeline_v5c(pipe: DegradationPipeline, fwhm_init: float = 2.5) -> None:
    """Set pipeline to v5c init: Tier 1 + out_* frozen, ring and others free."""
    with torch.no_grad():
        # Tier 1: frozen
        N_clamped = float(np.clip(TIER1_N, _N_LO + 1, _N_HI - 1))
        pipe.N_raw.fill_(sigmoid_box_inverse(N_clamped))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(TIER1_SG, 1e-4)))
        pipe.p.fill_(TIER1_P)
        # out_*: frozen to manual
        pipe.out_low.fill_(OUT_LOW_FROZEN)
        pipe.out_high.fill_(OUT_HIGH_FROZEN)
        # Tier 2: free, initialized from THETA_MANUAL (fwhm_init may differ)
        pipe.fwhm_raw.fill_(softplus_inverse(max(fwhm_init, 1e-4)))
        pipe.alpha_raw.fill_(softplus_inverse(THETA_MANUAL['alpha']))
        pipe.in_low.fill_(np.clip(THETA_MANUAL['in_low'], IN_LOW_MIN, IN_LOW_MAX))
        pipe.in_high.fill_(np.clip(THETA_MANUAL['in_high'], IN_HIGH_MIN, IN_HIGH_MAX))
        pipe.ring_strength_raw.fill_(softplus_inverse(THETA_MANUAL['ring_strength']))
        pipe.ring_n_raw.fill_(softplus_inverse(THETA_MANUAL['ring_n']))

    # Freeze Tier 1 + out_*
    pipe.N_raw.requires_grad_(False)
    pipe.sigma_g_raw.requires_grad_(False)
    pipe.p.requires_grad_(False)
    pipe.out_low.requires_grad_(False)
    pipe.out_high.requires_grad_(False)


def project_constraints_v5c(pipe: DegradationPipeline) -> None:
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
def run_one_calibration_v5c(pipe, loss_fn, templates, rep_seed, fwhm_init,
                             out_dir, run_label):
    set_pipeline_v5c(pipe, fwhm_init=fwhm_init)
    _seed_all(rep_seed)

    # Optimizer: Tier 2 only (out_* excluded)
    tier2_raws   = [pipe.fwhm_raw, pipe.alpha_raw, pipe.ring_strength_raw, pipe.ring_n_raw]
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
        project_constraints_v5c(pipe)

        loss_val = (batch_loss / BATCH_SIZE).item()
        phys = get_theta_physical(pipe)
        row = dict(step=step, L_total=loss_val, elapsed=time.time()-t0)
        row.update({k: v/BATCH_SIZE for k, v in batch_parts.items()})
        row.update(phys)
        trace.append(row)

        if (step+1) % 200 == 0:
            print(f"  [{run_label}] step {step+1:4d}  L={loss_val:.5f}  "
                  f"fwhm={phys['fwhm']:.3f}  alpha={phys['alpha']:.3f}  "
                  f"ring_strength={phys['ring_strength']:.4f}  "
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
all_results = []  # list of (label, fwhm_init, theta_hat)
t_total = time.time()

print(f"\n{'='*65}")
print("Phase 4 v5c: out_*-Frozen Re-calibration (HANDOFF v6.2)")
print(f"  Tier 1 frozen: N={TIER1_N}, σ_g={TIER1_SG}")
print(f"  out_* frozen:  out_low={OUT_LOW_FROZEN}, out_high={OUT_HIGH_FROZEN}")
print(f"  Tier 2 free:   fwhm, alpha, in_low, in_high, ring_strength, ring_n")
print(f"{'='*65}")

rep_results = []  # main 3 reps only for mean/std
perturb_results = []

for run_idx, (seed, fwhm_init, label, dir_name) in enumerate(MAIN_SEEDS):
    rep_dir = PHASE4_DIR / dir_name
    is_main = run_idx < N_REPS
    print(f"\n[{'Rep' if is_main else 'Perturbed'} {run_idx if is_main else run_idx - N_REPS}]"
          f"  seed={seed}  fwhm_init={fwhm_init}")
    theta_hat, trace = run_one_calibration_v5c(
        pipe_opt, loss_fn, templates_t,
        rep_seed=seed, fwhm_init=fwhm_init,
        out_dir=rep_dir, run_label=label,
    )
    all_results.append((label, fwhm_init, theta_hat))
    if is_main:
        rep_results.append(theta_hat)
    else:
        perturb_results.append(theta_hat)
    print(f"  θ̂_v5c {label}: fwhm={theta_hat['fwhm']:.4f}  "
          f"ring_strength={theta_hat['ring_strength']:.4f}  "
          f"alpha={theta_hat['alpha']:.4f}  "
          f"in_low={theta_hat['in_low']:.4f}  in_high={theta_hat['in_high']:.4f}")

print(f"\nAll runs done. Total: {(time.time()-t_total)/60:.1f} min")

# ── Aggregate (주 3 rep) ─────────────────────────────────────────────
theta_mean = {p: float(np.mean([r[p] for r in rep_results])) for p in PARAMS_TIER2}
theta_std  = {p: float(np.std( [r[p] for r in rep_results])) for p in PARAMS_TIER2}
theta_mean.update({'N': TIER1_N, 'sigma_g': TIER1_SG, 'p': TIER1_P,
                   'out_low': OUT_LOW_FROZEN, 'out_high': OUT_HIGH_FROZEN})
theta_std.update({'N': 0.0, 'sigma_g': 0.0, 'p': 0.0,
                  'out_low': 0.0, 'out_high': 0.0})

# ── v5 / v5c comparison ───────────────────────────────────────────────
print(f"\n{'='*65}")
print("v5 vs v5c comparison (out_* frozen effect):")
print(f"{'Param':<16} {'θ_manual':>10} {'v5_mean':>10} {'v5c_mean':>10} {'Δ':>8}")
print('-'*56)
compare_params = ['fwhm', 'alpha', 'in_low', 'in_high', 'ring_strength', 'ring_n']
for p in compare_params:
    man    = THETA_MANUAL.get(p, float('nan'))
    v5_val = V5_MEAN.get(p, float('nan'))
    v5c_val = theta_mean.get(p, float('nan'))
    delta  = v5c_val - v5_val
    print(f"  {p:<14} {man:>10.4f} {v5_val:>10.4f} {v5c_val:>10.4f} {delta:>+8.4f}")

fwhm_v5  = V5_MEAN['fwhm']
fwhm_v5c = theta_mean['fwhm']
fwhm_shift_pct = 100 * (fwhm_v5c - fwhm_v5) / (fwhm_v5 + 1e-12)
print(f"\n  fwhm shift v5→v5c: {fwhm_v5:.4f} → {fwhm_v5c:.4f}  "
      f"(Δ={fwhm_v5c-fwhm_v5:+.4f}, {fwhm_shift_pct:+.1f}%)")
print(f"  fwhm std: {theta_std.get('fwhm',0):.4f}")

# Perturbed anchor check
if perturb_results:
    print(f"\n  Perturbed init results:")
    for (lbl, fi, r), pr in zip(
            [(l, f, t) for l, f, t in all_results if l.startswith('v5c_p')],
            perturb_results):
        print(f"    {lbl} (fwhm_init={fi}): fwhm={r['fwhm']:.4f}  "
              f"ring_strength={r['ring_strength']:.4f}")

# ── Save ──────────────────────────────────────────────────────────────
all_params_list = ['N', 'sigma_g', 'p', 'fwhm', 'alpha', 'in_low', 'in_high',
                   'out_low', 'out_high', 'ring_strength', 'ring_n']
theta_mean_full = {p: theta_mean.get(p, float('nan')) for p in all_params_list}
theta_std_full  = {p: theta_std.get(p, 0.0) for p in all_params_list}

comparison_table = []
for p in compare_params:
    comparison_table.append({
        'param':      p,
        'manual':     THETA_MANUAL.get(p, None),
        'v5_mean':    V5_MEAN.get(p, None),
        'v5c_mean':   theta_mean.get(p, None),
        'v5c_std':    theta_std.get(p, None),
        'delta_v5_v5c': float(theta_mean.get(p, 0.0) - V5_MEAN.get(p, 0.0)),
    })

output = {
    "version": "v5c",
    "purpose": "HANDOFF v6.2: out_low/out_high 동결 후 v5 재실행. 유일 변인 = out_* 동결.",
    "handoff_ref": "HANDOFF_v6.2_out_param_ruling.md",
    "tier1_frozen":   {"N": TIER1_N, "sigma_g": TIER1_SG, "p": TIER1_P},
    "out_frozen":     {"out_low": OUT_LOW_FROZEN, "out_high": OUT_HIGH_FROZEN,
                       "source": "v6.2 판정 (영구 제외)", "manual_value": True},
    "tier2_free":     PARAMS_TIER2,
    "prenormalization": {"a": PRENORM_A, "b": PRENORM_B},
    "theta_real_mean": theta_mean_full,
    "theta_real_std":  theta_std_full,
    "theta_hat_reps":  rep_results,
    "theta_hat_perturbed": [
        {"label": l, "fwhm_init": fi, "theta": t}
        for l, fi, t in all_results if l.startswith('v5c_p')
    ],
    "theta_manual":    THETA_MANUAL,
    "v5_reference":    V5_MEAN,
    "comparison_table": comparison_table,
    "fwhm_shift_v5_v5c": {
        "fwhm_v5":       fwhm_v5,
        "fwhm_v5c":      fwhm_v5c,
        "delta":         float(fwhm_v5c - fwhm_v5),
        "delta_pct":     float(fwhm_shift_pct),
    },
    "config": {
        "template_dir": str(TEMPLATE_DIR), "real_dir": str(REAL_DIR),
        "n_steps": N_STEPS, "batch_size": BATCH_SIZE,
        "n_reps": N_REPS, "seeds": MAIN_SEEDS,
        "lr_default": LR_DEFAULT, "device": DEVICE,
    }
}

out_path = PHASE4_DIR / 'theta_real_v5c.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_path}")

print(f"\n{'='*65}")
print("Phase 4 v5c complete")
print(f"  Total: {(time.time()-t_total)/60:.1f} min")
print(f"  fwhm v5={fwhm_v5:.4f}  v5c={fwhm_v5c:.4f}  shift={fwhm_v5c-fwhm_v5:+.4f} ({fwhm_shift_pct:+.1f}%)")
print(f"  ring_strength v5c mean: {theta_mean.get('ring_strength', float('nan')):.4f}")
print(f"\nNext: run v5c-b (run_phase4_v5cb.py) with ring frozen to v5c ring value")
