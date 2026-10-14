#!/usr/bin/env python3
"""
Phase 4 v5d — fwhm ESF-보정값 동결 재캘리브레이션 (HANDOFF v6.3 §2)
=====================================================================
ESF 실측 보정값: fwhm_corrected = 2.057 × (2.500/2.454) = 2.095

v5c와 유일 변인 = fwhm 동결.

동결: Tier1 (N=300, σ_g=0.0408, p=0.001) + out_low=0.10 + out_high=0.90 + fwhm=2.095
자유: alpha, in_low, in_high, ring_strength, ring_n

Seeds: 980000, 981000, 982000 (3 reps × 2000 steps)

산출물:
  phase4/theta_real_v5d.json   — v5c↔v5d 대조표 포함
  phase4/rep_v5d_{0,1,2}/calib_trace.csv
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

# fwhm frozen: ESF 보정값 = 2.057 × (2.500 / 2.454) (HANDOFF v6.3 §1)
ESF_REAL_MEDIAN     = 2.057
ESF_SV_MEDIAN       = 2.454
ESF_APPLIED_FWHM    = 2.500
FWHM_FROZEN = ESF_REAL_MEDIAN * (ESF_APPLIED_FWHM / ESF_SV_MEDIAN)
print(f"fwhm_frozen = {ESF_REAL_MEDIAN:.4f} × ({ESF_APPLIED_FWHM:.3f}/{ESF_SV_MEDIAN:.4f}) = {FWHM_FROZEN:.4f}")
# → 2.0957 ≈ 2.095 (HANDOFF 명시값 확인)

# Tier 2 constraints
IN_LOW_MIN,  IN_LOW_MAX  = 0.00, 0.30
IN_HIGH_MIN, IN_HIGH_MAX = 0.70, 1.00

PARAMS_TIER2 = ['alpha', 'in_low', 'in_high', 'ring_strength', 'ring_n']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
print(f"Tier 1 FROZEN: N={TIER1_N}, σ_g={TIER1_SG}, p={TIER1_P}")
print(f"out_* FROZEN:  out_low={OUT_LOW_FROZEN}, out_high={OUT_HIGH_FROZEN}")
print(f"fwhm  FROZEN:  {FWHM_FROZEN:.4f}  (ESF 보정, v6.3 §1)")
print(f"Tier 2 FREE:   alpha, in_low, in_high, ring_strength, ring_n")

# v5c reference for comparison
V5C_MEAN = dict(
    fwhm=1.3563, alpha=0.3171, in_low=0.1457, in_high=0.7000,
    ring_strength=0.0284, ring_n=7.9314,
)

MAIN_SEEDS = [980000, 981000, 982000]

# ── Helpers ───────────────────────────────────────────────────────────
def set_pipeline_v5d(pipe: DegradationPipeline) -> None:
    with torch.no_grad():
        N_clamped = float(np.clip(TIER1_N, _N_LO + 1, _N_HI - 1))
        pipe.N_raw.fill_(sigmoid_box_inverse(N_clamped))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(TIER1_SG, 1e-4)))
        pipe.p.fill_(TIER1_P)
        pipe.out_low.fill_(OUT_LOW_FROZEN)
        pipe.out_high.fill_(OUT_HIGH_FROZEN)
        pipe.fwhm_raw.fill_(softplus_inverse(FWHM_FROZEN))
        # free params: init from theta_manual
        pipe.alpha_raw.fill_(softplus_inverse(0.35))
        pipe.in_low.fill_(0.05)
        pipe.in_high.fill_(0.90)
        pipe.ring_strength_raw.fill_(softplus_inverse(0.12))
        pipe.ring_n_raw.fill_(softplus_inverse(8.0))

    pipe.N_raw.requires_grad_(False)
    pipe.sigma_g_raw.requires_grad_(False)
    pipe.p.requires_grad_(False)
    pipe.out_low.requires_grad_(False)
    pipe.out_high.requires_grad_(False)
    pipe.fwhm_raw.requires_grad_(False)   # KEY: fwhm frozen at 2.095


def project_constraints_v5d(pipe: DegradationPipeline) -> None:
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
print(f"  mapped: mean={np.mean([t.mean().item() for t in templates_t]):.4f}")

print(f"Loading {N_REAL} real (dirty2) images …")
real_paths = sorted(REAL_DIR.glob('*.png'))[:N_REAL]
real_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
           for p in real_paths]

print(f"\nBuilding CalibrationLoss …")
loss_fn = CalibrationLoss(real_np).to(DEVICE)
print("  Done.")


# ── Training loop ─────────────────────────────────────────────────────
def run_one_calibration_v5d(pipe, loss_fn, templates, rep_seed, out_dir, run_label):
    set_pipeline_v5d(pipe)
    _seed_all(rep_seed)

    tier2_raws   = [pipe.alpha_raw, pipe.ring_strength_raw, pipe.ring_n_raw]
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
        project_constraints_v5d(pipe)

        loss_val = (batch_loss / BATCH_SIZE).item()
        phys = get_theta_physical(pipe)
        row = dict(step=step, L_total=loss_val, elapsed=time.time()-t0)
        row.update({k: v/BATCH_SIZE for k, v in batch_parts.items()})
        row.update(phys)
        trace.append(row)

        if (step+1) % 200 == 0:
            print(f"  [{run_label}] step {step+1:4d}  L={loss_val:.5f}  "
                  f"alpha={phys['alpha']:.4f}  in_low={phys['in_low']:.4f}  "
                  f"in_high={phys['in_high']:.4f}  ring_s={phys['ring_strength']:.4f}  ring_n={phys['ring_n']:.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'calib_trace.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(trace[0].keys()))
        writer.writeheader()
        writer.writerows(trace)
    print(f"  [{run_label}] done: {N_STEPS} steps  elapsed={time.time()-t0:.0f}s  "
          f"L_final={trace[-1]['L_total']:.5f}")
    return get_theta_physical(pipe), trace


# ── Main ──────────────────────────────────────────────────────────────
pipe_opt = DegradationPipeline().to(DEVICE)
rep_results = []
rep_final_losses = []
t_total = time.time()

print(f"\n{'='*65}")
print("Phase 4 v5d: fwhm=2.095 동결 재캘리브레이션 (HANDOFF v6.3 §2)")
print(f"  fwhm frozen: {FWHM_FROZEN:.4f}  (ESF 보정)")
print(f"  Tier 2 free: alpha, in_low, in_high, ring_strength, ring_n")
print(f"{'='*65}")

for rep, rep_seed in enumerate(MAIN_SEEDS):
    rep_dir = PHASE4_DIR / f'rep_v5d_{rep}'
    print(f"\n[Rep {rep}] seed={rep_seed}")
    theta_hat, trace = run_one_calibration_v5d(
        pipe_opt, loss_fn, templates_t,
        rep_seed=rep_seed, out_dir=rep_dir,
        run_label=f"v5d_r{rep}",
    )
    rep_results.append(theta_hat)
    rep_final_losses.append(trace[-1]['L_total'])
    print(f"  θ̂_v5d rep {rep}: alpha={theta_hat['alpha']:.4f}  "
          f"in_low={theta_hat['in_low']:.4f}  in_high={theta_hat['in_high']:.4f}  "
          f"ring_s={theta_hat['ring_strength']:.4f}  ring_n={theta_hat['ring_n']:.3f}")

print(f"\nAll reps done. Total: {(time.time()-t_total)/60:.1f} min")

# ── Aggregate ─────────────────────────────────────────────────────────
theta_mean = {p: float(np.mean([r[p] for r in rep_results])) for p in PARAMS_TIER2}
theta_std  = {p: float(np.std( [r[p] for r in rep_results])) for p in PARAMS_TIER2}
theta_mean.update({'N': TIER1_N, 'sigma_g': TIER1_SG, 'p': TIER1_P,
                   'fwhm': FWHM_FROZEN,
                   'out_low': OUT_LOW_FROZEN, 'out_high': OUT_HIGH_FROZEN})
theta_std.update({'N': 0.0, 'sigma_g': 0.0, 'p': 0.0,
                  'fwhm': 0.0, 'out_low': 0.0, 'out_high': 0.0})

# ── §3 판정표 평가 ────────────────────────────────────────────────────
BOUND_DEFS = {
    'in_low':  (IN_LOW_MIN,  IN_LOW_MAX),
    'in_high': (IN_HIGH_MIN, IN_HIGH_MAX),
}

def is_boundary_stuck(val, lo, hi, tol=0.05):
    margin = tol * (hi - lo)
    return (val - lo) < margin or (hi - val) < margin

def is_rep_inconsistent(std, mean, thresh=0.05):
    if abs(mean) < 1e-9:
        return False
    return (std / abs(mean)) >= thresh

print(f"\n{'='*65}")
print("§3 판정표 평가 (HANDOFF v6.3):")
row1_flag = False
row1_details = []
for p in ['alpha', 'in_low', 'in_high']:
    mu = theta_mean[p]; sd = theta_std[p]
    rep_bad = is_rep_inconsistent(sd, mu)
    if p in BOUND_DEFS:
        lo, hi = BOUND_DEFS[p]
        stuck  = is_boundary_stuck(mu, lo, hi)
    else:
        stuck = False
    if stuck or rep_bad:
        row1_flag = True
        row1_details.append(f"{p}={mu:.4f}±{sd:.4f} (stuck={stuck}, rep_cv={sd/abs(mu):.3f})")
    print(f"  {p}: mean={mu:.4f}  std={sd:.4f}  cv={sd/abs(mu):.3f}  "
          f"stuck={'YES' if stuck else 'no'}  rep_ok={'NO' if rep_bad else 'yes'}")

ring_s = theta_mean['ring_strength']
row2_flag = ring_s > 0.03
print(f"  ring_strength: {ring_s:.4f}  {'> 0.03 → 행 2 발동' if row2_flag else '≤ 0.03 OK'}")

if row1_flag:
    section3_row = 1
    section3_status = "PENDING_CONSULTATION"
    print(f"\n  → §3 행 1: {row1_details}  PENDING_CONSULTATION")
elif row2_flag:
    section3_row = 2
    section3_status = "PENDING_CONSULTATION"
    print(f"\n  → §3 행 2: ring_strength={ring_s:.4f} > 0.03  PENDING_CONSULTATION")
else:
    section3_row = 3
    section3_status = "FINAL"
    print(f"\n  → §3 행 3: 전원 내부점 + rep 일치 + ring ≤ 0.03  → FINAL 사전 승인")
    print(f"     approved_by: HANDOFF_v6.3 §3 행3")

# ── v5c vs v5d comparison ─────────────────────────────────────────────
print(f"\n{'='*65}")
print("v5c vs v5d 대조표 (fwhm 동결 효과):")
print(f"{'Param':<16} {'v5c_mean':>10} {'v5d_mean':>10} {'Δ':>8}")
print('-'*46)
compare_params = ['alpha', 'in_low', 'in_high', 'ring_strength', 'ring_n']
for p in compare_params:
    v5c_val = V5C_MEAN.get(p, float('nan'))
    v5d_val = theta_mean.get(p, float('nan'))
    delta   = v5d_val - v5c_val
    print(f"  {p:<14} {v5c_val:>10.4f} {v5d_val:>10.4f} {delta:>+8.4f}")
print(f"  {'fwhm':<14} {'1.3563':>10} {FWHM_FROZEN:>10.4f}  (동결)")

loss_v5c_ref = 0.04013  # v5c rep mean final loss (from log)
loss_v5d = float(np.mean(rep_final_losses))
loss_pct  = 100 * (loss_v5d - loss_v5c_ref) / loss_v5c_ref
print(f"\n  최종 loss: v5c≈{loss_v5c_ref:.5f}  v5d={loss_v5d:.5f}  "
      f"악화={loss_v5d-loss_v5c_ref:+.5f} ({loss_pct:+.1f}%)")
print(f"  (fwhm을 노이즈 최적값에서 떼어냈으므로 악화는 정상)")

# ── Build D_calib if FINAL ────────────────────────────────────────────
d_calib_theta = None
if section3_status == "FINAL":
    d_calib_theta = {
        "N": TIER1_N, "sigma_g": TIER1_SG, "p": TIER1_P,
        "fwhm": FWHM_FROZEN,
        "alpha": theta_mean['alpha'],
        "in_low": theta_mean['in_low'],
        "in_high": theta_mean['in_high'],
        "out_low": OUT_LOW_FROZEN, "out_high": OUT_HIGH_FROZEN,
        "ring_strength": theta_mean['ring_strength'],
        "ring_n": theta_mean['ring_n'],
    }
    print(f"\n  D_calib θ̂:")
    for k, v in d_calib_theta.items():
        print(f"    {k}: {v}")

# ── Save ──────────────────────────────────────────────────────────────
all_params_list = ['N', 'sigma_g', 'p', 'fwhm', 'alpha', 'in_low', 'in_high',
                   'out_low', 'out_high', 'ring_strength', 'ring_n']
theta_mean_full = {p: theta_mean.get(p, float('nan')) for p in all_params_list}
theta_std_full  = {p: theta_std.get(p, 0.0) for p in all_params_list}

comparison_table = []
for p in compare_params:
    comparison_table.append({
        'param':     p,
        'v5c_mean':  V5C_MEAN.get(p),
        'v5d_mean':  theta_mean.get(p),
        'v5d_std':   theta_std.get(p),
        'delta':     float(theta_mean.get(p, 0.0) - V5C_MEAN.get(p, 0.0)),
    })

output = {
    "version": "v5d",
    "purpose": "HANDOFF v6.3 §2: fwhm=ESF보정값(2.095) 동결 → D_calib 최종 산출",
    "handoff_ref": "HANDOFF_v6.3_esf_ruling.md",
    "esf_correction": {
        "esf_real_median":  ESF_REAL_MEDIAN,
        "esf_sv_median":    ESF_SV_MEDIAN,
        "esf_applied_fwhm": ESF_APPLIED_FWHM,
        "fwhm_corrected":   FWHM_FROZEN,
        "formula": "fwhm_corrected = esf_real_median × (applied_fwhm / sv_median)",
    },
    "tier1_frozen":   {"N": TIER1_N, "sigma_g": TIER1_SG, "p": TIER1_P},
    "out_frozen":     {"out_low": OUT_LOW_FROZEN, "out_high": OUT_HIGH_FROZEN},
    "fwhm_frozen":    FWHM_FROZEN,
    "tier2_free":     PARAMS_TIER2,
    "prenormalization": {"a": PRENORM_A, "b": PRENORM_B},
    "theta_real_mean": theta_mean_full,
    "theta_real_std":  theta_std_full,
    "theta_hat_reps":  rep_results,
    "rep_final_losses": rep_final_losses,
    "v5c_reference":   V5C_MEAN,
    "comparison_table": comparison_table,
    "loss_comparison": {
        "v5c_ref_loss": loss_v5c_ref,
        "v5d_mean_loss": loss_v5d,
        "delta_abs": float(loss_v5d - loss_v5c_ref),
        "delta_pct": float(loss_pct),
        "note": "fwhm을 노이즈 최적값(1.356)에서 ESF값(2.095)으로 이동 → loss 악화는 정상",
    },
    "section3_ruling": {
        "row": section3_row,
        "status": section3_status,
        "row1_details": row1_details,
        "ring_strength_mean": ring_s,
        "ring_gt_0p03": row2_flag,
        "approved_by": "HANDOFF_v6.3 §3 행3" if section3_status == "FINAL" else None,
    },
    "d_calib_theta": d_calib_theta,
    "config": {
        "template_dir": str(TEMPLATE_DIR), "real_dir": str(REAL_DIR),
        "n_steps": N_STEPS, "batch_size": BATCH_SIZE,
        "n_reps": N_REPS, "seeds": MAIN_SEEDS,
        "lr_default": LR_DEFAULT, "device": DEVICE,
    }
}

out_path = PHASE4_DIR / 'theta_real_v5d.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_path}")

print(f"\n{'='*65}")
print(f"Phase 4 v5d complete  ({(time.time()-t_total)/60:.1f} min)")
print(f"  §3 판정: 행 {section3_row} → {section3_status}")
if section3_status == "FINAL":
    print(f"  D_calib FINAL 사전 승인 (approved_by: HANDOFF_v6.3 §3 행3)")
    print(f"  → P4.2 TOST 즉시 착수 가능")
