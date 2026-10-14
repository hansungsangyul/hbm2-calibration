#!/usr/bin/env python3
"""
Phase 4 v5e — σ_post 도입 재캘리브레이션 (HANDOFF v6.4 §1.3)
=============================================================
v5d와 유일 변인 = σ_post 자유.

동결: Tier1 (N=300, σ_g=0.0408, p=0.001) + out_low=0.10 + out_high=0.90 + fwhm=2.0956
자유: alpha, in_low, in_high, σ_post, ring_strength, ring_n

Seeds: 990000, 991000, 992000 (3 reps × 2000 steps)

사전 예측 (v6.3 §1 진단 검증):
  - σ_post 도입 시 ring ≤ 0.03 복귀
  - in_high 내부점 이탈 (고착 탈출)
  - alpha 정상 범위 회복

산출물:
  phase4/theta_real_v5e.json   — v5d↔v5e 대조표, §2 판정 포함
  phase4/rep_v5e_{0,1,2}/calib_trace.csv
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
LR_DEFAULT  = 0.01

PRENORM_A = 0.501834
PRENORM_B = 0.173320

TIER1_N  = 300.0
TIER1_SG = 0.0408
TIER1_P  = 0.001
OUT_LOW_FROZEN  = 0.10
OUT_HIGH_FROZEN = 0.90
FWHM_FROZEN     = 2.057 * (2.500 / 2.454)   # = 2.0956

# σ_post bounds
SIGMA_POST_MIN = 1e-4
SIGMA_POST_MAX = 0.1

# Tier 2 bounds
IN_LOW_MIN,  IN_LOW_MAX  = 0.00, 0.30
IN_HIGH_MIN, IN_HIGH_MAX = 0.70, 1.00

PARAMS_TIER2 = ['alpha', 'in_low', 'in_high', 'sigma_post', 'ring_strength', 'ring_n']

# v5d reference
V5D_MEAN = dict(
    alpha=0.0564, in_low=0.2702, in_high=0.7000,
    sigma_post=0.0, ring_strength=0.1247, ring_n=7.949,
)
V5D_LOSS = 0.20670

MAIN_SEEDS = [990000, 991000, 992000]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
print(f"fwhm frozen: {FWHM_FROZEN:.4f}")
print(f"Tier 2 free: alpha, in_low, in_high, sigma_post, ring_strength, ring_n")
print(f"사전 예측: ring≤0.03, in_high 내부점, alpha 정상 범위")


# ── Helpers ───────────────────────────────────────────────────────────
def set_pipeline_v5e(pipe: DegradationPipeline) -> None:
    with torch.no_grad():
        N_c = float(np.clip(TIER1_N, _N_LO+1, _N_HI-1))
        pipe.N_raw.fill_(sigmoid_box_inverse(N_c))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(TIER1_SG, 1e-4)))
        pipe.p.fill_(TIER1_P)
        pipe.out_low.fill_(OUT_LOW_FROZEN)
        pipe.out_high.fill_(OUT_HIGH_FROZEN)
        pipe.fwhm_raw.fill_(softplus_inverse(FWHM_FROZEN))
        # free params: init from theta_manual / midpoint
        pipe.alpha_raw.fill_(softplus_inverse(0.35))
        pipe.in_low.fill_(0.05)
        pipe.in_high.fill_(0.90)
        pipe.sigma_post.fill_(0.01)
        pipe.ring_strength_raw.fill_(softplus_inverse(0.12))
        pipe.ring_n_raw.fill_(softplus_inverse(8.0))

    pipe.N_raw.requires_grad_(False)
    pipe.sigma_g_raw.requires_grad_(False)
    pipe.p.requires_grad_(False)
    pipe.out_low.requires_grad_(False)
    pipe.out_high.requires_grad_(False)
    pipe.fwhm_raw.requires_grad_(False)


def project_constraints_v5e(pipe: DegradationPipeline) -> None:
    with torch.no_grad():
        pipe.in_low.clamp_(IN_LOW_MIN, IN_LOW_MAX)
        pipe.in_high.clamp_(IN_HIGH_MIN, IN_HIGH_MAX)
        pipe.sigma_post.clamp_(SIGMA_POST_MIN, SIGMA_POST_MAX)


def get_theta_physical(pipe: DegradationPipeline) -> dict:
    with torch.no_grad():
        return dict(
            N=float(_N_LO + (_N_HI - _N_LO) * torch.sigmoid(pipe.N_raw)),
            sigma_g=float(F.softplus(pipe.sigma_g_raw)),
            p=float(pipe.p),
            fwhm=float(F.softplus(pipe.fwhm_raw)),
            alpha=float(F.softplus(pipe.alpha_raw)),
            in_low=float(pipe.in_low),
            in_high=float(pipe.in_high),
            out_low=float(pipe.out_low),
            out_high=float(pipe.out_high),
            sigma_post=float(pipe.sigma_post),
            ring_strength=float(F.softplus(pipe.ring_strength_raw)),
            ring_n=float(F.softplus(pipe.ring_n_raw)),
        )


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Load images ───────────────────────────────────────────────────────
print(f"\nLoading {N_TEMPLATES} templates …")
template_paths = sorted(TEMPLATE_DIR.glob('*.png'))[:N_TEMPLATES]
templates_raw = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
                 for p in template_paths]
H, W = templates_raw[0].shape
templates_t = [torch.from_numpy(PRENORM_A * t + PRENORM_B).to(DEVICE) for t in templates_raw]
print(f"  H={H}, W={W}")

print(f"Loading {N_REAL} real images …")
real_paths = sorted(REAL_DIR.glob('*.png'))[:N_REAL]
real_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
           for p in real_paths]

print("Building CalibrationLoss …")
loss_fn = CalibrationLoss(real_np).to(DEVICE)
print("  Done.")


# ── Training loop ─────────────────────────────────────────────────────
def run_one_calibration_v5e(pipe, loss_fn, templates, rep_seed, out_dir, run_label):
    set_pipeline_v5e(pipe)
    _seed_all(rep_seed)

    tier2_raws   = [pipe.alpha_raw, pipe.ring_strength_raw, pipe.ring_n_raw]
    tier2_direct = [pipe.in_low, pipe.in_high, pipe.sigma_post]
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
                'eps1':     torch.randn(H, W, device=DEVICE),
                'eps2':     torch.randn(H, W, device=DEVICE),
                'eps3':     torch.rand(H, W, device=DEVICE),
                'eps_post': torch.randn(H, W, device=DEVICE),
            }
            syn = pipe(x, eps)
            L, parts = loss_fn(syn)
            batch_loss = batch_loss + L
            for k in batch_parts:
                batch_parts[k] += parts[k].item()

        (batch_loss / BATCH_SIZE).backward()
        optimizer.step()
        project_constraints_v5e(pipe)

        loss_val = (batch_loss / BATCH_SIZE).item()
        phys = get_theta_physical(pipe)
        row = dict(step=step, L_total=loss_val, elapsed=time.time()-t0)
        row.update({k: v/BATCH_SIZE for k, v in batch_parts.items()})
        row.update(phys)
        trace.append(row)

        if (step+1) % 200 == 0:
            print(f"  [{run_label}] step {step+1:4d}  L={loss_val:.5f}  "
                  f"alpha={phys['alpha']:.4f}  in_low={phys['in_low']:.4f}  "
                  f"in_high={phys['in_high']:.4f}  σ_post={phys['sigma_post']:.4f}  "
                  f"ring_s={phys['ring_strength']:.4f}")

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
print("Phase 4 v5e: σ_post 도입 재캘리브레이션 (HANDOFF v6.4 §1.3)")
print(f"  fwhm frozen: {FWHM_FROZEN:.4f}")
print(f"  Tier 2 free: alpha, in_low, in_high, σ_post, ring_strength, ring_n")
print(f"{'='*65}")

for rep, rep_seed in enumerate(MAIN_SEEDS):
    rep_dir = PHASE4_DIR / f'rep_v5e_{rep}'
    print(f"\n[Rep {rep}] seed={rep_seed}")
    theta_hat, trace = run_one_calibration_v5e(
        pipe_opt, loss_fn, templates_t,
        rep_seed=rep_seed, out_dir=rep_dir,
        run_label=f"v5e_r{rep}",
    )
    rep_results.append(theta_hat)
    rep_final_losses.append(trace[-1]['L_total'])
    print(f"  θ̂_v5e rep {rep}: alpha={theta_hat['alpha']:.4f}  "
          f"in_low={theta_hat['in_low']:.4f}  in_high={theta_hat['in_high']:.4f}  "
          f"σ_post={theta_hat['sigma_post']:.4f}  ring_s={theta_hat['ring_strength']:.4f}")

print(f"\nAll reps done. Total: {(time.time()-t_total)/60:.1f} min")

# ── Aggregate ─────────────────────────────────────────────────────────
theta_mean = {p: float(np.mean([r[p] for r in rep_results])) for p in PARAMS_TIER2}
theta_std  = {p: float(np.std( [r[p] for r in rep_results])) for p in PARAMS_TIER2}
theta_mean.update({'N': TIER1_N, 'sigma_g': TIER1_SG, 'p': TIER1_P,
                   'fwhm': FWHM_FROZEN, 'out_low': OUT_LOW_FROZEN, 'out_high': OUT_HIGH_FROZEN})
theta_std.update({'N': 0.0, 'sigma_g': 0.0, 'p': 0.0,
                  'fwhm': 0.0, 'out_low': 0.0, 'out_high': 0.0})

# ── §2 판정표 평가 ────────────────────────────────────────────────────
BOUND_DEFS = {
    'in_low':     (IN_LOW_MIN,  IN_LOW_MAX),
    'in_high':    (IN_HIGH_MIN, IN_HIGH_MAX),
    'sigma_post': (SIGMA_POST_MIN, SIGMA_POST_MAX),
}

def is_boundary_stuck(val, lo, hi, tol=0.05):
    margin = tol * (hi - lo)
    return (val - lo) < margin or (hi - val) < margin

def is_rep_inconsistent(std, mean, thresh=0.05):
    if abs(mean) < 1e-9:
        return False
    return (std / abs(mean)) >= thresh

print(f"\n{'='*65}")
print("§2 판정표 평가 (HANDOFF v6.4):")

sigma_post_mean = theta_mean['sigma_post']
row1_flag = sigma_post_mean > 0.08
print(f"  σ_post mean: {sigma_post_mean:.4f}  {'> 0.08 → 행 1 발동' if row1_flag else '≤ 0.08 OK'}")

row2_flag = False
row2_details = []
ring_s = theta_mean['ring_strength']
if not row1_flag:
    for p in ['alpha', 'in_low', 'in_high', 'sigma_post']:
        mu = theta_mean[p]; sd = theta_std[p]
        rep_bad = is_rep_inconsistent(sd, mu)
        if p in BOUND_DEFS:
            lo, hi = BOUND_DEFS[p]
            stuck = is_boundary_stuck(mu, lo, hi)
        else:
            stuck = False
        if stuck or rep_bad:
            row2_flag = True
            row2_details.append(f"{p}={mu:.4f}±{sd:.4f} (stuck={stuck}, cv={sd/max(abs(mu),1e-9):.3f})")
        print(f"  {p}: mean={mu:.4f}  std={sd:.4f}  cv={sd/max(abs(mu),1e-9):.3f}  "
              f"stuck={'YES' if stuck else 'no'}  rep_ok={'NO' if rep_bad else 'yes'}")
    ring_ok = ring_s <= 0.03
    print(f"  ring_strength: {ring_s:.4f}  {'≤ 0.03 OK' if ring_ok else '> 0.03 FAIL'}")
    if not ring_ok:
        row2_flag = True
        row2_details.append(f"ring_strength={ring_s:.4f} > 0.03")

if row1_flag:
    section2_row = 1
    section2_status = "PENDING_CONSULTATION"
elif row2_flag:
    section2_row = 3
    section2_status = "FINAL"
    approved_by = "HANDOFF_v6.4 §2 행3-fallback"
else:
    section2_row = 2
    section2_status = "FINAL"
    approved_by = "HANDOFF_v6.4 §2 행2"

print(f"\n  → §2 행 {section2_row}: {section2_status}")
if section2_status == "FINAL":
    print(f"     approved_by: {approved_by}")
    print(f"     → P4.2 TOST 즉시 착수")

# ── 사전 예측 적중 여부 ───────────────────────────────────────────────
print(f"\n사전 예측 적중 여부 (v6.3 §1 진단 최종 검증):")
pred_ring = ring_s <= 0.03
pred_in_high = not is_boundary_stuck(theta_mean['in_high'], IN_HIGH_MIN, IN_HIGH_MAX)
pred_alpha = theta_mean['alpha'] > 0.10
print(f"  ring ≤ 0.03:          {ring_s:.4f} → {'✓ 적중' if pred_ring else '✗ 미적중'}")
print(f"  in_high 내부점 이탈:   {theta_mean['in_high']:.4f} → {'✓ 적중' if pred_in_high else '✗ 미적중'}")
print(f"  alpha 정상 범위 (>0.1): {theta_mean['alpha']:.4f} → {'✓ 적중' if pred_alpha else '✗ 미적중'}")

# ── v5d vs v5e 대조표 ────────────────────────────────────────────────
print(f"\nv5d vs v5e 대조표:")
print(f"{'Param':<16} {'v5d_mean':>10} {'v5e_mean':>10} {'Δ':>8}")
print('-'*48)
compare_params = ['alpha', 'in_low', 'in_high', 'sigma_post', 'ring_strength', 'ring_n']
for p in compare_params:
    v5d_val = V5D_MEAN.get(p, 0.0)
    v5e_val = theta_mean.get(p, 0.0)
    print(f"  {p:<14} {v5d_val:>10.4f} {v5e_val:>10.4f} {v5e_val-v5d_val:>+8.4f}")

loss_v5e = float(np.mean(rep_final_losses))
print(f"  loss:           {V5D_LOSS:>10.5f} {loss_v5e:>10.5f} {loss_v5e-V5D_LOSS:>+8.5f}")

# ── D_calib 구성 ─────────────────────────────────────────────────────
d_calib_theta = None
if section2_status == "FINAL":
    if section2_row == 2:
        d_calib_theta = {
            "N": TIER1_N, "sigma_g": TIER1_SG, "p": TIER1_P,
            "fwhm": FWHM_FROZEN,
            "alpha": theta_mean['alpha'],
            "in_low": theta_mean['in_low'],
            "in_high": theta_mean['in_high'],
            "out_low": OUT_LOW_FROZEN, "out_high": OUT_HIGH_FROZEN,
            "sigma_post": theta_mean['sigma_post'],
            "ring_strength": theta_mean['ring_strength'],
            "ring_n": theta_mean['ring_n'],
        }
    else:  # row 3 fallback: use manual for alpha/in_low/in_high/ring, sigma_post=0
        d_calib_theta = {
            "N": TIER1_N, "sigma_g": TIER1_SG, "p": TIER1_P,
            "fwhm": FWHM_FROZEN,
            "alpha": 0.35, "in_low": 0.05, "in_high": 0.90,
            "out_low": OUT_LOW_FROZEN, "out_high": OUT_HIGH_FROZEN,
            "sigma_post": 0.0,
            "ring_strength": 0.12, "ring_n": 8.0,
        }
    print(f"\n  D_calib θ̂ (행 {section2_row}):")
    for k, v in d_calib_theta.items():
        print(f"    {k}: {v}")

# ── Save ──────────────────────────────────────────────────────────────
all_params = ['N', 'sigma_g', 'p', 'fwhm', 'alpha', 'in_low', 'in_high',
              'out_low', 'out_high', 'sigma_post', 'ring_strength', 'ring_n']
theta_mean_full = {p: theta_mean.get(p, float('nan')) for p in all_params}
theta_std_full  = {p: theta_std.get(p, 0.0) for p in all_params}

output = {
    "version": "v5e",
    "purpose": "HANDOFF v6.4 §1.3: σ_post 도입 재캘리브레이션 → D_calib 최종",
    "handoff_ref": "HANDOFF_v6.4_sigma_post_ruling.md",
    "fwhm_frozen": FWHM_FROZEN,
    "tier1_frozen": {"N": TIER1_N, "sigma_g": TIER1_SG, "p": TIER1_P},
    "out_frozen": {"out_low": OUT_LOW_FROZEN, "out_high": OUT_HIGH_FROZEN},
    "tier2_free": PARAMS_TIER2,
    "prenormalization": {"a": PRENORM_A, "b": PRENORM_B},
    "theta_real_mean": theta_mean_full,
    "theta_real_std": theta_std_full,
    "theta_hat_reps": rep_results,
    "rep_final_losses": rep_final_losses,
    "v5d_reference": V5D_MEAN,
    "comparison_table": [{
        "param": p,
        "v5d_mean": V5D_MEAN.get(p, 0.0),
        "v5e_mean": theta_mean.get(p),
        "v5e_std": theta_std.get(p),
        "delta": float(theta_mean.get(p, 0.0) - V5D_MEAN.get(p, 0.0)),
    } for p in compare_params],
    "loss_comparison": {
        "v5d_loss": V5D_LOSS,
        "v5e_loss": loss_v5e,
        "delta_abs": float(loss_v5e - V5D_LOSS),
        "delta_pct": float(100*(loss_v5e-V5D_LOSS)/V5D_LOSS),
    },
    "prediction_check": {
        "ring_le_003": {"predicted": True, "actual": ring_s, "hit": pred_ring},
        "in_high_internal": {"predicted": True, "actual": theta_mean['in_high'], "hit": pred_in_high},
        "alpha_normal": {"predicted": True, "actual": theta_mean['alpha'], "hit": pred_alpha},
    },
    "section2_ruling": {
        "row": section2_row,
        "status": section2_status,
        "sigma_post_mean": sigma_post_mean,
        "ring_strength_mean": ring_s,
        "row2_details": row2_details,
        "approved_by": approved_by if section2_status == "FINAL" else None,
    },
    "d_calib_theta": d_calib_theta,
    "config": {
        "template_dir": str(TEMPLATE_DIR), "real_dir": str(REAL_DIR),
        "n_steps": N_STEPS, "batch_size": BATCH_SIZE,
        "n_reps": N_REPS, "seeds": MAIN_SEEDS,
        "lr_default": LR_DEFAULT, "device": DEVICE,
    }
}

out_path = PHASE4_DIR / 'theta_real_v5e.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_path}")

print(f"\n{'='*65}")
print(f"Phase 4 v5e complete  ({(time.time()-t_total)/60:.1f} min)")
print(f"  §2 판정: 행 {section2_row} → {section2_status}")
if section2_status == "FINAL":
    print(f"  approved_by: {approved_by}")
    print(f"  → P4.2 TOST 착수 가능")
