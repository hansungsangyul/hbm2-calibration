#!/usr/bin/env python3
"""
Phase 4 v6-free — 전 파라미터 자유 최적화 재실행 (HANDOFF v7.6 §2, 지시 68)
==========================================================================
v1 프로토콜 + 유일 차이 = 파이프라인에 σ_post 항 존재.

전부 자유: N, σ_g, fwhm, alpha, in_low, in_high, out_low, out_high,
           ring_strength, ring_n, σ_post
동결: p=0.001 (최적화기 미포함 — v1과 동일, 출력=0.001)

무제약: project_constraints 미적용

Seeds: 910000/911000/912000 (v2 동일 시드)
N_STEPS: 2000

사전 판정표 (HANDOFF v7.6 §2):
  (i)  (N,σ_g,f)가 v1처럼 비물리 조합으로 방황 + reps 간 상이 수렴
       → RIDGE_PERSISTS  (§3.3 유지·강화)
  (ii) 실측·ESF 근방으로 안정 수렴 (σ_g≤0.042, fwhm∈[2.0±0.2])
       → RIDGE_DISSOLVED = 즉시 보고 (원고 구조 변경 사안)

v1 비교 기준: N≈278.75, σ_g≈0.192, fwhm≈1.753
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
LR_N        = 1.0

PRENORM_A = 0.501834
PRENORM_B = 0.173320

THETA_MANUAL = dict(
    N=300.0, sigma_g=0.0408, p=0.001, fwhm=2.5, alpha=0.35,
    in_low=0.05, in_high=0.90, out_low=0.10, out_high=0.90,
    ring_strength=0.12, ring_n=8.0, sigma_post=0.01,
)

V1_RESULT = dict(
    N=278.754, sigma_g=0.19181, fwhm=1.75287,
    alpha=0.54672, in_low=-0.05789, in_high=1.05618,
    ring_strength=0.00896, ring_n=7.72855, p=0.001,
)

MAIN_SEEDS = [910000, 911000, 912000]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
print("v6-free: 전 파라미터 자유, 무제약, sigma_post 포함")
print(f"Seeds: {MAIN_SEEDS}")


# ── Helpers ───────────────────────────────────────────────────────────
def set_pipeline_v6free(pipe: DegradationPipeline) -> None:
    with torch.no_grad():
        N_c = float(np.clip(THETA_MANUAL['N'], _N_LO+1, _N_HI-1))
        pipe.N_raw.fill_(sigmoid_box_inverse(N_c))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(THETA_MANUAL['sigma_g'], 1e-4)))
        pipe.p.fill_(THETA_MANUAL['p'])
        pipe.fwhm_raw.fill_(softplus_inverse(THETA_MANUAL['fwhm']))
        pipe.alpha_raw.fill_(softplus_inverse(THETA_MANUAL['alpha']))
        pipe.in_low.fill_(THETA_MANUAL['in_low'])
        pipe.in_high.fill_(THETA_MANUAL['in_high'])
        pipe.out_low.fill_(THETA_MANUAL['out_low'])
        pipe.out_high.fill_(THETA_MANUAL['out_high'])
        pipe.ring_strength_raw.fill_(softplus_inverse(THETA_MANUAL['ring_strength']))
        pipe.ring_n_raw.fill_(softplus_inverse(THETA_MANUAL['ring_n']))
        pipe.sigma_post.fill_(THETA_MANUAL['sigma_post'])
    # p stays frozen at 0.001 (not passed to optimizer — v1과 동일)
    pipe.p.requires_grad_(False)


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
def run_one_calibration_v6free(pipe, loss_fn, templates, rep_seed, out_dir, run_label):
    set_pipeline_v6free(pipe)
    _seed_all(rep_seed)

    # All params free except p
    raw_params = [pipe.N_raw, pipe.sigma_g_raw, pipe.fwhm_raw,
                  pipe.alpha_raw, pipe.ring_strength_raw, pipe.ring_n_raw]
    direct_params = [pipe.in_low, pipe.in_high, pipe.out_low, pipe.out_high, pipe.sigma_post]
    optimizer = torch.optim.Adam([
        {'params': [pipe.N_raw], 'lr': LR_N},
        {'params': [pipe.sigma_g_raw, pipe.fwhm_raw, pipe.alpha_raw,
                    pipe.ring_strength_raw, pipe.ring_n_raw], 'lr': LR_DEFAULT},
        {'params': direct_params, 'lr': LR_DEFAULT},
    ])

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
        # 무제약 — project_constraints 미적용

        loss_val = (batch_loss / BATCH_SIZE).item()
        phys = get_theta_physical(pipe)
        row = dict(step=step, L_total=loss_val, elapsed=time.time()-t0)
        row.update({k: v/BATCH_SIZE for k, v in batch_parts.items()})
        row.update(phys)
        trace.append(row)

        if (step+1) % 200 == 0:
            print(f"  [{run_label}] step {step+1:4d}  L={loss_val:.5f}  "
                  f"N={phys['N']:.1f}  sg={phys['sigma_g']:.5f}  "
                  f"fwhm={phys['fwhm']:.3f}  alpha={phys['alpha']:.4f}  "
                  f"in_low={phys['in_low']:.4f}  in_high={phys['in_high']:.4f}  "
                  f"sp={phys['sigma_post']:.5f}")

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
print("Phase 4 v6-free: 전 파라미터 자유 최적화 (HANDOFF v7.6 §2)")
print(f"  v1 비교 기준: N≈{V1_RESULT['N']:.1f}, σ_g≈{V1_RESULT['sigma_g']:.4f}, fwhm≈{V1_RESULT['fwhm']:.3f}")
print(f"{'='*65}")

for rep, rep_seed in enumerate(MAIN_SEEDS):
    rep_dir = PHASE4_DIR / f'v6free_rep_{rep}'
    print(f"\n[Rep {rep}] seed={rep_seed}")
    theta_hat, trace = run_one_calibration_v6free(
        pipe_opt, loss_fn, templates_t,
        rep_seed=rep_seed, out_dir=rep_dir,
        run_label=f"v6free_r{rep}",
    )
    rep_results.append(theta_hat)
    rep_final_losses.append(trace[-1]['L_total'])
    print(f"  θ̂_v6free rep {rep}: N={theta_hat['N']:.1f}  sg={theta_hat['sigma_g']:.5f}  "
          f"fwhm={theta_hat['fwhm']:.3f}  sp={theta_hat['sigma_post']:.5f}")

print(f"\nAll reps done. Total: {(time.time()-t_total)/60:.1f} min")

# ── Aggregate ─────────────────────────────────────────────────────────
ALL_PARAMS = ['N', 'sigma_g', 'p', 'fwhm', 'alpha', 'in_low', 'in_high',
              'out_low', 'out_high', 'sigma_post', 'ring_strength', 'ring_n']
theta_mean = {p: float(np.mean([r[p] for r in rep_results])) for p in ALL_PARAMS}
theta_std  = {p: float(np.std( [r[p] for r in rep_results])) for p in ALL_PARAMS}

# ── v1 vs v6-free 대비표 ─────────────────────────────────────────────
print(f"\n{'='*70}")
print("v1 vs v6-free 대비표:")
cmp_params = ['N', 'sigma_g', 'fwhm', 'alpha', 'in_low', 'in_high', 'sigma_post']
print(f"{'Param':<16} {'v1_mean':>10} {'v6free_mean':>12} {'v6free_std':>12} {'Δ':>10}")
print('-'*63)
for p in cmp_params:
    v1_val   = V1_RESULT.get(p, float('nan'))
    v6f_val  = theta_mean.get(p, float('nan'))
    v6f_std  = theta_std.get(p, float('nan'))
    delta    = v6f_val - v1_val if not (np.isnan(v1_val) or np.isnan(v6f_val)) else float('nan')
    v1_str   = f"{v1_val:.5f}" if not np.isnan(v1_val) else "N/A (new)"
    print(f"  {p:<14} {v1_str:>10} {v6f_val:>12.5f} {v6f_std:>12.5f} {delta:>+10.5f}")

loss_v6f = float(np.mean(rep_final_losses))
print(f"  {'loss':14} {'N/A':>10} {loss_v6f:>12.5f} {'':>12}")

# ── 사전 판정표 적용 ─────────────────────────────────────────────────
print(f"\n{'='*65}")
print("사전 판정표 적용 (HANDOFF v7.6 §2):")

# 비물리 기준
NON_PHYS_SG   = 0.10   # > 2× 물리 상한 0.042
NON_PHYS_FWHM = 1.90   # ESF 기준 2.057보다 낮은 임계
NON_PHYS_N_LO = 50.0   # N이 너무 낮음

sg_mean   = theta_mean['sigma_g']
fwhm_mean = theta_mean['fwhm']
N_mean    = theta_mean['N']

flag_sg   = sg_mean   > NON_PHYS_SG
flag_fwhm = fwhm_mean < NON_PHYS_FWHM
flag_N    = N_mean    < NON_PHYS_N_LO

# rep 간 불일치: max-min across reps > 20% of mean
rep_divergence = {}
for p in ['N', 'sigma_g', 'fwhm']:
    vals = [r[p] for r in rep_results]
    spread = max(vals) - min(vals)
    mn = abs(np.mean(vals))
    flag = (spread / mn > 0.20) if mn > 1e-9 else False
    rep_divergence[p] = {'vals': vals, 'spread': spread, 'flag': flag}
    print(f"  {p}: reps={[f'{v:.4f}' for v in vals]}  spread={spread:.4f}  inconsistent={flag}")

print(f"  sg_mean={sg_mean:.5f}  > {NON_PHYS_SG} → nonphys={flag_sg}")
print(f"  fwhm_mean={fwhm_mean:.4f} < {NON_PHYS_FWHM} → nonphys={flag_fwhm}")
print(f"  N_mean={N_mean:.1f} < {NON_PHYS_N_LO} → nonphys={flag_N}")

any_nonphys = flag_sg or flag_fwhm or flag_N
any_inconsistent = any(v['flag'] for v in rep_divergence.values())

# 능선 소멸 조건 (RIDGE_DISSOLVED): 물리값 근방 + rep 일치
sg_ok   = sg_mean   <= 0.042
fwhm_ok = 1.8 <= fwhm_mean <= 2.3
N_ok    = 200 <= N_mean <= 350
rep_ok  = not any_inconsistent

if any_nonphys and any_inconsistent:
    verdict = "RIDGE_PERSISTS"
    detail  = "비물리 파라미터 + reps 간 불일치 수렴 — v1 능선 패턴 유지 (σ_post 추가로 해소되지 않음)"
elif sg_ok and fwhm_ok and N_ok and rep_ok:
    verdict = "RIDGE_DISSOLVED"
    detail  = "물리 근방 안정 수렴 — 즉시 보고 필요 (원고 §3.3 재서술 사안)"
elif any_nonphys:
    verdict = "RIDGE_PERSISTS"
    detail  = "비물리 파라미터 (reps 일치 여부 불문) — 능선 잔존"
else:
    verdict = "RIDGE_PARTIAL"
    detail  = f"부분 물리적 수렴 — 정밀 검토 필요 (sg_ok={sg_ok}, fwhm_ok={fwhm_ok}, N_ok={N_ok})"

print(f"\n  → 판정: {verdict}")
print(f"     {detail}")
if verdict == "RIDGE_DISSOLVED":
    print(f"\n  ★★★ RIDGE_DISSOLVED — 즉시 원고 검증 인스턴스에 보고 필요 ★★★")

# ── Save ──────────────────────────────────────────────────────────────
output = {
    "version": "v6free",
    "purpose": "HANDOFF v7.6 §2 지시68: 전 파라미터 자유 (σ_post 포함) — 능선 잔존 여부 확인",
    "handoff_ref": "HANDOFF_v7.6_review_response.md §2",
    "protocol": {
        "all_params_free": True,
        "no_constraints": True,
        "sigma_post_included": True,
        "seeds": MAIN_SEEDS,
        "n_steps": N_STEPS,
        "batch_size": BATCH_SIZE,
        "lr_N": LR_N,
        "lr_default": LR_DEFAULT,
        "prenormalization": {"a": PRENORM_A, "b": PRENORM_B},
        "only_diff_from_v1": "pipeline now has sigma_post parameter",
    },
    "v1_reference": V1_RESULT,
    "theta_real_mean": theta_mean,
    "theta_real_std":  theta_std,
    "theta_hat_reps":  rep_results,
    "rep_final_losses": rep_final_losses,
    "mean_final_loss": loss_v6f,
    "nonphys_flags": {
        "sigma_g_gt_010": flag_sg,
        "fwhm_lt_190": flag_fwhm,
        "N_lt_50": flag_N,
        "any_nonphys": any_nonphys,
    },
    "rep_inconsistency": {p: v for p, v in rep_divergence.items()},
    "any_rep_inconsistent": any_inconsistent,
    "verdict": verdict,
    "verdict_detail": detail,
    "judgment_criteria": {
        "RIDGE_PERSISTS": "σ_g>0.10 OR fwhm<1.90 OR N<50, AND reps inconsistent (spread/mean>20%)",
        "RIDGE_DISSOLVED": "σ_g≤0.042 AND fwhm∈[1.8,2.3] AND N∈[200,350] AND reps consistent",
    },
    "config": {
        "template_dir": str(TEMPLATE_DIR), "real_dir": str(REAL_DIR),
        "n_steps": N_STEPS, "batch_size": BATCH_SIZE,
        "n_reps": N_REPS, "seeds": MAIN_SEEDS, "device": DEVICE,
    }
}

out_path = PHASE4_DIR / 'theta_real_v6free.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_path}")
print(f"\n{'='*65}")
print(f"Phase 4 v6-free complete  ({(time.time()-t_total)/60:.1f} min)")
print(f"  판정: {verdict}")
