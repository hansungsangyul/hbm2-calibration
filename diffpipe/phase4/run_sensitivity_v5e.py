#!/usr/bin/env python3
"""
HANDOFF v6.4 §1.2 민감도 미니 체크
θ_manual + σ_post=0.02 기준점에서 σ_post와 ring_strength 각 ±20% 섭동
→ 4개 loss 성분 변화율 표

통과 기준: L_nps에 대해 두 파라미터의 반응 패턴이 구분됨
  - σ_post: 전 주파수 NPS 플로어 이동
  - ring: 국소 봉우리 (radial PSD 패턴 차이)
"""
import sys, json
from pathlib import Path

import cv2, numpy as np

ROOT = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
sys.path.insert(0, str(ROOT / 'src'))

import torch
import torch.nn.functional as F
from pipeline import DegradationPipeline, softplus_inverse, sigmoid_box_inverse, _N_LO, _N_HI
from losses import CalibrationLoss

TEMPLATE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_clean')
REAL_DIR     = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2')
PHASE4_DIR   = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4')

PRENORM_A = 0.501834
PRENORM_B = 0.173320
BATCH     = 32
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

# θ_manual 기준점 (σ_post=0.02 추가)
THETA_BASE = dict(
    N=300.0, sigma_g=0.0408, p=0.001, fwhm=2.0956,
    alpha=0.35, in_low=0.05, in_high=0.90,
    out_low=0.10, out_high=0.90,
    sigma_post=0.02,    # 기준점
    ring_strength=0.12, ring_n=8.0,
)

PERTURB_RATIO = 0.20   # ±20%

# ── Load images ───────────────────────────────────────────────────────
print("Loading images …")
templates_raw = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
                 for p in sorted(TEMPLATE_DIR.glob('*.png'))[:10]]
H, W = templates_raw[0].shape
templates_t = [torch.from_numpy(PRENORM_A * t + PRENORM_B).to(DEVICE) for t in templates_raw]

real_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
           for p in sorted(REAL_DIR.glob('*.png'))[:50]]
loss_fn = CalibrationLoss(real_np).to(DEVICE)
print(f"  {len(templates_t)} templates, {len(real_np)} real, device={DEVICE}")


def build_pipe(theta: dict) -> DegradationPipeline:
    pipe = DegradationPipeline().to(DEVICE)
    with torch.no_grad():
        N_c = float(np.clip(theta['N'], _N_LO+1, _N_HI-1))
        pipe.N_raw.fill_(sigmoid_box_inverse(N_c))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(theta['sigma_g'], 1e-4)))
        pipe.p.fill_(theta['p'])
        pipe.fwhm_raw.fill_(softplus_inverse(theta['fwhm']))
        pipe.alpha_raw.fill_(softplus_inverse(theta['alpha']))
        pipe.in_low.fill_(theta['in_low'])
        pipe.in_high.fill_(theta['in_high'])
        pipe.out_low.fill_(theta['out_low'])
        pipe.out_high.fill_(theta['out_high'])
        pipe.sigma_post.fill_(theta['sigma_post'])
        pipe.ring_strength_raw.fill_(softplus_inverse(theta['ring_strength']))
        pipe.ring_n_raw.fill_(softplus_inverse(theta['ring_n']))
    for param in pipe.parameters():
        param.requires_grad_(False)
    return pipe


def measure_loss(theta: dict, seed: int = 42) -> dict:
    pipe = build_pipe(theta)
    torch.manual_seed(seed)
    total = {k: 0.0 for k in ['L_vm', 'L_nps', 'L_rad', 'L_hist', 'L_total']}
    n = 0
    x = templates_t[0]
    for _ in range(BATCH):
        eps = {
            'eps1':     torch.randn(H, W, device=DEVICE),
            'eps2':     torch.randn(H, W, device=DEVICE),
            'eps3':     torch.rand(H, W, device=DEVICE),
            'eps_post': torch.randn(H, W, device=DEVICE),
        }
        syn = pipe(x, eps)
        L, parts = loss_fn(syn)
        total['L_total'] += L.item()
        for k in ['L_vm', 'L_nps', 'L_rad', 'L_hist']:
            total[k] += parts[k].item()
        n += 1
    return {k: v/n for k, v in total.items()}


def pct_change(base_val, pert_val):
    if abs(base_val) < 1e-12:
        return float('nan')
    return 100.0 * (pert_val - base_val) / abs(base_val)


# ── 측정 ─────────────────────────────────────────────────────────────
print("\n기준점 loss 측정 (σ_post=0.02, ring=0.12) …")
base = measure_loss(THETA_BASE)
print(f"  base: L_vm={base['L_vm']:.5f}  L_nps={base['L_nps']:.5f}  "
      f"L_rad={base['L_rad']:.5f}  L_hist={base['L_hist']:.5f}  total={base['L_total']:.5f}")

results = {}
for param_name, base_val_key in [('sigma_post', 'sigma_post'), ('ring_strength', 'ring_strength')]:
    base_val = THETA_BASE[base_val_key]
    for direction, sign in [('+20%', 1+PERTURB_RATIO), ('-20%', 1-PERTURB_RATIO)]:
        label = f"{param_name} {direction}"
        theta_pert = dict(THETA_BASE)
        theta_pert[base_val_key] = base_val * sign
        print(f"측정: {label} ({base_val_key}={theta_pert[base_val_key]:.4f}) …")
        pert = measure_loss(theta_pert)
        results[label] = {
            'perturbed_val': theta_pert[base_val_key],
            'L_vm_pct':   pct_change(base['L_vm'],   pert['L_vm']),
            'L_nps_pct':  pct_change(base['L_nps'],  pert['L_nps']),
            'L_rad_pct':  pct_change(base['L_rad'],  pert['L_rad']),
            'L_hist_pct': pct_change(base['L_hist'], pert['L_hist']),
        }

# ── 결과 표 ─────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("§1.2 민감도 표 — 각 파라미터 ±20% 섭동 시 loss 변화율 (%)")
print(f"  기준: σ_post={THETA_BASE['sigma_post']}, ring_strength={THETA_BASE['ring_strength']}")
print(f"{'섭동':30} {'L_vm%':>8} {'L_nps%':>8} {'L_rad%':>8} {'L_hist%':>8}")
print('-'*64)
for label, r in results.items():
    print(f"  {label:<28} {r['L_vm_pct']:>+7.1f}%  {r['L_nps_pct']:>+7.1f}%  "
          f"{r['L_rad_pct']:>+7.1f}%  {r['L_hist_pct']:>+7.1f}%")

# ── 통과 판정 ────────────────────────────────────────────────────────
# L_nps에 대해 σ_post와 ring_strength의 부호 또는 크기가 구분돼야 함
sp_plus_nps  = results['sigma_post +20%']['L_nps_pct']
sp_minus_nps = results['sigma_post -20%']['L_nps_pct']
ri_plus_nps  = results['ring_strength +20%']['L_nps_pct']
ri_minus_nps = results['ring_strength -20%']['L_nps_pct']

sp_nps_range = abs(sp_plus_nps - sp_minus_nps)
ri_nps_range = abs(ri_plus_nps - ri_minus_nps)

# 구분 가능: σ_post와 ring의 L_nps 응답 패턴이 다름
# 기준: 두 파라미터의 L_nps 변화 방향이나 크기 차이가 유의
pattern_diff = abs(sp_nps_range - ri_nps_range)
passed = pattern_diff > 0.5 or (sp_plus_nps * ri_plus_nps < 0)  # 방향 다르거나 크기 차이

print(f"\n패턴 분석 (L_nps):")
print(f"  σ_post: +20%→{sp_plus_nps:+.1f}%  -20%→{sp_minus_nps:+.1f}%  range={sp_nps_range:.1f}%")
print(f"  ring:   +20%→{ri_plus_nps:+.1f}%  -20%→{ri_minus_nps:+.1f}%  range={ri_nps_range:.1f}%")
print(f"  pattern_diff = {pattern_diff:.2f}  → {'PASS (구분 가능)' if passed else 'FAIL (구분 불가)'}")

# ── 저장 ─────────────────────────────────────────────────────────────
output = {
    "handoff_ref": "HANDOFF_v6.4_sigma_post_ruling.md §1.2",
    "base_theta": THETA_BASE,
    "perturb_ratio": PERTURB_RATIO,
    "base_loss": base,
    "perturbed_results": results,
    "pattern_analysis": {
        "sigma_post_nps_range_pct": sp_nps_range,
        "ring_nps_range_pct": ri_nps_range,
        "pattern_diff": pattern_diff,
        "passed": passed,
    }
}
out_path = PHASE4_DIR / 'sensitivity_v5e.json'
with open(out_path, 'w') as f:
    import json
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_path}")
print(f"§1.2 민감도 체크: {'PASS → v5e 착수 가능' if passed else 'FAIL → 착수 전 보고 필요'}")
