#!/usr/bin/env python3
"""
Phase 3 — 기능적 등가성 기준선 (지시 16, §M)
=============================================
각 GT 세트 θ*에서 ε 시드만 다른 두 실현 생성 → d_realization 측정

d_realization = pixel-wise MAE between image(θ*, ε_A) and image(θ*, ε_B)
  - 동일 파라미터, 다른 ε 시드 → 순수 노이즈 실현 변동 (자연 변동 기준선)
  - N_PAIRS 쌍 평균하여 안정화

d_gt_est = 기존 functional_equivalence.csv의 'mae' 열
  (pixel-wise MAE between GT images and estimated images)

ratio = d_gt_est / d_realization
pass  = ratio ≤ 1.5

논문 해석: "추정 오차가 노이즈 실현 간 자연 변동 수준에 묻힘"
→ ratio ≤ 1.5이면 'GT와 θ̂ 이미지 차이'가 '순수 노이즈 변동' 수준

Run:
    cd /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe
    PYTHONUNBUFFERED=1 /mnt/gstore/home/xiangyue/HBM_paper/envs/hbm_cudnn/bin/python \
        phase3/compute_d_realization.py 2>&1 | tee phase3/compute_d_realization.log
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import torch
from pipeline import DegradationPipeline, softplus_inverse, sigmoid_box_inverse, _N_LO, _N_HI

PHASE3_DIR = Path(__file__).parent
REAL_DIR   = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_clean')

N_TEMPLATES = 50
N_PAIRS     = 10   # pairs per GT set (seed_A_i / seed_B_i, i=0..9)
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
PASS_THRESHOLD = 1.5

print(f"Device: {DEVICE}")
print(f"Metric: pixel-wise MAE between image(θ*, ε_A) and image(θ*, ε_B)")
print(f"N_PAIRS: {N_PAIRS} per GT set")
print(f"Pass threshold: ratio ≤ {PASS_THRESHOLD}")

# ── Load templates ────────────────────────────────────────────────────
img_paths = sorted(REAL_DIR.glob('*.png'))[:N_TEMPLATES]
templates_np = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                for p in img_paths]
H, W = templates_np[0].shape
templates_t = [torch.from_numpy(t).to(DEVICE) for t in templates_np]
print(f"Loaded {N_TEMPLATES} templates, shape {H}×{W}\n")

# ── Load GT params ────────────────────────────────────────────────────
gt_data = json.load(open(PHASE3_DIR / 'gt_params.json'))
gt_sets = {s['set_id']: s for s in gt_data['gt_sets']}


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_pipeline_to_theta(pipe: DegradationPipeline, theta: dict) -> None:
    with torch.no_grad():
        N_clamped = float(np.clip(theta['N'], _N_LO + 1e-3, _N_HI - 1e-3))
        pipe.N_raw.fill_(sigmoid_box_inverse(N_clamped))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(theta['sigma_g'], 1e-6)))
        pipe.fwhm_raw.fill_(softplus_inverse(theta['fwhm']))
        pipe.alpha_raw.fill_(softplus_inverse(theta['alpha']))
        pipe.ring_strength_raw.fill_(softplus_inverse(theta.get('ring_strength', 0.12)))
        pipe.ring_n_raw.fill_(softplus_inverse(theta.get('ring_n', 8.0)))
        pipe.in_low.fill_(theta.get('in_low', 0.05))
        pipe.in_high.fill_(theta.get('in_high', 0.90))
        pipe.out_low.fill_(theta.get('out_low', 0.10))
        pipe.out_high.fill_(theta.get('out_high', 0.90))
        pipe.p.fill_(theta.get('p', 0.001))


def gen_image(pipe, template, seed):
    seed_all(seed)
    x = template
    eps = {'eps1': torch.randn(H, W, device=DEVICE),
           'eps2': torch.randn(H, W, device=DEVICE),
           'eps3': torch.rand(H, W, device=DEVICE)}
    with torch.no_grad():
        return pipe(x, eps).cpu().numpy()


# ── Compute d_realization for each GT set ────────────────────────────
pipe = DegradationPipeline().to(DEVICE)

print("Computing d_realization (pixel-wise MAE, ε-seed pairs) …")
d_real_results = {}
for set_id in sorted(gt_sets.keys()):
    theta_star = gt_sets[set_id]
    set_pipeline_to_theta(pipe, theta_star)

    pair_maes = []
    for pair_i in range(N_PAIRS):
        tmpl = templates_t[pair_i % len(templates_t)]
        seed_A = 860000 + set_id * 100 + pair_i
        seed_B = 870000 + set_id * 100 + pair_i
        img_A = gen_image(pipe, tmpl, seed_A)
        img_B = gen_image(pipe, tmpl, seed_B)
        pair_maes.append(float(np.mean(np.abs(img_A - img_B))))

    d_real = float(np.mean(pair_maes))
    d_real_std = float(np.std(pair_maes))
    d_real_results[set_id] = {
        'd_realization': d_real,
        'd_realization_std': d_real_std,
        'pair_maes': pair_maes,
        'N_star': float(theta_star['N']),
        'sigma_g_star': float(theta_star['sigma_g']),
        'fwhm_star': float(theta_star['fwhm']),
    }
    print(f"  set_{set_id:02d}  N={theta_star['N']:5.1f} σ_g={theta_star['sigma_g']:.4f} "
          f"fwhm={theta_star['fwhm']:.2f}  d_real={d_real:.5f}±{d_real_std:.5f}")


# ── Read existing functional_equivalence.csv ──────────────────────────
print(f"\nReading {PHASE3_DIR / 'functional_equivalence.csv'} …")
existing_rows = []
with open(PHASE3_DIR / 'functional_equivalence.csv', 'r') as f:
    reader = csv.DictReader(f)
    # read only original columns (strip any previously added columns)
    orig_fields = ['set_id','mean_gt','mean_hat','var_gt','var_hat',
                   'mean_diff','var_diff','mae','mean_relE','var_relE']
    for row in reader:
        existing_rows.append({k: float(row[k]) for k in orig_fields if k in row})


# ── Compute ratio and pass/fail ───────────────────────────────────────
print("\nPer-set results (d_gt_est = mae column from CSV):")
print(f"{'set':<6} {'mae(d_gt_est)':>14} {'d_realization':>14} {'ratio':>8} {'pass':>6}")
print("-" * 55)

new_rows = []
for row in existing_rows:
    set_id = int(row['set_id'])
    d_gt_est = row['mae']   # pixel-wise MAE from GT vs estimated images
    d_real   = d_real_results[set_id]['d_realization']

    ratio = d_gt_est / d_real if d_real > 1e-12 else float('inf')
    pass_flag = (ratio <= PASS_THRESHOLD)

    new_row = dict(row)
    new_row['d_gt_est']      = d_gt_est
    new_row['d_realization'] = d_real
    new_row['ratio']         = ratio
    new_row['equiv_pass']    = int(pass_flag)
    new_rows.append(new_row)

    print(f"  set_{set_id:02d}  {d_gt_est:>14.5f}  {d_real:>14.5f}  {ratio:>8.3f}  "
          f"{'PASS ✓' if pass_flag else 'FAIL ✗'}")


# Summary
n_pass = sum(r['equiv_pass'] for r in new_rows)
all_ratios = [r['ratio'] for r in new_rows]
print(f"\nSummary: {n_pass}/{len(new_rows)} sets pass (ratio ≤ {PASS_THRESHOLD})")
print(f"  ratio range: [{min(all_ratios):.3f}, {max(all_ratios):.3f}]")
print(f"  ratio median: {np.median(all_ratios):.3f}")

if n_pass == len(new_rows):
    interp = "전 세트 PASS — 추정 오차가 노이즈 실현 간 자연 변동 수준에 묻힘 확정"
elif n_pass >= len(new_rows) // 2:
    interp = f"{n_pass}/{len(new_rows)} PASS — 과반 세트 기능적 등가. 실패 세트 분석 필요"
else:
    interp = (f"{n_pass}/{len(new_rows)} PASS — 기능적 비동등. "
              "비식별 파라미터(N·σ_g)가 원인일 가능성.")
print(f"  해석: {interp}")


# ── Write updated CSV ────────────────────────────────────────────────
# Preserve original columns + new ones
out_path = PHASE3_DIR / 'functional_equivalence.csv'
fieldnames = ['set_id','mean_gt','mean_hat','var_gt','var_hat',
              'mean_diff','var_diff','mae','mean_relE','var_relE',
              'd_gt_est','d_realization','ratio','equiv_pass']
with open(out_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(new_rows)
print(f"\nUpdated: {out_path}")
print("Added columns: d_gt_est, d_realization, ratio, equiv_pass")

# ── Save d_realization details ────────────────────────────────────────
detail_path = PHASE3_DIR / 'd_realization_details.json'
with open(detail_path, 'w') as f:
    json.dump({
        'threshold': PASS_THRESHOLD,
        'metric': 'pixel-wise MAE between image(θ*, ε_A) and image(θ*, ε_B)',
        'd_gt_est_source': 'mae column of functional_equivalence.csv (GT vs estimated pixel-wise MAE)',
        'seeds': {
            'A_pattern': '860000 + set_id*100 + pair_i',
            'B_pattern': '870000 + set_id*100 + pair_i',
        },
        'n_pairs': N_PAIRS,
        'pass_count': n_pass,
        'total_sets': len(new_rows),
        'ratios': {str(r['set_id']): round(r['ratio'], 4) for r in new_rows},
        'interpretation': interp,
        'per_set': {str(sid): v for sid, v in d_real_results.items()},
    }, f, indent=2)
print(f"Saved: {detail_path}")
print("\n지시 16 (§M) 기능적 등가성 기준선 분석 완료.")
