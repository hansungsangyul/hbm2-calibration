#!/usr/bin/env python3
"""
HANDOFF v7.7 실험 2 — 실영상 무결함 위양성률 (M5)
================================================
100개 모델 × dirty2 이미지 5매 → 픽셀 FPR + 거짓 검출 수

지표:
  pixel_fpr         : 예측 비배경(클래스 1,2,3) 픽셀 수 / 전체 픽셀 수
  n_false_components: scipy.ndimage.label, 최소 50픽셀 연결 성분 수

판정: scipy.stats.mannwhitneyu (calib vs manual n_false_components)
  calib < manual & p<0.05 → CALIB_LOWER
  p >= 0.05               → EQUIVALENT
  calib > manual & p<0.05 → CALIB_HIGHER

출력:
  phase4/v77/exp2_results.csv
  phase4/v77/exp2_summary.json
"""
from __future__ import annotations
import json, sys, csv
from pathlib import Path

import numpy as np
import cv2
import torch
from scipy.ndimage import label as nd_label
from scipy.stats import mannwhitneyu

# ── Paths ──────────────────────────────────────────────────────────────────────
DIFFPIPE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
AUGMENT_DIR  = Path('/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation')
REAL_DIR     = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2')
CLEAN_DIR    = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_clean')
PHASE4_DIR   = DIFFPIPE_DIR / 'phase4'
V77_DIR      = PHASE4_DIR / 'v77'
TESTSETS_DIR = PHASE4_DIR / 'p42r_testsets'
P42R_RUNS    = PHASE4_DIR / 'p42r_v2_runs'
P43_RUNS     = PHASE4_DIR / 'p43_runs'

V77_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV  = V77_DIR / 'exp2_results.csv'
OUT_JSON = V77_DIR / 'exp2_summary.json'

if OUT_CSV.exists():
    print(f'[SKIP] {OUT_CSV} already exists')
    sys.exit(0)

sys.path.insert(0, str(DIFFPIPE_DIR / 'src'))
sys.path.insert(0, str(AUGMENT_DIR))

from model import UNet
import config as cfg

# ── Constants ──────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MIN_COMPONENT_SIZE = 50   # 최소 50픽셀 연결 성분 (전작 결함 최소 크기 규약)
print(f'[exp2] device={DEVICE}')

# ── Checkpoint registry ────────────────────────────────────────────────────────
CHECKPOINTS = []
for run_idx in range(20):
    CHECKPOINTS.append(('calib',  run_idx, P42R_RUNS / 'calib'  / f'run{run_idx}' / 'final.pth'))
    CHECKPOINTS.append(('manual', run_idx, P42R_RUNS / 'manual' / f'run{run_idx}' / 'final.pth'))
for ab in ('ab1', 'ab2', 'ab3'):
    for run_idx in range(20):
        CHECKPOINTS.append((ab, run_idx, P43_RUNS / ab / f'run{run_idx}' / 'final.pth'))

# ── dirty2 이미지 로드 ─────────────────────────────────────────────────────────
def load_dirty2_tensor(img_path: Path) -> tuple[torch.Tensor, str]:
    """Load a dirty2 PNG, apply training-identical preprocessing, return [1,3,H,W] tensor."""
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # 패딩 (training과 동일)
    img = cv2.copyMakeBorder(img, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
    img_3ch = np.stack([img] * 3, axis=0).astype(np.float32)
    tensor = torch.from_numpy(img_3ch).unsqueeze(0)   # [1, 3, H, W]
    return tensor, img_path.name


dirty2_paths = sorted(REAL_DIR.glob('*.png'))[:5]
print(f'[exp2] dirty2 images: {[p.name for p in dirty2_paths]}')

dirty2_tensors = []
for p in dirty2_paths:
    t, name = load_dirty2_tensor(p)
    dirty2_tensors.append((t, name))
    print(f'  {name}: shape={tuple(t.shape)}')


# ── FPR 계산 ───────────────────────────────────────────────────────────────────
def compute_fpr_metrics(pred_logits: torch.Tensor) -> tuple[float, int]:
    """
    pred_logits: [1, C, H, W]
    Returns (pixel_fpr, n_false_components)
    """
    pred_class = pred_logits.argmax(dim=1).squeeze(0).cpu().numpy()  # [H, W]
    total_pixels = pred_class.size
    non_bg_mask = (pred_class != 0).astype(np.uint8)
    pixel_fpr = float(non_bg_mask.sum()) / total_pixels

    # connected component 기준 거짓 검출 수
    struct = np.ones((3, 3), dtype=np.int32)
    labeled, n_comp = nd_label(non_bg_mask, structure=struct)
    n_false = 0
    for comp_id in range(1, n_comp + 1):
        if (labeled == comp_id).sum() >= MIN_COMPONENT_SIZE:
            n_false += 1

    return pixel_fpr, n_false


# ── Main loop ──────────────────────────────────────────────────────────────────
rows = []
total = len(CHECKPOINTS)

for i, (condition, run_idx, ckpt_path) in enumerate(CHECKPOINTS):
    print(f'\n[exp2] [{i+1}/{total}] condition={condition}  run{run_idx}')

    if not ckpt_path.exists():
        print(f'  [WARN] checkpoint not found: {ckpt_path}')
        for _, img_name in dirty2_tensors:
            rows.append({
                'condition': condition, 'run_idx': run_idx,
                'image_name': img_name,
                'pixel_fpr': float('nan'),
                'n_false_components': -1,
            })
        continue

    model = UNet(cfg.IN_CHANNELS, cfg.NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        for tensor, img_name in dirty2_tensors:
            tensor_dev = tensor.to(DEVICE)
            logits = model(tensor_dev)
            pixel_fpr, n_false = compute_fpr_metrics(logits)
            print(f'  {img_name}: pixel_fpr={pixel_fpr:.5f}  n_false_components={n_false}')
            rows.append({
                'condition': condition, 'run_idx': run_idx,
                'image_name': img_name,
                'pixel_fpr': pixel_fpr,
                'n_false_components': n_false,
            })

    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

# ── Save CSV ───────────────────────────────────────────────────────────────────
fieldnames = ['condition', 'run_idx', 'image_name', 'pixel_fpr', 'n_false_components']
with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f'\n[exp2] Saved CSV: {OUT_CSV}')

# ── Summary ────────────────────────────────────────────────────────────────────
CONDITIONS_ALL = ['calib', 'manual', 'ab1', 'ab2', 'ab3']

def cond_values(rows, condition, field):
    return [r[field] for r in rows
            if r['condition'] == condition and r[field] != -1
            and not (isinstance(r[field], float) and np.isnan(r[field]))]

summary = {'per_condition': {}, 'mannwhitney_calib_vs_manual': {}, 'verdict': ''}

for cond in CONDITIONS_ALL:
    fprs = cond_values(rows, cond, 'pixel_fpr')
    nfcs = cond_values(rows, cond, 'n_false_components')
    summary['per_condition'][cond] = {
        'median_pixel_fpr': float(np.median(fprs)) if fprs else float('nan'),
        'median_n_false_components': float(np.median(nfcs)) if nfcs else float('nan'),
        'n_samples': len(fprs),
    }

# Mann-Whitney: calib vs manual on n_false_components
calib_nfc  = cond_values(rows, 'calib',  'n_false_components')
manual_nfc = cond_values(rows, 'manual', 'n_false_components')

if len(calib_nfc) > 0 and len(manual_nfc) > 0:
    stat, p_value = mannwhitneyu(calib_nfc, manual_nfc, alternative='two-sided')
    calib_med  = float(np.median(calib_nfc))
    manual_med = float(np.median(manual_nfc))

    if p_value < 0.05 and calib_med < manual_med:
        verdict = 'CALIB_LOWER'
    elif p_value < 0.05 and calib_med > manual_med:
        verdict = 'CALIB_HIGHER'
    else:
        verdict = 'EQUIVALENT'

    summary['mannwhitney_calib_vs_manual'] = {
        'statistic': float(stat),
        'p_value': float(p_value),
        'calib_median_nfc': calib_med,
        'manual_median_nfc': manual_med,
    }
    summary['verdict'] = verdict
else:
    verdict = 'MISSING_DATA'
    summary['verdict'] = verdict

with open(OUT_JSON, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'[exp2] Saved JSON: {OUT_JSON}')

# ── Console verdict summary ────────────────────────────────────────────────────
print('\n' + '=' * 65)
print('EXPERIMENT 2 VERDICT SUMMARY')
print('=' * 65)
for cond in CONDITIONS_ALL:
    s = summary['per_condition'][cond]
    print(f"  {cond:8s}  median_pixel_fpr={s['median_pixel_fpr']:.5f}"
          f"  median_n_false={s['median_n_false_components']:.1f}")
mw = summary.get('mannwhitney_calib_vs_manual', {})
if mw:
    print(f"\n  Mann-Whitney (calib vs manual n_false_components):")
    print(f"    calib_median={mw['calib_median_nfc']:.1f}"
          f"  manual_median={mw['manual_median_nfc']:.1f}"
          f"  p={mw['p_value']:.4f}")
print(f'\n  → VERDICT: {verdict}')
print('=' * 65)
print('[exp2] Done.')
