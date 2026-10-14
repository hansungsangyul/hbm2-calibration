#!/usr/bin/env python3
"""
HANDOFF v7.8 §2 — 실험 2 전처리 수정 후 재실행
==============================================
수정사항: 조건별 추론 전처리 분기
  - prenorm 학습 조건 (calib, ab1, exp3): PRENORM_A * img + PRENORM_B 적용 후 추론
  - 원본 학습 조건 (manual, ab2, ab3):    원본 [0,1] 그대로 추론

추가: exp3 (σ_post) 모델 20개 포함

정합성 문턱 (사전 등록): 수정 후 pixel_fpr < 0.5 도달 시에만 v7.7 §2 판정표 적용.
수정 후에도 FPR ≥ 0.5 → 실영상 일반화 실패로 정직 보고.

출력:
  phase4/v77/exp2_results_v2.csv
  phase4/v77/exp2_summary_v2.json
  phase4/v77/overlays_v2/  (수정 전·후 오버레이 PNG 각 5매)
  phase4/v77/report_exp2_v2.md
"""
from __future__ import annotations
import json, sys, csv
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import torch
from scipy.ndimage import label as nd_label
from scipy.stats import mannwhitneyu

DIFFPIPE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
AUGMENT_DIR  = Path('/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation')
REAL_DIR     = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2')
PHASE4_DIR   = DIFFPIPE_DIR / 'phase4'
V77_DIR      = PHASE4_DIR / 'v77'
P42R_RUNS    = PHASE4_DIR / 'p42r_v2_runs'
P43_RUNS     = PHASE4_DIR / 'p43_runs'
EXP3_RUNS    = V77_DIR / 'exp3_runs'
OVERLAY_DIR  = V77_DIR / 'overlays_v2'

OUT_CSV  = V77_DIR / 'exp2_results_v2.csv'
OUT_JSON = V77_DIR / 'exp2_summary_v2.json'
OUT_REP  = V77_DIR / 'report_exp2_v2.md'

if OUT_CSV.exists():
    print(f'[SKIP] {OUT_CSV} already exists')
    sys.exit(0)

OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(DIFFPIPE_DIR / 'src'))
sys.path.insert(0, str(AUGMENT_DIR))

from model import UNet
import config as cfg

from exp_config import PRENORM_A, PRENORM_B, USES_PRENORM

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MIN_COMPONENT_SIZE = 50
print(f'[exp2_v2] device={DEVICE}')

CHECKPOINTS = []
for run_idx in range(20):
    CHECKPOINTS.append(('calib',  run_idx, P42R_RUNS / 'calib'  / f'run{run_idx}' / 'final.pth'))
    CHECKPOINTS.append(('manual', run_idx, P42R_RUNS / 'manual' / f'run{run_idx}' / 'final.pth'))
for ab in ('ab1', 'ab2', 'ab3'):
    for run_idx in range(20):
        CHECKPOINTS.append((ab, run_idx, P43_RUNS / ab / f'run{run_idx}' / 'final.pth'))
for run_idx in range(20):
    CHECKPOINTS.append(('exp3', run_idx, EXP3_RUNS / f'run{run_idx}' / 'final.pth'))


def load_dirty2_raw(img_path: Path):
    """Load dirty2 image as raw float [0,1], same padding as training."""
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    img = cv2.copyMakeBorder(img, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
    return img  # [H, W] float32 [0,1]


def make_tensor(img_raw: np.ndarray, uses_prenorm: bool) -> torch.Tensor:
    """Apply conditional prenorm and return [1,3,H,W] tensor."""
    if uses_prenorm:
        img = PRENORM_A * img_raw + PRENORM_B
    else:
        img = img_raw
    img_3ch = np.stack([img] * 3, axis=0).astype(np.float32)
    return torch.from_numpy(img_3ch).unsqueeze(0)


def compute_fpr_metrics(pred_logits: torch.Tensor):
    pred_class = pred_logits.argmax(dim=1).squeeze(0).cpu().numpy()
    total_pixels = pred_class.size
    non_bg_mask = (pred_class != 0).astype(np.uint8)
    pixel_fpr = float(non_bg_mask.sum()) / total_pixels
    struct = np.ones((3, 3), dtype=np.int32)
    labeled, n_comp = nd_label(non_bg_mask, structure=struct)
    n_false = sum(1 for cid in range(1, n_comp + 1) if (labeled == cid).sum() >= MIN_COMPONENT_SIZE)
    return pixel_fpr, n_false, pred_class


dirty2_paths = sorted(REAL_DIR.glob('*.png'))[:5]
print(f'[exp2_v2] dirty2 images: {[p.name for p in dirty2_paths]}')
dirty2_raw = [(load_dirty2_raw(p), p.name) for p in dirty2_paths]


# ── 오버레이 생성 함수 ─────────────────────────────────────────────────────────
CLASS_COLORS = np.array([[0,0,0],[255,0,0],[0,255,0],[0,0,255]], dtype=np.uint8)

def save_overlay(raw_img: np.ndarray, pred_class: np.ndarray,
                 out_path: Path, title: str = ''):
    H, W = raw_img.shape
    rgb = (np.stack([raw_img]*3, axis=-1) * 255).astype(np.uint8)
    color_mask = CLASS_COLORS[pred_class]
    overlay = cv2.addWeighted(rgb, 0.6, color_mask, 0.4, 0)
    # title 텍스트
    cv2.putText(overlay, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), overlay)


# ── 대표 모델 오버레이 (수정 전·후 비교) ───────────────────────────────────────
def generate_overlays(condition: str, run_idx: int, ckpt_path: Path, version: str):
    if not ckpt_path.exists():
        return
    model = UNet(cfg.IN_CHANNELS, cfg.NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    uses_prenorm = USES_PRENORM.get(condition, False)
    for raw_img, img_name in dirty2_raw:
        # 수정 전 (원본, 항상)
        t_orig = make_tensor(raw_img, False).to(DEVICE)
        with torch.no_grad():
            _, _, pred_orig = compute_fpr_metrics(model(t_orig))
        save_overlay(raw_img, pred_orig,
                     OVERLAY_DIR / f'{version}_orig_{condition}_{run_idx}_{img_name}',
                     f'{condition}/{run_idx} orig')
        # 수정 후 (조건별 prenorm)
        t_fix = make_tensor(raw_img, uses_prenorm).to(DEVICE)
        with torch.no_grad():
            _, _, pred_fix = compute_fpr_metrics(model(t_fix))
        label = 'prenorm' if uses_prenorm else 'orig'
        save_overlay(raw_img, pred_fix,
                     OVERLAY_DIR / f'{version}_fixed_{condition}_{run_idx}_{img_name}',
                     f'{condition}/{run_idx} {label}')
    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

# 대표 모델: calib/run0, manual/run0
print('[exp2_v2] Generating overlay PNGs...')
generate_overlays('calib',  0, P42R_RUNS / 'calib'  / 'run0' / 'final.pth', 'rep')
generate_overlays('manual', 0, P42R_RUNS / 'manual' / 'run0' / 'final.pth', 'rep')
print(f'  Overlays saved → {OVERLAY_DIR}')


# ── Main loop ──────────────────────────────────────────────────────────────────
rows = []
total = len(CHECKPOINTS)

for i, (condition, run_idx, ckpt_path) in enumerate(CHECKPOINTS):
    print(f'\n[exp2_v2] [{i+1}/{total}] condition={condition}  run{run_idx}')
    uses_prenorm = USES_PRENORM.get(condition, False)

    if not ckpt_path.exists():
        print(f'  [WARN] checkpoint not found: {ckpt_path}')
        for raw_img, img_name in dirty2_raw:
            rows.append({'condition': condition, 'run_idx': run_idx,
                         'prenorm_inference': uses_prenorm,
                         'image_name': img_name,
                         'pixel_fpr': float('nan'), 'n_false_components': -1})
        continue

    model = UNet(cfg.IN_CHANNELS, cfg.NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        for raw_img, img_name in dirty2_raw:
            tensor = make_tensor(raw_img, uses_prenorm).to(DEVICE)
            logits = model(tensor)
            pixel_fpr, n_false, _ = compute_fpr_metrics(logits)
            print(f'  {img_name} [{("prenorm" if uses_prenorm else "orig")}]: '
                  f'pixel_fpr={pixel_fpr:.4f}  n_false={n_false}')
            rows.append({'condition': condition, 'run_idx': run_idx,
                         'prenorm_inference': uses_prenorm,
                         'image_name': img_name,
                         'pixel_fpr': pixel_fpr, 'n_false_components': n_false})
    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

fieldnames = ['condition', 'run_idx', 'prenorm_inference',
              'image_name', 'pixel_fpr', 'n_false_components']
with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader(); w.writerows(rows)
print(f'\n[exp2_v2] Saved CSV: {OUT_CSV}')


# ── Summary ────────────────────────────────────────────────────────────────────
def cond_vals(cond, field):
    return [r[field] for r in rows
            if r['condition'] == cond and r[field] != -1
            and not (isinstance(r[field], float) and np.isnan(r[field]))]

CONDITIONS_ALL = ['calib', 'manual', 'ab1', 'ab2', 'ab3', 'exp3']
summary = {'per_condition': {}, 'mannwhitney_calib_vs_manual': {},
           'mannwhitney_exp3_vs_manual': {}, 'verdict': '', 'fpr_threshold_met': False}

for cond in CONDITIONS_ALL:
    fprs = cond_vals(cond, 'pixel_fpr')
    nfcs = cond_vals(cond, 'n_false_components')
    if not fprs:
        continue
    summary['per_condition'][cond] = {
        'median_pixel_fpr': float(np.median(fprs)),
        'median_n_false_components': float(np.median(nfcs)) if nfcs else float('nan'),
        'n_samples': len(fprs),
    }

calib_fpr = summary.get('per_condition', {}).get('calib', {}).get('median_pixel_fpr', 1.0)
summary['fpr_threshold_met'] = calib_fpr < 0.5

calib_nfc  = cond_vals('calib',  'n_false_components')
manual_nfc = cond_vals('manual', 'n_false_components')
exp3_nfc   = cond_vals('exp3',   'n_false_components')

def mw_test(a, b, label):
    if not a or not b:
        return {'verdict': 'MISSING_DATA'}
    stat, p = mannwhitneyu(a, b, alternative='two-sided')
    med_a, med_b = float(np.median(a)), float(np.median(b))
    if p < 0.05 and med_a < med_b:
        vd = 'A_LOWER'
    elif p < 0.05 and med_a > med_b:
        vd = 'A_HIGHER'
    else:
        vd = 'EQUIVALENT'
    return {'statistic': float(stat), 'p_value': float(p),
            'a_median_nfc': med_a, 'b_median_nfc': med_b, 'verdict': vd}

mw_cm = mw_test(calib_nfc, manual_nfc, 'calib vs manual')
mw_em = mw_test(exp3_nfc,  manual_nfc, 'exp3 vs manual')
summary['mannwhitney_calib_vs_manual'] = mw_cm
summary['mannwhitney_exp3_vs_manual']  = mw_em

if not summary['fpr_threshold_met']:
    summary['verdict'] = 'FPR_ABOVE_THRESHOLD'
else:
    summary['verdict'] = mw_cm.get('verdict', 'MISSING_DATA')

with open(OUT_JSON, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'[exp2_v2] Saved JSON: {OUT_JSON}')

print('\n' + '='*65)
print('EXPERIMENT 2 v2 VERDICT SUMMARY')
print('='*65)
for cond in CONDITIONS_ALL:
    s = summary['per_condition'].get(cond, {})
    print(f'  {cond:8s}: median_pixel_fpr={s.get("median_pixel_fpr", float("nan")):.4f}'
          f'  median_n_false={s.get("median_n_false_components", float("nan")):.1f}')
print(f'\n  FPR 문턱(<0.5) 충족: {summary["fpr_threshold_met"]}')
print(f'  calib vs manual → {mw_cm.get("verdict")}  p={mw_cm.get("p_value", float("nan")):.4f}')
print(f'  exp3  vs manual → {mw_em.get("verdict")}  p={mw_em.get("p_value", float("nan")):.4f}')
print(f'  → VERDICT: {summary["verdict"]}')
print('='*65)


# ── 보고서 ────────────────────────────────────────────────────────────────────
now = datetime.now().strftime('%Y-%m-%d %H:%M')
lines = [
    f'# HANDOFF v7.8 §2 — 실험 2 재실행 보고서',
    f'',
    f'**생성일**: {now}',
    f'**CSV**: exp2_results_v2.csv ({len(rows)}행)',
    f'**오버레이**: overlays_v2/ (calib+manual 대표 모델 × 5매 × 수정전/후)',
    f'',
    f'## 전처리 수정 내역',
    f'',
    f'| 조건 | 학습 prenorm | 추론 전처리 (v7→v8) |',
    f'|------|------------|-------------------|',
    f'| calib | ✓ | 원본→prenorm 수정 |',
    f'| manual | ✗ | 원본 유지 |',
    f'| ab1 | ✓ | 원본→prenorm 수정 |',
    f'| ab2 | ✗ | 원본 유지 |',
    f'| ab3 | ✗ | 원본 유지 |',
    f'| exp3 (σ_post) | ✓ | 원본→prenorm 수정 (신규 포함) |',
    f'',
    f'## 결과',
    f'',
    f'| 조건 | median_pixel_fpr | median_n_false |',
    f'|------|----------------|---------------|',
]
for cond in CONDITIONS_ALL:
    s = summary['per_condition'].get(cond, {})
    fpr = s.get('median_pixel_fpr', float('nan'))
    nfc = s.get('median_n_false_components', float('nan'))
    lines.append(f'| {cond} | {fpr:.4f} | {nfc:.1f} |')

lines += [
    f'',
    f'**FPR 정합성 문턱 (< 0.5) 충족**: {summary["fpr_threshold_met"]}',
    f'',
    f'## Mann-Whitney 검정',
    f'',
    f'| 비교 | calib/exp3 중앙값 | manual 중앙값 | p-value | 판정 |',
    f'|------|----------------|-------------|---------|------|',
    f'| calib vs manual | {mw_cm.get("a_median_nfc", float("nan")):.1f} | {mw_cm.get("b_median_nfc", float("nan")):.1f} | {mw_cm.get("p_value", float("nan")):.4f} | **{mw_cm.get("verdict", "N/A")}** |',
    f'| exp3  vs manual | {mw_em.get("a_median_nfc", float("nan")):.1f} | {mw_em.get("b_median_nfc", float("nan")):.1f} | {mw_em.get("p_value", float("nan")):.4f} | **{mw_em.get("verdict", "N/A")}** |',
    f'',
    f'## 최종 판정',
    f'',
    f'**{summary["verdict"]}**',
]

if not summary['fpr_threshold_met']:
    lines.append(f'  → FPR ≥ 0.5: "합성 학습 모델의 실영상 일반화 실패" — §6 한계 승격')
elif summary['verdict'] == 'EQUIVALENT':
    lines.append(f'  → "실영상 위양성 동등 — 합성 우위가 실영상 오검출 비용 없이 성립"')
elif summary['verdict'] == 'A_LOWER':
    lines.append(f'  → "실영상 통계 정합의 직접 증거" — §4.3에 실영상 검증 절 추가')
elif summary['verdict'] == 'A_HIGHER':
    lines.append(f'  → 그대로 보고 + §6 한계 승격. 은폐·재측정 금지')

lines += ['', f'*HANDOFF v7.8 §2 완료*']

with open(OUT_REP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print(f'[exp2_v2] Saved report: {OUT_REP}')
print('[exp2_v2] Done.')
