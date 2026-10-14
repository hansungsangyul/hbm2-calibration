#!/usr/bin/env python3
"""
HANDOFF v7.7 실험 6 보고서 — DeepLabV3+ 아키텍처 전이
=====================================================
exp6_runs/ 의 20개 results.json (4 cells × 5 reps) 읽어서:
  - cell별 GRV 평균 계산
  - (sigma_g, f=5.0) vs (sigma_g, f=1.25) 비교
  - 4개 (sigma_g, ·) 쌍 전부에서 GRV(f=5.0) > GRV(f=1.25) 확인
  - Mann-Whitney 합동 p<0.05 체크
  - 판정:
      두 조건 모두 만족 → TRANSFER_SUPPORTED
      하나라도 실패    → UNET_SPECIFIC

출력: V77_DIR / 'report_6.md'
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from datetime import datetime

import numpy as np
try:
    from scipy.stats import mannwhitneyu
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

V77_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/v77')
V77_DIR.mkdir(parents=True, exist_ok=True)

EXP6_DIR = V77_DIR / 'exp6_runs'
OUT_MD   = V77_DIR / 'report_6.md'

from exp_config import CELLS_ARCH as CELLS

N_REPS = 5


# ── 데이터 로드 ────────────────────────────────────────────────────────────────
print('[report_6] exp6_runs 로드...')

# cell_data[cell_id] = {'grv_means': [...], 'test_id': [...], 'test_ood': [...],
#                       'diverged_cnt': int, 'missing_cnt': int}
cell_data = {}

for cell in CELLS:
    cid = cell['cell_id']
    grv_means = []
    test_ids  = []
    test_oods = []
    n_div     = 0
    n_mis     = 0

    for rep in range(N_REPS):
        path = EXP6_DIR / f'cell{cid}_rep{rep}' / 'results.json'
        if not path.exists():
            print(f'  [MISSING] {path}')
            n_mis += 1
            continue

        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f'  [ERROR] {path}: {e}')
            n_mis += 1
            continue

        diverged = data.get('diverged', False)
        if diverged:
            n_div += 1
            continue

        grv_hist = data.get('grv_history', [])
        if grv_hist:
            grv_means.append(float(np.mean(grv_hist)))

        def _f1(field):
            if field is None: return float('nan')
            if isinstance(field, dict): return field.get('mean_f1', float('nan'))
            return float(field)

        test_ids.append(_f1(data.get('test_id')))
        test_oods.append(_f1(data.get('test_ood')))

    cell_data[cid] = {
        'sigma_g':      cell['sigma_g'],
        'fwhm':         cell['fwhm'],
        'grv_means':    grv_means,
        'test_id':      test_ids,
        'test_ood':     test_oods,
        'n_diverged':   n_div,
        'n_missing':    n_mis,
        'n_conv':       len(grv_means),
    }
    print(f'  cell{cid} (σ_g={cell["sigma_g"]}, f={cell["fwhm"]}): '
          f'수렴={len(grv_means)}  발산={n_div}  누락={n_mis}  '
          f'GRV_mean={(np.mean(grv_means) if grv_means else 0.0):.4f}')

n_total = sum(d['n_conv'] + d['n_diverged'] + d['n_missing'] for d in cell_data.values())
n_found = sum(d['n_conv'] + d['n_diverged'] for d in cell_data.values())
all_missing = all(d['n_missing'] == N_REPS for d in cell_data.values())


# ── 판정 로직 ─────────────────────────────────────────────────────────────────
# (sigma_g=0.0205) 쌍: cell0(f=1.25) vs cell1(f=5.0)
# (sigma_g=0.041)  쌍: cell2(f=1.25) vs cell3(f=5.0)

def mean_safe(lst):
    if not lst: return float('nan')
    return float(np.mean(lst))

grv_c0 = cell_data.get(0, {}).get('grv_means', [])
grv_c1 = cell_data.get(1, {}).get('grv_means', [])
grv_c2 = cell_data.get(2, {}).get('grv_means', [])
grv_c3 = cell_data.get(3, {}).get('grv_means', [])

mean_grv = {i: mean_safe(cell_data.get(i, {}).get('grv_means', [])) for i in range(4)}

# 조건 1: 4개 쌍 전부에서 GRV(f=5.0) > GRV(f=1.25)
pair_lo = [0, 2]  # f=1.25 cell_ids
pair_hi = [1, 3]  # f=5.0  cell_ids
cond1_results = []
for lo, hi in zip(pair_lo, pair_hi):
    grv_lo = mean_grv[lo]
    grv_hi = mean_grv[hi]
    if np.isnan(grv_lo) or np.isnan(grv_hi):
        cond1_results.append(None)
    else:
        cond1_results.append(grv_hi > grv_lo)

cond1_all = all(r is True for r in cond1_results)

# 조건 2: Mann-Whitney 합동 p<0.05
# 모든 f=5.0 runs vs 모든 f=1.25 runs
all_grv_lo = grv_c0 + grv_c2  # f=1.25 (모든 sigma_g 합산)
all_grv_hi = grv_c1 + grv_c3  # f=5.0

pval_mw = float('nan')
cond2_ok = False
if _SCIPY_OK and len(all_grv_lo) >= 1 and len(all_grv_hi) >= 1:
    try:
        stat, pval_mw = mannwhitneyu(all_grv_hi, all_grv_lo, alternative='greater')
        pval_mw = float(pval_mw)
        cond2_ok = pval_mw < 0.05
    except Exception as e:
        print(f'  [WARN] Mann-Whitney 오류: {e}')

# 최종 판정
if all_missing:
    verdict = 'PENDING'
elif cond1_all and cond2_ok:
    verdict = 'TRANSFER_SUPPORTED'
else:
    verdict = 'UNET_SPECIFIC'

print(f'  조건1(모든 쌍 GRV hi>lo)={cond1_all}  '
      f'조건2(MW p<0.05, p={pval_mw:.4f})={cond2_ok}  → {verdict}')


# ── 마크다운 보고서 ────────────────────────────────────────────────────────────
def fmt(v, p=4):
    if v is None or (isinstance(v, float) and v != v):
        return 'N/A'
    return f'{v:.{p}f}'


lines = []
lines.append(f'# HANDOFF v7.7 실험 6 보고서 — DeepLabV3+ 아키텍처 전이')
lines.append(f'')
lines.append(f'**생성일**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  ')
lines.append(f'**총 runs**: 4 cells × 5 reps = 20  |  **데이터 발견**: {n_found}')
lines.append(f'')
lines.append(f'---')
lines.append(f'')

if all_missing:
    lines.append(f'> **STATUS: PENDING** — exp6_runs 디렉토리가 비어있거나 없습니다.')
    lines.append(f'> `run_exp6_arch.py --cell_id 0~3 --rep 0~4` 완료 후 재실행하십시오.')
else:
    # Cell 수치표
    lines.append(f'## 1. Cell별 GRV 및 성능')
    lines.append(f'')
    lines.append(f'| cell_id | σ_g | fwhm | n_conv | GRV mean | TEST_ID | TEST_OOD | 발산 | 누락 |')
    lines.append(f'|---------|-----|------|--------|----------|---------|----------|------|------|')
    for cid in range(4):
        d = cell_data.get(cid, {})
        lines.append(
            f'| {cid} | {d.get("sigma_g", "?")} | {d.get("fwhm", "?")} | '
            f'{d.get("n_conv", 0)} | {fmt(mean_grv.get(cid))} | '
            f'{fmt(mean_safe(d.get("test_id", [])))} | '
            f'{fmt(mean_safe(d.get("test_ood", [])))} | '
            f'{d.get("n_diverged", 0)} | {d.get("n_missing", 0)} |'
        )
    lines.append(f'')

    # 쌍별 비교
    lines.append(f'## 2. fwhm 효과 비교 (GRV)')
    lines.append(f'')
    lines.append(f'| σ_g 쌍 | GRV(f=1.25) | GRV(f=5.0) | GRV(f=5.0) > GRV(f=1.25)? |')
    lines.append(f'|--------|-------------|------------|---------------------------|')
    for lo, hi, sg in [(0, 1, 0.0205), (2, 3, 0.041)]:
        lo_v = mean_grv.get(lo, float('nan'))
        hi_v = mean_grv.get(hi, float('nan'))
        ok   = (not np.isnan(lo_v) and not np.isnan(hi_v) and hi_v > lo_v)
        lines.append(f'| σ_g={sg} | {fmt(lo_v)} | {fmt(hi_v)} | {"YES" if ok else "NO"} |')
    lines.append(f'')

    # Mann-Whitney
    lines.append(f'## 3. Mann-Whitney 합동 검정')
    lines.append(f'')
    lines.append(f'- f=5.0 샘플 (n={len(all_grv_hi)}): cell1 + cell3 GRV 값')
    lines.append(f'- f=1.25 샘플 (n={len(all_grv_lo)}): cell0 + cell2 GRV 값')
    lines.append(f'- 단측 검정: GRV(f=5.0) > GRV(f=1.25)')
    lines.append(f'- p-value: **{fmt(pval_mw, 4)}** (임계 0.05)')
    lines.append(f'- 조건 충족: **{"YES" if cond2_ok else "NO"}**')
    lines.append(f'')

    # 판정
    lines.append(f'## 4. HANDOFF v7.7 §6 판정')
    lines.append(f'')
    lines.append(f'| 판정 조건 | 결과 |')
    lines.append(f'|-----------|------|')
    lines.append(f'| 모든 쌍에서 GRV(f=5.0) > GRV(f=1.25) | {"충족" if cond1_all else "미충족"} |')
    lines.append(f'| Mann-Whitney p < 0.05 | {"충족" if cond2_ok else "미충족"} |')
    lines.append(f'| **최종 판정** | **{verdict}** |')
    lines.append(f'')

    lines.append(f'**결론**: **{verdict}**')
    if verdict == 'TRANSFER_SUPPORTED':
        lines.append(f'  → fwhm 효과(GRV 신호)가 DeepLabV3+에서도 재현됨')
        lines.append(f'  → 관측 효과가 UNet에 특이적이지 않고 범용 아키텍처에 전이됨')
        lines.append(f'  → HANDOFF v7.7 Self-Calibrating 프레임 강화')
    else:
        lines.append(f'  → fwhm 효과가 DeepLabV3+에서 재현되지 않음')
        lines.append(f'  → 관측 효과가 UNet-특이적일 가능성 — 추가 검토 필요')

lines.append(f'')
lines.append(f'---')
lines.append(f'*PENDING 항목 포함 시 run 완료 후 재실행.*')

with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

print(f'[report_6] 저장: {OUT_MD}')
print(f'[report_6] 판정: {verdict}')
