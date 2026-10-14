#!/usr/bin/env python3
"""
HANDOFF v7.7 실험 3·4 통합 보고서
====================================
exp3_runs/ (sigma_post 학습), exp4_runs/ (z-score 학습) 각 20 runs 집계.
D_calib (TEST_ID=0.8757, TEST_OOD=0.787) 대비 판정.

판정표 (HANDOFF v7.7 §3·§4):
  양쪽 셋 모두 결손 < 0.03 → NON_INFERIOR
  어느 한 셋 ≥ 0.03        → INFERIOR
  D_calib 초과             → SUPERIOR

출력:
  V77_DIR / 'report_3_4.md'
  V77_DIR / 'exp3_results.csv'
  V77_DIR / 'exp4_results.csv'
"""
from __future__ import annotations
import json, csv, sys
from pathlib import Path
from datetime import datetime

import numpy as np
try:
    from scipy.stats import ttest_1samp, t as t_dist
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

V77_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/v77')
V77_DIR.mkdir(parents=True, exist_ok=True)

OUT_MD = V77_DIR / 'report_3_4.md'

from exp_config import D_CALIB_ID, D_CALIB_OOD

# D_calib 기준값
N_RUNS      = 20
DEFICIT_THRESHOLD = 0.03


# ── 유틸리티 ──────────────────────────────────────────────────────────────────
def extract_mean_f1(field):
    if field is None:
        return float('nan')
    if isinstance(field, dict):
        return field.get('mean_f1', float('nan'))
    return float(field)


def load_exp_runs(exp_dir: Path, cond_name: str, out_csv: Path):
    """exp_dir에서 run_{i}/results.json 또는 run{i}/results.json 로드.
    CSV 저장 후 (rows_data, conv_id, conv_ood, n_diverged, n_missing) 반환.
    """
    rows_data  = []
    conv_id    = []
    conv_ood   = []
    n_diverged = 0
    n_missing  = 0

    for run_idx in range(N_RUNS):
        candidates = [
            exp_dir / f'run_idx_{run_idx}' / 'results.json',
            exp_dir / f'run{run_idx}'       / 'results.json',
            exp_dir / f'{run_idx}'           / 'results.json',
        ]
        data = None
        for cand in candidates:
            if cand.exists():
                try:
                    with open(cand) as f:
                        data = json.load(f)
                    break
                except Exception as e:
                    print(f'  [WARN] {cand}: {e}')

        if data is None:
            n_missing += 1
            rows_data.append({
                'condition':  cond_name,
                'run_idx':    run_idx,
                'seed':       None,
                'diverged':   None,
                'test_id_f1': float('nan'),
                'test_ood_f1': float('nan'),
                'test_cal_f1': float('nan'),
            })
            continue

        seed     = data.get('seed', None)
        diverged = data.get('diverged', False)
        test_id  = extract_mean_f1(data.get('test_id',  data.get('test_id_f1',  None)))
        test_ood = extract_mean_f1(data.get('test_ood', data.get('test_ood_f1', None)))
        test_cal = extract_mean_f1(data.get('test_cal', data.get('test_cal_f1', None)))

        rows_data.append({
            'condition':  cond_name,
            'run_idx':    run_idx,
            'seed':       seed,
            'diverged':   diverged,
            'test_id_f1':  test_id,
            'test_ood_f1': test_ood,
            'test_cal_f1': test_cal,
        })

        if diverged:
            n_diverged += 1
        else:
            if not (np.isnan(test_id) or np.isnan(test_ood)):
                conv_id.append(test_id)
                conv_ood.append(test_ood)

    # CSV 저장
    fieldnames = ['condition', 'run_idx', 'seed', 'diverged',
                  'test_id_f1', 'test_ood_f1', 'test_cal_f1']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_data)
    print(f'  CSV 저장: {out_csv}')

    n_found = N_RUNS - n_missing
    print(f'  [{cond_name}] 발견={n_found}  수렴={len(conv_id)}  '
          f'발산={n_diverged}  누락={n_missing}')
    return rows_data, conv_id, conv_ood, n_diverged, n_missing


def compute_stats(conv_id, conv_ood):
    """수렴 run 통계 + 결손 + t-검정 반환."""
    n = len(conv_id)
    if n == 0:
        nan = float('nan')
        return {
            'mean_id': nan, 'std_id': nan,
            'mean_ood': nan, 'std_ood': nan,
            'deficit_id': nan, 'deficit_ood': nan,
            'pval_id': nan, 'pval_ood': nan,
            'ci95_id': (nan, nan), 'ci95_ood': (nan, nan),
        }

    mean_id  = float(np.mean(conv_id))
    mean_ood = float(np.mean(conv_ood))
    std_id   = float(np.std(conv_id,  ddof=1)) if n > 1 else 0.0
    std_ood  = float(np.std(conv_ood, ddof=1)) if n > 1 else 0.0

    deficit_id  = D_CALIB_ID  - mean_id
    deficit_ood = D_CALIB_OOD - mean_ood

    pval_id  = float('nan')
    pval_ood = float('nan')
    ci95_id  = (float('nan'), float('nan'))
    ci95_ood = (float('nan'), float('nan'))

    if _SCIPY_OK and n >= 2:
        try:
            res_id  = ttest_1samp(conv_id,  D_CALIB_ID)
            res_ood = ttest_1samp(conv_ood, D_CALIB_OOD)
            pval_id  = float(res_id.pvalue)
            pval_ood = float(res_ood.pvalue)
            se_id  = std_id  / n**0.5
            se_ood = std_ood / n**0.5
            t_crit = t_dist.ppf(0.975, df=n-1)
            ci95_id  = (mean_id  - t_crit * se_id,  mean_id  + t_crit * se_id)
            ci95_ood = (mean_ood - t_crit * se_ood, mean_ood + t_crit * se_ood)
        except Exception as e:
            print(f'  [WARN] 통계 계산 오류: {e}')

    return {
        'mean_id': mean_id, 'std_id': std_id,
        'mean_ood': mean_ood, 'std_ood': std_ood,
        'deficit_id': deficit_id, 'deficit_ood': deficit_ood,
        'pval_id': pval_id, 'pval_ood': pval_ood,
        'ci95_id': ci95_id, 'ci95_ood': ci95_ood,
    }


def get_verdict(s):
    di = s['deficit_id']; do = s['deficit_ood']
    mi = s['mean_id'];    mo = s['mean_ood']
    if np.isnan(di) or np.isnan(do):
        return 'INSUFFICIENT_DATA'
    if mi > D_CALIB_ID and mo > D_CALIB_OOD:
        return 'SUPERIOR'
    if di < DEFICIT_THRESHOLD and do < DEFICIT_THRESHOLD:
        return 'NON_INFERIOR'
    return 'INFERIOR'


def fmt(v, p=4):
    if v is None or (isinstance(v, float) and v != v):
        return 'N/A'
    return f'{v:.{p}f}'


# ── 실험 3 (sigma_post) ──────────────────────────────────────────────────────
print('[report_3_4] 실험 3 (sigma_post) 로드...')
exp3_dir = V77_DIR / 'exp3_runs'
rows3, conv3_id, conv3_ood, n_div3, n_mis3 = load_exp_runs(
    exp3_dir, 'sigma_post', V77_DIR / 'exp3_results.csv')
s3 = compute_stats(conv3_id, conv3_ood)
v3 = get_verdict(s3)

# ── 실험 4 (z-score) ─────────────────────────────────────────────────────────
print('[report_3_4] 실험 4 (z-score) 로드...')
exp4_dir = V77_DIR / 'exp4_runs'
rows4, conv4_id, conv4_ood, n_div4, n_mis4 = load_exp_runs(
    exp4_dir, 'zscore', V77_DIR / 'exp4_results.csv')
s4 = compute_stats(conv4_id, conv4_ood)
v4 = get_verdict(s4)

print(f'  Exp3 판정: {v3}  Exp4 판정: {v4}')


# ── 보고서 작성 ────────────────────────────────────────────────────────────────
def exp_section(label, cond_name, s, n_conv, n_diverged, n_missing, verdict):
    lines = []
    if n_missing == N_RUNS:
        lines.append(f'> **STATUS: PENDING** — {cond_name}_runs 디렉토리가 비어있거나 없습니다.')
        return lines

    lines.append(f'| 셋 | mean±std (n={n_conv}) | D_calib (기준) | 결손 | 95% CI | p-value |')
    lines.append(f'|----|----------------------|----------------|------|--------|---------|')

    for ts, mean, std, d_calib, deficit, ci, pval in [
        ('TEST_ID',  s['mean_id'],  s['std_id'],  D_CALIB_ID,  s['deficit_id'],  s['ci95_id'],  s['pval_id']),
        ('TEST_OOD', s['mean_ood'], s['std_ood'], D_CALIB_OOD, s['deficit_ood'], s['ci95_ood'], s['pval_ood']),
    ]:
        ci_str = (f'[{ci[0]:.4f}, {ci[1]:.4f}]'
                  if not np.isnan(ci[0]) else 'N/A')
        lines.append(
            f'| {ts} | {fmt(mean)} ± {fmt(std)} | {d_calib} | '
            f'{fmt(deficit)} | {ci_str} | {fmt(pval)} |'
        )

    lines.append(f'')
    lines.append(f'**수렴/발산/누락**: {n_conv} / {n_diverged} / {n_missing}')
    lines.append(f'')
    lines.append(f'**HANDOFF v7.7 판정**: **{verdict}**')

    if verdict == 'NON_INFERIOR':
        lines.append(f'  → 양쪽 결손 < {DEFICIT_THRESHOLD} — Self-Calibrating 프레임 유지')
    elif verdict == 'INFERIOR':
        which = []
        if not np.isnan(s['deficit_id'])  and s['deficit_id']  >= DEFICIT_THRESHOLD:
            which.append(f'TEST_ID(결손={s["deficit_id"]:.4f})')
        if not np.isnan(s['deficit_ood']) and s['deficit_ood'] >= DEFICIT_THRESHOLD:
            which.append(f'TEST_OOD(결손={s["deficit_ood"]:.4f})')
        lines.append(f'  → {", ".join(which)} — 재프레임 C안 검토')
    elif verdict == 'SUPERIOR':
        lines.append(f'  → D_calib 초과 — Self-Calibrating 프레임에 유리')
    else:
        lines.append(f'  → 데이터 불충분')
    return lines


out_lines = []
out_lines.append(f'# HANDOFF v7.7 실험 3·4 통합 보고서')
out_lines.append(f'')
out_lines.append(f'**생성일**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  ')
out_lines.append(f'**D_calib 기준**: TEST_ID={D_CALIB_ID}  TEST_OOD={D_CALIB_OOD}')
out_lines.append(f'')
out_lines.append(f'---')
out_lines.append(f'')

# 실험 3
out_lines.append(f'## 실험 3 — sigma_post 추가 학습')
out_lines.append(f'')
out_lines.append(f'**조건**: run_exp3_sigma_post.py --run_idx 0~19 (총 {N_RUNS} runs)')
out_lines.append(f'')
out_lines.extend(exp_section(
    '실험 3', 'sigma_post', s3, len(conv3_id), n_div3, n_mis3, v3))
out_lines.append(f'')
out_lines.append(f'---')
out_lines.append(f'')

# 실험 4
out_lines.append(f'## 실험 4 — z-score 정규화 학습')
out_lines.append(f'')
out_lines.append(f'**조건**: run_exp4_zscore.py --run_idx 0~19 (총 {N_RUNS} runs)')
out_lines.append(f'')
out_lines.extend(exp_section(
    '실험 4', 'zscore', s4, len(conv4_id), n_div4, n_mis4, v4))
out_lines.append(f'')
out_lines.append(f'---')
out_lines.append(f'')

# 종합
out_lines.append(f'## 종합 비교')
out_lines.append(f'')
out_lines.append(f'| 실험 | 조건 | TEST_ID | TEST_OOD | 판정 |')
out_lines.append(f'|------|------|---------|----------|------|')
out_lines.append(
    f'| Exp3 | sigma_post | {fmt(s3["mean_id"])} | {fmt(s3["mean_ood"])} | **{v3}** |')
out_lines.append(
    f'| Exp4 | z-score    | {fmt(s4["mean_id"])} | {fmt(s4["mean_ood"])} | **{v4}** |')
out_lines.append(f'| D_calib | (기준) | {D_CALIB_ID} | {D_CALIB_OOD} | — |')
out_lines.append(f'')
out_lines.append(f'---')
out_lines.append(f'*PENDING 항목 포함 시 run 완료 후 재실행.*')

with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out_lines) + '\n')

print(f'[report_3_4] 저장: {OUT_MD}')
print(f'  Exp3={v3}  Exp4={v4}')
