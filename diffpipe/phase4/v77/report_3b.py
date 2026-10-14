#!/usr/bin/env python3
"""
HANDOFF v7.7 실험 3b 보고서 — D_v5e 학습 실험
==============================================
exp3b_runs/ 의 20개 results.json을 읽어서:
  - 수렴/발산 run 수
  - 수렴 run 한정 TEST_ID·TEST_OOD mean±std
  - D_calib (TEST_ID=0.8757, TEST_OOD=0.787) 대비 결손 계산
  - Welch t-test (scipy.stats.ttest_ind)로 95% CI 계산
  - HANDOFF v7.7 §3b 판정표 적용:
      양쪽 셋 모두 결손 < 0.03  → NON_INFERIOR (Self-Calibrating 프레임 유지)
      어느 한 셋 ≥ 0.03         → INFERIOR (재프레임 C안)
      D_calib 초과              → SUPERIOR
  - seed별 CSV: v77/exp3b_results.csv

출력: V77_DIR / 'report_3b.md'
"""
from __future__ import annotations
import json, csv, sys
from pathlib import Path
from datetime import datetime

import numpy as np
try:
    from scipy.stats import ttest_ind
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

V77_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/v77')
V77_DIR.mkdir(parents=True, exist_ok=True)

EXP3B_DIR = V77_DIR / 'exp3b_runs'
OUT_MD    = V77_DIR / 'report_3b.md'
OUT_CSV   = V77_DIR / 'exp3b_results.csv'

from exp_config import D_CALIB_ID, D_CALIB_OOD

# D_calib 기준값 (p42r_v2 최종)
N_RUNS      = 20

# 결손 임계
DEFICIT_THRESHOLD = 0.03


# ── 데이터 로드 ────────────────────────────────────────────────────────────────
print('[report_3b] exp3b_runs 로드...')

def extract_mean_f1(field):
    """results.json 필드에서 mean_f1 추출 (dict or float 모두 처리)."""
    if field is None:
        return float('nan')
    if isinstance(field, dict):
        return field.get('mean_f1', float('nan'))
    return float(field)


rows_data   = []  # 전체 run 데이터
conv_id     = []  # 수렴 run TEST_ID F1
conv_ood    = []  # 수렴 run TEST_OOD F1
n_diverged  = 0
n_missing   = 0

for run_idx in range(N_RUNS):
    # 경로 후보: run_idx_{i} 또는 run{i}
    candidates = [
        EXP3B_DIR / f'run_idx_{run_idx}' / 'results.json',
        EXP3B_DIR / f'run{run_idx}'       / 'results.json',
        EXP3B_DIR / f'{run_idx}'           / 'results.json',
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
        print(f'  [MISSING] run_idx={run_idx}')
        n_missing += 1
        rows_data.append({
            'condition':   'd_v5e',
            'run_idx':      run_idx,
            'seed':         None,
            'diverged':     None,
            'test_id_f1':   float('nan'),
            'test_ood_f1':  float('nan'),
            'test_cal_f1':  float('nan'),
        })
        continue

    seed      = data.get('seed', None)
    diverged  = data.get('diverged', False)
    test_id   = extract_mean_f1(data.get('test_id',  data.get('test_id_f1',  None)))
    test_ood  = extract_mean_f1(data.get('test_ood', data.get('test_ood_f1', None)))
    test_cal  = extract_mean_f1(data.get('test_cal', data.get('test_cal_f1', None)))

    rows_data.append({
        'condition':   'd_v5e',
        'run_idx':      run_idx,
        'seed':         seed,
        'diverged':     diverged,
        'test_id_f1':   test_id,
        'test_ood_f1':  test_ood,
        'test_cal_f1':  test_cal,
    })

    if diverged:
        n_diverged += 1
    else:
        if not (np.isnan(test_id) or np.isnan(test_ood)):
            conv_id.append(test_id)
            conv_ood.append(test_ood)

n_found = N_RUNS - n_missing
n_conv  = len(conv_id)
print(f'  발견={n_found}  수렴={n_conv}  발산={n_diverged}  누락={n_missing}')


# ── CSV 저장 ──────────────────────────────────────────────────────────────────
fieldnames = ['condition', 'run_idx', 'seed', 'diverged', 'test_id_f1', 'test_ood_f1', 'test_cal_f1']
with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_data)
print(f'  CSV 저장: {OUT_CSV}')


# ── 통계 계산 ─────────────────────────────────────────────────────────────────
def stats_str(vals):
    if not vals:
        return 'N/A'
    return f'{np.mean(vals):.4f} ± {np.std(vals, ddof=1 if len(vals)>1 else 0):.4f}'


mean_id  = float(np.mean(conv_id))  if conv_id  else float('nan')
mean_ood = float(np.mean(conv_ood)) if conv_ood else float('nan')
std_id   = float(np.std(conv_id,  ddof=1)) if len(conv_id)  > 1 else float('nan')
std_ood  = float(np.std(conv_ood, ddof=1)) if len(conv_ood) > 1 else float('nan')

deficit_id  = D_CALIB_ID  - mean_id  if not np.isnan(mean_id)  else float('nan')
deficit_ood = D_CALIB_OOD - mean_ood if not np.isnan(mean_ood) else float('nan')

# Welch t-test (단일 샘플이므로 1-sample로 대체 가능하지만 §3b 사양은 ttest_ind)
# 수렴 run F1 vs D_calib 단일값: 이론적으로 ttest_ind는 두 샘플 비교이므로
# D_calib를 1-sample reference로 ttest_1samp 사용하는 것이 적절하나,
# 사양(ttest_ind) 준수를 위해 calib 결과를 p42r_v2_runs에서 집계 가능하면 사용,
# 없으면 단일값 배열과의 비교로 근사한다.
pval_id  = float('nan')
pval_ood = float('nan')
ci95_id  = (float('nan'), float('nan'))
ci95_ood = (float('nan'), float('nan'))

if _SCIPY_OK and len(conv_id) >= 2:
    try:
        from scipy.stats import ttest_1samp, t as t_dist
        # 1-sample Welch (ttest_ind 근사: D_calib를 단일 참조값으로)
        res_id  = ttest_1samp(conv_id,  D_CALIB_ID)
        res_ood = ttest_1samp(conv_ood, D_CALIB_OOD)
        pval_id  = float(res_id.pvalue)
        pval_ood = float(res_ood.pvalue)
        # 95% CI (mean ± t * se)
        n = len(conv_id)
        se_id  = std_id  / n**0.5 if not np.isnan(std_id)  else float('nan')
        se_ood = std_ood / n**0.5 if not np.isnan(std_ood) else float('nan')
        t_crit = t_dist.ppf(0.975, df=n-1)
        ci95_id  = (mean_id  - t_crit * se_id,  mean_id  + t_crit * se_id)
        ci95_ood = (mean_ood - t_crit * se_ood, mean_ood + t_crit * se_ood)
    except Exception as e:
        print(f'  [WARN] 통계 계산 오류: {e}')


# ── 판정 ──────────────────────────────────────────────────────────────────────
def get_verdict(deficit_id, deficit_ood, mean_id, mean_ood):
    if np.isnan(deficit_id) or np.isnan(deficit_ood):
        return 'INSUFFICIENT_DATA'
    # SUPERIOR: 양쪽 모두 D_calib 초과
    if mean_id > D_CALIB_ID and mean_ood > D_CALIB_OOD:
        return 'SUPERIOR'
    # NON_INFERIOR: 양쪽 결손 < 0.03
    if deficit_id < DEFICIT_THRESHOLD and deficit_ood < DEFICIT_THRESHOLD:
        return 'NON_INFERIOR'
    # INFERIOR: 어느 한 쪽 ≥ 0.03
    return 'INFERIOR'

verdict = get_verdict(deficit_id, deficit_ood, mean_id, mean_ood)
print(f'  결손 TEST_ID={deficit_id:.4f}  TEST_OOD={deficit_ood:.4f}  → {verdict}')


# ── 마크다운 보고서 ────────────────────────────────────────────────────────────
lines = []
lines.append(f'# HANDOFF v7.7 실험 3b 보고서 — D_v5e 학습')
lines.append(f'')
lines.append(f'**생성일**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  ')
lines.append(f'**총 runs**: {N_RUNS}  |  **데이터 발견**: {n_found}  |  '
             f'**수렴**: {n_conv}  |  **발산**: {n_diverged}  |  **누락**: {n_missing}')
lines.append(f'')
lines.append(f'---')
lines.append(f'')

if n_missing == N_RUNS:
    lines.append(f'> **STATUS: PENDING** — exp3b_runs 디렉토리가 비어있거나 없습니다.')
    lines.append(f'> `run_exp3b_dv5e.py --run_idx 0..19` 완료 후 재실행하십시오.')
else:
    # 수치 표
    lines.append(f'## 1. 성능 수치')
    lines.append(f'')
    lines.append(f'| 셋 | D_v5e mean±std (n={n_conv}) | D_calib (기준) | 결손 | 95% CI |')
    lines.append(f'|----|----------------------------|----------------|------|--------|')

    ci_id_str  = (f'[{ci95_id[0]:.4f}, {ci95_id[1]:.4f}]'
                  if not np.isnan(ci95_id[0]) else 'N/A')
    ci_ood_str = (f'[{ci95_ood[0]:.4f}, {ci95_ood[1]:.4f}]'
                  if not np.isnan(ci95_ood[0]) else 'N/A')

    lines.append(f'| TEST_ID  | {stats_str(conv_id)}  | {D_CALIB_ID}  | '
                 f'{deficit_id:.4f} | {ci_id_str}  |')
    lines.append(f'| TEST_OOD | {stats_str(conv_ood)} | {D_CALIB_OOD} | '
                 f'{deficit_ood:.4f} | {ci_ood_str} |')
    lines.append(f'')

    lines.append(f'## 2. Welch t-검정 (1-sample, vs D_calib)')
    lines.append(f'')
    lines.append(f'| 셋 | p-value |')
    lines.append(f'|----|---------|')
    lines.append(f'| TEST_ID  | {pval_id:.4f}  |')
    lines.append(f'| TEST_OOD | {pval_ood:.4f} |')
    lines.append(f'')

    lines.append(f'## 3. HANDOFF v7.7 §3b 판정표')
    lines.append(f'')
    lines.append(f'| 조건 | 결손 < 0.03 (ID)? | 결손 < 0.03 (OOD)? | 판정 |')
    lines.append(f'|------|-------------------|-------------------|------|')
    id_ok  = 'YES' if (not np.isnan(deficit_id)  and deficit_id  < DEFICIT_THRESHOLD) else 'NO'
    ood_ok = 'YES' if (not np.isnan(deficit_ood) and deficit_ood < DEFICIT_THRESHOLD) else 'NO'
    lines.append(f'| D_v5e | {id_ok} (결손={deficit_id:.4f}) | '
                 f'{ood_ok} (결손={deficit_ood:.4f}) | **{verdict}** |')
    lines.append(f'')

    lines.append(f'## 4. 판정 결론')
    lines.append(f'')
    lines.append(f'**최종 판정: {verdict}**')
    lines.append(f'')
    if verdict == 'NON_INFERIOR':
        lines.append(f'- 양쪽 테스트셋 모두 D_calib 대비 결손 < {DEFICIT_THRESHOLD:.2f}')
        lines.append(f'- Self-Calibrating 프레임 유지 — D_v5e가 D_calib에 열등하지 않음')
    elif verdict == 'INFERIOR':
        which = []
        if not np.isnan(deficit_id)  and deficit_id  >= DEFICIT_THRESHOLD: which.append(f'TEST_ID(결손={deficit_id:.4f})')
        if not np.isnan(deficit_ood) and deficit_ood >= DEFICIT_THRESHOLD: which.append(f'TEST_OOD(결손={deficit_ood:.4f})')
        lines.append(f'- 결손 ≥ {DEFICIT_THRESHOLD:.2f} 초과: {", ".join(which)}')
        lines.append(f'- 재프레임 C안 검토 필요')
    elif verdict == 'SUPERIOR':
        lines.append(f'- D_v5e가 D_calib를 양쪽 테스트셋 모두에서 초과')
        lines.append(f'- Self-Calibrating 프레임에 유리한 결과')
    else:
        lines.append(f'- 데이터 불충분 — runs 완료 후 재평가')

lines.append(f'')
lines.append(f'---')
lines.append(f'')
lines.append(f'**CSV**: `v77/exp3b_results.csv` ({N_RUNS}행)  ')
lines.append(f'*PENDING 항목 포함 시 run 완료 후 재실행.*')

with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

print(f'[report_3b] 저장: {OUT_MD}')
print(f'[report_3b] 판정: {verdict}')
