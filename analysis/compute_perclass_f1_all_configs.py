#!/usr/bin/env python3
"""
(1) 절제 조건별 클래스 F1 집계 — perclass_f1_all_configs.csv

데이터 소스 (모두 results.json에서 직접 읽기 — 새 inference 불필요):
  calib/manual : p42r_v2_perclass_f1.csv (5차 회신 기준)
  ab1/ab2/ab3  : p43_runs/{ab1,ab2,ab3}/run*/results.json
  v5e          : v77/exp3b_runs/run*/results.json
  sigma_post   : v77/exp3_runs/run*/results.json
  zscore       : v77/exp4_runs/run*/results.json

모집단 정의: 불안정 판정(diverged=True) run도 종료 시점 F1로 포함 (5차 동일)
"""
from __future__ import annotations
import csv, json
from pathlib import Path

import numpy as np

DIFFPIPE  = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
V77       = DIFFPIPE / 'phase4' / 'v77'
P43       = DIFFPIPE / 'phase4' / 'p43_runs'
OUT_DIR   = Path('/mnt/gstore/home/xiangyue/HBM_paper2')

TESTSETS = ['test_id', 'test_ood', 'test_cal']

# ── 기존 calib/manual 데이터 (5차 perclass_f1.csv 재사용) ─────────────────────
def read_existing_calib_manual():
    rows = []
    with open(OUT_DIR / 'perclass_f1.csv') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

# ── results.json 묶음 집계 ────────────────────────────────────────────────────
def aggregate_runs(run_dirs: list[Path], config_name: str, n_expected: int):
    """results.json 목록에서 testset별 per-class F1 20-run mean/std 반환"""
    # per_ts[ts] = list of [c0,c1,c2,c3] per run
    per_ts: dict[str, list] = {ts: [] for ts in TESTSETS}

    found = 0
    for run_dir in sorted(run_dirs):
        rj = run_dir / 'results.json'
        if not rj.exists():
            print(f"  WARN: {rj} not found, skipping")
            continue
        d = json.load(open(rj))
        found += 1
        for ts in TESTSETS:
            ts_data = d.get(ts, {})
            f1_by_class = ts_data.get('f1_by_class', None)
            if f1_by_class is None:
                print(f"  WARN: {config_name}/{run_dir.name}/{ts}: f1_by_class missing")
                per_ts[ts].append([float('nan')] * 4)
            else:
                per_ts[ts].append(f1_by_class[:4])

    print(f"  {config_name}: {found}/{n_expected} runs loaded")

    result_rows = []
    for ts in TESTSETS:
        data = np.array(per_ts[ts])        # (n_runs, 4)
        mean = np.nanmean(data, axis=0)
        std  = np.nanstd(data, axis=0)
        macro_4 = float(np.nanmean(mean))
        defect   = float(np.nanmean(mean[1:4]))  # c1~c3
        result_rows.append({
            'config': config_name,
            'testset': ts,
            'n_runs': found,
            'f1_class0': round(float(mean[0]), 6),
            'f1_class1': round(float(mean[1]), 6),
            'f1_class2': round(float(mean[2]), 6),
            'f1_class3': round(float(mean[3]), 6),
            'f1_macro_4class':       round(macro_4, 6),
            'f1_macro_defect_only':  round(defect,  6),
            'f1_class0_std': round(float(std[0]), 6),
            'f1_class1_std': round(float(std[1]), 6),
            'f1_class2_std': round(float(std[2]), 6),
            'f1_class3_std': round(float(std[3]), 6),
            'f1_macro_std':  round(float(np.nanstd(np.nanmean(data, axis=1))), 6),
        })
    return result_rows


# ── 주 처리 ──────────────────────────────────────────────────────────────────
all_rows = []

# 1. calib / manual (5차 CSV 그대로 흡수, 열 이름 통일)
print("Loading calib/manual from existing perclass_f1.csv ...")
for r in read_existing_calib_manual():
    # 5차 CSV 열명과 동일 → 그대로 추가
    # f1_macro_defect_only가 있는지 확인
    if 'f1_macro_defect_only' not in r:
        c0,c1,c2,c3 = float(r['f1_class0']),float(r['f1_class1']),float(r['f1_class2']),float(r['f1_class3'])
        r['f1_macro_defect_only'] = round((c1+c2+c3)/3, 6)
    # n_runs 형 통일
    r['n_runs'] = int(r['n_runs'])
    all_rows.append(r)
    print(f"  calib_manual/{r['config']}/{r['testset']}: f1_macro={r['f1_macro_4class']}")

# 2. AB-1 / AB-2 / AB-3 (p43_runs)
for cond in ['ab1', 'ab2', 'ab3']:
    print(f"\nLoading {cond} from p43_runs ...")
    run_dirs = sorted((P43 / cond).glob('run*'))
    rows = aggregate_runs(run_dirs, cond, n_expected=20)
    all_rows.extend(rows)
    for r in rows:
        print(f"  {cond}/{r['testset']}: 4class={r['f1_macro_4class']:.4f}  defect={r['f1_macro_defect_only']:.4f}")

# 3. D_v5e (exp3b)
print(f"\nLoading v5e from v77/exp3b_runs ...")
run_dirs = sorted((V77 / 'exp3b_runs').glob('run*'))
rows = aggregate_runs(run_dirs, 'v5e', n_expected=20)
all_rows.extend(rows)
for r in rows:
    print(f"  v5e/{r['testset']}: 4class={r['f1_macro_4class']:.4f}  defect={r['f1_macro_defect_only']:.4f}")

# 4. D_σp / sigma_post (exp3)
print(f"\nLoading sigma_post from v77/exp3_runs ...")
run_dirs = sorted((V77 / 'exp3_runs').glob('run*'))
rows = aggregate_runs(run_dirs, 'sigma_post', n_expected=20)
all_rows.extend(rows)
for r in rows:
    print(f"  sigma_post/{r['testset']}: 4class={r['f1_macro_4class']:.4f}  defect={r['f1_macro_defect_only']:.4f}")

# 5. z-score (exp4)
print(f"\nLoading zscore from v77/exp4_runs ...")
run_dirs = sorted((V77 / 'exp4_runs').glob('run*'))
rows = aggregate_runs(run_dirs, 'zscore', n_expected=20)
all_rows.extend(rows)
for r in rows:
    print(f"  zscore/{r['testset']}: 4class={r['f1_macro_4class']:.4f}  defect={r['f1_macro_defect_only']:.4f}")

# ── 저장 ─────────────────────────────────────────────────────────────────────
out_csv = OUT_DIR / 'perclass_f1_all_configs.csv'
fieldnames = ['config','testset','n_runs',
              'f1_class0','f1_class1','f1_class2','f1_class3',
              'f1_macro_4class','f1_macro_defect_only',
              'f1_class0_std','f1_class1_std','f1_class2_std','f1_class3_std',
              'f1_macro_std']
with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(all_rows)

print(f"\nSaved: {out_csv}  ({len(all_rows)} rows)")
