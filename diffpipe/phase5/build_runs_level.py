#!/usr/bin/env python3
"""
HANDOFF v6.10 §1 지시 38: run 수준 CSV 생성 + 자기 일관성 검사
출력: phase5/phase5_runs_level.csv (150행 + 헤더)
"""
import json, csv, sys
import numpy as np
from pathlib import Path
from scipy.signal import medfilt

BASE    = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase5')
RUNS    = BASE / 'runs'
OUT_CSV = BASE / 'phase5_runs_level.csv'
REF_CSV = BASE / 'phase5_final_150runs.csv'

COLS = ['run_id','cell_id','sigma_g','fwhm','seed',
        'grv5','lm5','diverged','divergence_reason','final_f1','osc_amp']

# ── 1. 150 runs 읽기 ────────────────────────────────────────────────────────
rows = []
for cell_dir in sorted(RUNS.iterdir()):
    for rep_dir in sorted(cell_dir.iterdir()):
        rj = rep_dir / 'results.json'
        if not rj.exists():
            continue
        with open(rj) as f:
            r = json.load(f)

        grv = r.get('grad_rv', [])
        grv5 = float(np.mean(grv[:5])) if len(grv) >= 5 else ''

        # lm5: lambda_max["5"] 단일치 (HANDOFF v6.10 §1 정의)
        lm = r.get('lambda_max', {})
        lm5 = lm.get('5', '')
        if lm5 is not None and lm5 != '' and not (isinstance(lm5, float) and np.isnan(lm5)):
            lm5 = float(lm5)
        else:
            lm5 = ''

        f1_raw = r.get('f1_raw', [])
        if len(f1_raw) >= 10:
            mf = medfilt(np.array(f1_raw[-10:]), 3)
            osc_amp = float(np.max(mf) - np.min(mf))
        elif len(f1_raw) > 0:
            osc_amp = float(np.max(f1_raw) - np.min(f1_raw))
        else:
            osc_amp = ''

        div = bool(r['diverged'])
        dr  = r.get('divergence_reason')
        if dr is None:
            dr = ''

        i_sig = r['i_sigma']
        j_fw  = r['j_fwhm']
        rep   = rep_dir.name.replace('rep', '')

        rows.append({
            'run_id':             f"sg{i_sig}_fw{j_fw}_rep{rep}",
            'cell_id':            r['cell_id'],
            'sigma_g':            r['sigma_g'],
            'fwhm':               r['fwhm'],
            'seed':               r['seed'],
            'grv5':               grv5,
            'lm5':                lm5,
            'diverged':           div,
            'divergence_reason':  dr,
            'final_f1':           r['final_f1'],
            'osc_amp':            osc_amp,
        })

# ── 2. 자기 일관성 검사 ─────────────────────────────────────────────────────
print("=== 자기 일관성 검사 ===")
errors = []

# 검사 2: 행 수 150
if len(rows) != 150:
    errors.append(f"행 수 불일치: {len(rows)} ≠ 150")
else:
    print(f"[OK] 행 수: {len(rows)} == 150")

# 검사 2: cell_id 30종 × 각 5행
from collections import Counter
cell_counts = Counter(r['cell_id'] for r in rows)
bad_cells = {k:v for k,v in cell_counts.items() if v != 5}
if bad_cells:
    errors.append(f"cell 5행 불일치: {bad_cells}")
else:
    print(f"[OK] cell_id 30종 × 각 5행")

# 검사 2: seed 중복 없음
seeds = [r['seed'] for r in rows]
if len(set(seeds)) != 150:
    errors.append(f"seed 중복: {150-len(set(seeds))}건")
else:
    print(f"[OK] seed 150개 모두 고유")

# 검사 3: diverged ↔ divergence_reason 일치
for r in rows:
    if r['diverged'] and r['divergence_reason'] not in (1,2,3):
        errors.append(f"div=True but reason={r['divergence_reason']} @ {r['run_id']}")
    if not r['diverged'] and r['divergence_reason'] != '':
        errors.append(f"div=False but reason={r['divergence_reason']} @ {r['run_id']}")
if not any('div=True but' in e or 'div=False but' in e for e in errors):
    print("[OK] diverged ↔ divergence_reason 상호 일치")

# 검사 1: 셀 집계 재생성 → 기존 CSV 비교
import csv as csvmod
ref = {}
with open(REF_CSV) as f:
    for row in csvmod.DictReader(f):
        ref[row['cell_id']] = row

TOL = 1e-9
agg_errors = []
from collections import defaultdict
cell_rows = defaultdict(list)
for r in rows:
    cell_rows[r['cell_id']].append(r)

for cid, reps in sorted(cell_rows.items()):
    if cid not in ref:
        agg_errors.append(f"{cid}: not in ref CSV"); continue
    R = ref[cid]
    n_done = len(reps)
    n_div  = sum(1 for r in reps if r['diverged'])
    div_c1 = sum(1 for r in reps if r['diverged'] and r['divergence_reason']==1)
    div_c2 = sum(1 for r in reps if r['diverged'] and r['divergence_reason']==2)
    div_c3 = sum(1 for r in reps if r['diverged'] and r['divergence_reason']==3)
    f1s    = [r['final_f1'] for r in reps]
    f1_mean= float(np.mean(f1s))
    f1_std = float(np.std(f1s))
    grv5s  = [r['grv5'] for r in reps if r['grv5'] != '']
    grv5_mean = float(np.mean(grv5s)) if grv5s else float('nan')
    # lm5_mean: ref CSV uses avg of ep5+ep10; new def uses ep5 only — skip lm5_mean check
    oscs   = [r['osc_amp'] for r in reps if r['osc_amp'] != '']
    osc_mean = float(np.mean(oscs)) if oscs else float('nan')

    checks = [
        ('n_done',  n_done,   int(R['n_done'])),
        ('n_div',   n_div,    int(R['n_div'])),
        ('div_c1',  div_c1,   int(R['div_c1'])),
        ('div_c2',  div_c2,   int(R['div_c2'])),
        ('div_c3',  div_c3,   int(R['div_c3'])),
        ('f1_mean', f1_mean,  float(R['f1_mean'])),
        ('f1_std',  f1_std,   float(R['f1_std'])),
        ('grv5_mean', grv5_mean, float(R['grv5_mean'])),
        ('osc_amp_mean', osc_mean, float(R['osc_amp_mean'])),
    ]
    for name, got, exp in checks:
        diff = abs(got - exp) if isinstance(got, float) else abs(got - exp)
        if diff > TOL:
            agg_errors.append(f"{cid}.{name}: got={got} exp={exp} diff={diff:.2e}")

if agg_errors:
    print(f"[FAIL] 셀 집계 불일치 {len(agg_errors)}건:")
    for e in agg_errors[:10]:
        print(f"  {e}")
    errors.extend(agg_errors)
else:
    print(f"[OK] run→셀 집계 재생성 전 수치 일치 (30 cells × 9 항목)")
    print(f"     (lm5_mean 제외: HANDOFF v6.10 §1 재정의 — ep5 단일치로 변경)")

# 결과
print()
if errors:
    print(f"[FAIL] 검사 실패 {len(errors)}건 — CSV 미저장")
    sys.exit(1)
else:
    print("[PASS] 전 검사 통과")

# ── 3. CSV 저장 ─────────────────────────────────────────────────────────────
with open(OUT_CSV, 'w', newline='') as f:
    w = csvmod.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(rows)

print(f"\n저장 완료: {OUT_CSV} ({len(rows)}행)")
print("\n[§1.3 A1 정의 명시]")
print("AUC=0.684: 예측 변수 = grv5 (연속값). roc_auc_score(diverged, grv5) 사용. GRV>60 이진 아님.")
print("λ_max r_pb=0.397 (이전): 입력 = lambda_max['5']+['10'] 평균 (ep5,ep10). 본 CSV의 lm5는 lambda_max['5'] 단일치(HANDOFF v6.10 재정의)로 변경 — 재검산 시 차이 가능.")
