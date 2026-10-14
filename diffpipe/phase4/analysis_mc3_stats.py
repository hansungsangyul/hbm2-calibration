#!/usr/bin/env python3
"""
MC3 통계 보강 재분석 (HANDOFF v7.6 §3.1, 지시 69)
===================================================
(a) Valley / Paradox / Interaction run-level Mann-Whitney
(b) Wilson 95% CI per cell
(c) crit3 민감도: 임계 0.05/0.10/0.15 재판정

입력: phase5/phase5_runs_level.csv + phase5/runs/ (results.json)
출력: reanalysis_mc3_mc5.md (MC3 섹션)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import medfilt

PHASE5_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase5')
CSV_PATH   = PHASE5_DIR / 'phase5_runs_level.csv'
OUT_MD     = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/reanalysis_mc3_mc5.md')

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH.name}")
print(f"sigma_g values: {sorted(df.sigma_g.unique())}")
print(f"fwhm values:    {sorted(df.fwhm.unique())}")
print(f"cells: {df.cell_id.nunique()}")

# Grid constants
SG_VALS   = sorted(df.sigma_g.unique())   # 5 values
FWHM_VALS = sorted(df.fwhm.unique())      # 6 values
# σ_g 물리 상한 = 0.041 (Tier1 기준)
PHYS_SG_UPPER = 0.041


# ═════════════════════════════════════════════════════════════════════
# (a) Run-level Mann-Whitney 검정
# ═════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("(a) Run-level Mann-Whitney 검정")
print("="*60)

# ── (a1) Valley: fwhm=1.25 행 vs 인접 행들 (fwhm=0.625, fwhm=2.5), σ_g ≤ 0.041 층화
print("\n[a1] Valley 검정: fwhm=1.25 vs {0.625, 2.5}, σ_g ≤ 0.041 층화")
fwhm_valley    = 1.25
fwhm_adjacent  = [0.625, 2.5]

valley_rows   = df[(df.fwhm == fwhm_valley)   & (df.sigma_g <= PHYS_SG_UPPER)]
adjacent_rows = df[(df.fwhm.isin(fwhm_adjacent)) & (df.sigma_g <= PHYS_SG_UPPER)]

print(f"  Valley group  (fwhm=1.25, σ_g≤{PHYS_SG_UPPER}):   n={len(valley_rows)}, osc_amp mean={valley_rows.osc_amp.mean():.4f}")
print(f"  Adjacent group (fwhm∈{{0.625,2.5}}, σ_g≤{PHYS_SG_UPPER}): n={len(adjacent_rows)}, osc_amp mean={adjacent_rows.osc_amp.mean():.4f}")

if len(valley_rows) >= 5 and len(adjacent_rows) >= 5:
    stat_v, pval_v = stats.mannwhitneyu(
        valley_rows.osc_amp, adjacent_rows.osc_amp, alternative='greater')
    n1, n2 = len(valley_rows), len(adjacent_rows)
    # Rank-biserial r effect size
    r_rb_v = 1 - 2*stat_v / (n1 * n2)
    print(f"  Mann-Whitney U={stat_v:.1f}  p={pval_v:.4f}  r_rb={r_rb_v:.4f}")
else:
    stat_v, pval_v, r_rb_v = float('nan'), float('nan'), float('nan')
    print("  insuffient data")

# ── (a2) Paradox: fwhm=0 열 내 σ_g 하위 2행 vs 중위 2행
print("\n[a2] Paradox 검정: fwhm=0, low-σ_g (2행) vs mid-σ_g (2행)")
fwhm_zero = 0.0
col_zero = df[df.fwhm == fwhm_zero].sort_values('sigma_g')
sg_sorted = sorted(col_zero.sigma_g.unique())
print(f"  σ_g 값 (fwhm=0): {sg_sorted}")

if len(sg_sorted) >= 5:
    low_sgs  = sg_sorted[:2]
    mid_sgs  = sg_sorted[2:4]
else:
    low_sgs  = sg_sorted[:max(1, len(sg_sorted)//2)]
    mid_sgs  = sg_sorted[len(low_sgs):len(low_sgs)+len(low_sgs)]

low_rows = col_zero[col_zero.sigma_g.isin(low_sgs)]
mid_rows = col_zero[col_zero.sigma_g.isin(mid_sgs)]

print(f"  Low σ_g {low_sgs}: n={len(low_rows)}, osc_amp={low_rows.osc_amp.mean():.4f}, f1={low_rows.final_f1.mean():.4f}")
print(f"  Mid σ_g {mid_sgs}: n={len(mid_rows)}, osc_amp={mid_rows.osc_amp.mean():.4f}, f1={mid_rows.final_f1.mean():.4f}")

if len(low_rows) >= 3 and len(mid_rows) >= 3:
    stat_p, pval_p = stats.mannwhitneyu(
        low_rows.osc_amp, mid_rows.osc_amp, alternative='two-sided')
    r_rb_p = 1 - 2*stat_p / (len(low_rows)*len(mid_rows))
    # f1_std proxy: use std of final_f1 within each group
    print(f"  osc_amp: U={stat_p:.1f}  p={pval_p:.4f}  r_rb={r_rb_p:.4f}")
    stat_pf, pval_pf = stats.mannwhitneyu(
        low_rows.final_f1, mid_rows.final_f1, alternative='two-sided')
    print(f"  final_f1: U={stat_pf:.1f}  p={pval_pf:.4f}")
else:
    stat_p, pval_p, r_rb_p = float('nan'), float('nan'), float('nan')
    pval_pf = float('nan')
    print("  insufficient data")

# ── (a3) Interaction: osc_amp의 f×σ_g 순열 상호작용
print("\n[a3] Interaction 검정: osc_amp의 fwhm×sigma_g 순열 상호작용")
# Rank-transform approach (Conover)
df_ia = df.copy()
df_ia['rank_osc'] = stats.rankdata(df_ia.osc_amp)

# 2-way ANOVA on ranks
# fwhm as categorical, sigma_g as categorical
from scipy.stats import f_oneway

# Per-level groups for interaction: cell-level mean ranks
cell_stats = df_ia.groupby('cell_id').agg(
    sigma_g=('sigma_g', 'first'),
    fwhm=('fwhm', 'first'),
    mean_rank=('rank_osc', 'mean'),
    n=('rank_osc', 'count'),
).reset_index()

# Permutation test: under H0 (no interaction), shuffle fwhm assignments
N_PERM = 5000
np.random.seed(42)

def interaction_stat(df_c):
    pivot = df_c.pivot_table(values='mean_rank', index='sigma_g', columns='fwhm', aggfunc='mean')
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return float('nan')
    row_means = pivot.mean(axis=1)
    col_means = pivot.mean(axis=0)
    grand     = pivot.values.mean()
    inter     = pivot.values - row_means.values[:,None] - col_means.values[None,:] + grand
    return float(np.nansum(inter**2))

obs_stat = interaction_stat(cell_stats)
perm_stats = []
for _ in range(N_PERM):
    shuffled = cell_stats.copy()
    shuffled['fwhm'] = np.random.permutation(cell_stats['fwhm'].values)
    perm_stats.append(interaction_stat(shuffled))
perm_stats = [s for s in perm_stats if not np.isnan(s)]
pval_ia = float(np.mean([s >= obs_stat for s in perm_stats]))
print(f"  Obs interaction SS (rank-transform) = {obs_stat:.2f}")
print(f"  Permutation p (N={N_PERM}) = {pval_ia:.4f}")


# ═════════════════════════════════════════════════════════════════════
# (b) Wilson 95% CI per cell
# ═════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("(b) Wilson 95% CI per cell")
print("="*60)

def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (float('nan'), float('nan'), float('nan'))
    z = stats.norm.ppf(1 - alpha/2)
    p_hat = k / n
    denom = 1 + z**2/n
    centre = (p_hat + z**2/(2*n)) / denom
    half = (z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2))) / denom
    return (float(centre - half), float(centre + half), float(p_hat))

cell_wilson = []
for cell_id, grp in df.groupby('cell_id'):
    k = grp.diverged.sum()
    n = len(grp)
    sg = grp.sigma_g.iloc[0]
    fw = grp.fwhm.iloc[0]
    lo, hi, p = wilson_ci(k, n)
    cell_wilson.append({
        'cell_id': cell_id, 'sigma_g': sg, 'fwhm': fw,
        'k_div': int(k), 'n': int(n), 'p_hat': p,
        'ci_lo': lo, 'ci_hi': hi,
    })
cell_wilson_df = pd.DataFrame(cell_wilson).sort_values(['sigma_g', 'fwhm'])

print(f"{'cell_id':<12} {'σ_g':>8} {'fwhm':>6} {'div/n':>7} {'p̂':>6} {'95% CI':>18}")
print('-'*60)
for _, row in cell_wilson_df.iterrows():
    print(f"  {row.cell_id:<10} {row.sigma_g:>8.4f} {row.fwhm:>6.3f} "
          f"  {int(row.k_div)}/{int(row.n)} {row.p_hat:>6.3f}  "
          f"[{row.ci_lo:.3f}, {row.ci_hi:.3f}]")


# ═════════════════════════════════════════════════════════════════════
# (c) crit3 민감도: 임계 0.05/0.10/0.15
# ═════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("(c) crit3 민감도: 임계 0.05/0.10/0.15")
print("="*60)

THRESHOLDS = [0.05, 0.10, 0.15]

# Load f1_filt from results.json for each run
print("Loading f1_filt from results.json …")
run_f1filt = {}
missing = 0
for _, row in df.iterrows():
    run_id = row.run_id
    # run_id format: sg{i}_fw{j}_rep{k}
    parts = run_id.split('_')
    sg_str = parts[0]   # sg0..sg4
    fw_str = parts[1]   # fw0..fw5
    rep_str = parts[2]  # rep0..rep4
    run_dir = PHASE5_DIR / 'runs' / f'{sg_str}_{fw_str}' / rep_str
    res_path = run_dir / 'results.json'
    if res_path.exists():
        with open(res_path) as f:
            res = json.load(f)
        run_f1filt[run_id] = res.get('f1_filt', res.get('f1_raw', []))
    else:
        run_f1filt[run_id] = None
        missing += 1
print(f"  Loaded: {sum(v is not None for v in run_f1filt.values())} / {len(df)} runs  (missing: {missing})")

def recompute_crit3(f1_filt, threshold):
    if f1_filt is None or len(f1_filt) < 10:
        return None
    window = f1_filt[-10:]
    return bool((max(window) - min(window)) > threshold)

# Per threshold: recompute diverged flag (crit3 only) and divergence rate per cell
thresh_results = {}
for thr in THRESHOLDS:
    records = []
    for _, row in df.iterrows():
        rid = row.run_id
        ff = run_f1filt.get(rid)
        crit3 = recompute_crit3(ff, thr)
        # Preserve crit1/crit2 from original (only crit3 changes)
        orig_div = bool(row.diverged)
        orig_reason = row.divergence_reason
        if orig_reason in [1.0, 2.0] or orig_reason in [1, 2]:
            new_div = True  # crit1/2 preserved
        elif crit3 is None:
            new_div = orig_div  # couldn't compute
        else:
            new_div = crit3
        records.append({
            'run_id': rid, 'cell_id': row.cell_id,
            'sigma_g': row.sigma_g, 'fwhm': row.fwhm,
            'diverged_new': new_div,
        })
    thresh_df = pd.DataFrame(records)
    # Divergence rate per cell
    cell_rates = thresh_df.groupby('cell_id').agg(
        sigma_g=('sigma_g', 'first'), fwhm=('fwhm', 'first'),
        k_div=('diverged_new', 'sum'), n=('diverged_new', 'count'),
    ).reset_index()
    cell_rates['rate'] = cell_rates['k_div'] / cell_rates['n']
    thresh_results[thr] = cell_rates

# Print comparison table
print(f"\n발생률 비교 (thr=0.05 | 0.10 | 0.15):")
print(f"{'cell_id':<12} {'σ_g':>8} {'fwhm':>6}  {'div_05':>7} {'div_10':>7} {'div_15':>7}")
print('-'*56)
base_df = thresh_results[0.05].sort_values(['sigma_g', 'fwhm'])
for _, r05 in base_df.iterrows():
    cid = r05.cell_id
    r10 = thresh_results[0.10][thresh_results[0.10].cell_id == cid].iloc[0]
    r15 = thresh_results[0.15][thresh_results[0.15].cell_id == cid].iloc[0]
    print(f"  {cid:<10} {r05.sigma_g:>8.4f} {r05.fwhm:>6.3f}  "
          f"  {r05.k_div}/{r05.n}   {r10.k_div}/{r10.n}   {r15.k_div}/{r15.n}")


# ═════════════════════════════════════════════════════════════════════
# Write markdown
# ═════════════════════════════════════════════════════════════════════
def md_wilson_table():
    lines = [
        "| cell_id | σ_g | fwhm | div/n | p̂ | 95% CI (Wilson) |",
        "|---------|-----|------|-------|---|-----------------|",
    ]
    for _, r in cell_wilson_df.iterrows():
        lines.append(
            f"| {r.cell_id} | {r.sigma_g:.4f} | {r.fwhm:.3f} | "
            f"{int(r.k_div)}/{int(r.n)} | {r.p_hat:.3f} | "
            f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}] |"
        )
    return '\n'.join(lines)

def md_crit3_table():
    lines = [
        "| cell_id | σ_g | fwhm | div_0.05 | div_0.10 | div_0.15 |",
        "|---------|-----|------|----------|----------|----------|",
    ]
    for _, r05 in base_df.iterrows():
        cid = r05.cell_id
        r10 = thresh_results[0.10][thresh_results[0.10].cell_id == cid].iloc[0]
        r15 = thresh_results[0.15][thresh_results[0.15].cell_id == cid].iloc[0]
        lines.append(
            f"| {cid} | {r05.sigma_g:.4f} | {r05.fwhm:.3f} | "
            f"{int(r05.k_div)}/{int(r05.n)} | {int(r10.k_div)}/{int(r10.n)} | {int(r15.k_div)}/{int(r15.n)} |"
        )
    return '\n'.join(lines)

mc3_text = f"""## MC3 통계 보강 재분석

**실행일**: 자동 생성 (HANDOFF v7.6 §3.1)
**입력**: `phase5/phase5_runs_level.csv` + `phase5/runs/*/results.json`

---

### (a) Run-level Mann-Whitney 검정

#### (a1) Valley 구조: fwhm=1.25 vs 인접 행 {{0.625, 2.5}}, σ_g ≤ {PHYS_SG_UPPER} 층화

| 그룹 | n | osc_amp 평균 |
|------|---|-------------|
| Valley (fwhm=1.25) | {len(valley_rows)} | {valley_rows.osc_amp.mean():.4f} |
| Adjacent (fwhm∈{{0.625,2.5}}) | {len(adjacent_rows)} | {adjacent_rows.osc_amp.mean():.4f} |

- Mann-Whitney U = {stat_v:.1f}, p = {pval_v:.4f} (단측: valley > adjacent)
- Rank-biserial r = {r_rb_v:.4f}

#### (a2) Paradox 구조: fwhm=0 열, 저-σ_g {low_sgs} vs 중-σ_g {mid_sgs}

| 그룹 | n | osc_amp 평균 | final_f1 평균 |
|------|---|-------------|-------------|
| 저 σ_g {low_sgs} | {len(low_rows)} | {low_rows.osc_amp.mean():.4f} | {low_rows.final_f1.mean():.4f} |
| 중 σ_g {mid_sgs} | {len(mid_rows)} | {mid_rows.osc_amp.mean():.4f} | {mid_rows.final_f1.mean():.4f} |

- osc_amp: U = {stat_p:.1f}, p = {pval_p:.4f} (양측)
- final_f1: U = {stat_pf:.1f}, p = {pval_pf:.4f} (양측)
- Rank-biserial r (osc_amp) = {r_rb_p:.4f}

#### (a3) 상호작용 검정 (순열 기반, 순위 변환)

- 관측 상호작용 SS (rank-transform) = {obs_stat:.2f}
- 순열 p 값 (N = {N_PERM}) = {pval_ia:.4f}

---

### (b) Wilson 95% CI per cell (발생률)

{md_wilson_table()}

---

### (c) crit3 임계 민감도 (0.05 / 0.10 / 0.15)

원시 `f1_filt[-10:]` (medfilt 적용) 기준 재판정. crit1·crit2는 원본 유지.

{md_crit3_table()}

---
"""

print("\n" + "="*60)
print("MC3 분석 완료")
print("="*60)

# Write intermediate result (MC5 appended by separate script)
existing = ""
if OUT_MD.exists():
    existing = OUT_MD.read_text()
    if "## MC3" in existing:
        # Replace MC3 section
        start = existing.find("## MC3")
        end = existing.find("## MC5", start)
        if end > 0:
            existing = existing[:start] + mc3_text + existing[end:]
        else:
            existing = existing[:start] + mc3_text
    else:
        existing = mc3_text + existing
else:
    existing = f"""# 재분석 보고서 (HANDOFF v7.6 §3, 지시 69)

**실행일**: 자동 생성
**목적**: MC3 통계 보강 + MC5 불확실성 CI + MC4 수치 패키지

---

{mc3_text}
"""

OUT_MD.write_text(existing)
print(f"Saved: {OUT_MD}")
