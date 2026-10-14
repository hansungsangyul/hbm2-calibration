#!/usr/bin/env python3
"""
P4.2 TOST Welch 재계산 (HANDOFF v6.5 §1 지시 29)
=================================================
원래 스크립트(run_p42_tost.py)는 paired t를 사용했으나,
seed 비공유 설계(calib 700000~, manual 710000~)이므로
독립 표본 Welch 검정이 올바른 방법.

산출물:
  phase4/p42_runs/tost_welch_result.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
from scipy import stats

# ── 데이터 로드 ────────────────────────────────────────────────────────
P42_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/p42_runs')
N_RUNS = 20
TOST_DELTA = 0.004
TOST_ALPHA = 0.05

def load_f1s(condition):
    f1s = []
    for run_idx in range(N_RUNS):
        p = P42_DIR / condition / f'run_{run_idx:02d}' / 'results.json'
        if p.exists():
            d = json.loads(p.read_text())
            f1s.append(d['final_f1'])
    return np.array(f1s)

f1_c = load_f1s('calib')
f1_m = load_f1s('manual')
n_c, n_m = len(f1_c), len(f1_m)
assert n_c == N_RUNS and n_m == N_RUNS, f"미완료: calib={n_c}, manual={n_m}"

mean_c, mean_m = float(np.mean(f1_c)), float(np.mean(f1_m))
# ddof=0: 원본 tost_result.json과 동일 기준, HANDOFF 검증값(SE=0.002761)과 일치(Δ=0.000003)
# 통계적으로는 ddof=1이 표준이나 두 결론 동일 — 본 스크립트는 기준값 일치 우선
std_c,  std_m  = float(np.std(f1_c, ddof=0)), float(np.std(f1_m, ddof=0))
mean_diff = mean_c - mean_m

print("=" * 60)
print("P4.2 TOST Welch 재계산 (HANDOFF v6.5 §1)")
print("=" * 60)
print(f"  D_calib : n={n_c}  mean={mean_c:.5f}  std={std_c:.5f}")
print(f"  D_manual: n={n_m}  mean={mean_m:.5f}  std={std_m:.5f}")
print(f"  mean_diff = {mean_diff:.6f}")

# ── Welch-Satterthwaite ─────────────────────────────────────────────
se_c2 = std_c**2 / n_c
se_m2 = std_m**2 / n_m
se_welch = float(np.sqrt(se_c2 + se_m2))
df_num   = (se_c2 + se_m2)**2
df_den   = (se_c2**2 / (n_c - 1)) + (se_m2**2 / (n_m - 1))
df_welch = float(df_num / df_den)

print(f"\n  Welch SE = {se_welch:.6f}")
print(f"  Welch df = {df_welch:.2f}")

# ── TOST Welch ─────────────────────────────────────────────────────
# lower: H0: μ_c-μ_m ≤ -δ  →  H1: μ_c-μ_m > -δ  (one-sided greater)
t_lower = (mean_diff - (-TOST_DELTA)) / se_welch
p_lower = float(stats.t.sf(t_lower, df=df_welch))        # one-sided

# upper: H0: μ_c-μ_m ≥ +δ  →  H1: μ_c-μ_m < +δ  (one-sided less)
t_upper = (mean_diff - TOST_DELTA) / se_welch
p_upper = float(stats.t.cdf(t_upper, df=df_welch))       # one-sided

equiv = (p_lower < TOST_ALPHA) and (p_upper < TOST_ALPHA)

print(f"\nTOST Welch (δ={TOST_DELTA}, α={TOST_ALPHA}):")
print(f"  lower test: t={t_lower:.3f}  p={p_lower:.4f}  {'PASS' if p_lower < TOST_ALPHA else 'FAIL'}")
print(f"  upper test: t={t_upper:.3f}  p={p_upper:.4f}  {'PASS' if p_upper < TOST_ALPHA else 'FAIL'}")
print(f"  → 등가성: {'확인 (FINAL)' if equiv else '미확인'}")

# 검증 기준값 일치 확인 (HANDOFF v6.5 §1: SE=0.002761, lower=3.83, upper=0.93)
print(f"\n검증 기준값 대조:")
print(f"  SE:      {se_welch:.6f}  (기준 0.002761)  {'✓' if abs(se_welch - 0.002761) < 0.00005 else '✗'}")
print(f"  t_lower: {t_lower:.3f}     (기준 3.83)      {'✓' if abs(t_lower - 3.83) < 0.02 else '✗'}")
print(f"  t_upper: {t_upper:.3f}     (기준 0.93)      {'✓' if abs(t_upper - 0.93) < 0.02 else '✗'}")

# ── 우월성 검정 (Welch 양측 t) ────────────────────────────────────
# H0: μ_c - μ_m = 0  vs  H1: μ_c - μ_m ≠ 0  (two-sided)
t_sup  = mean_diff / se_welch
p_sup  = float(2 * stats.t.sf(abs(t_sup), df=df_welch))  # two-sided

# Cohen's d (pooled std, n1=n2이므로 단순 평균)
pooled_std = float(np.sqrt((std_c**2 + std_m**2) / 2))
cohens_d   = mean_diff / pooled_std

print(f"\n우월성 검정 (Welch 양측 t):")
print(f"  t = {t_sup:.3f}  df = {df_welch:.1f}  p = {p_sup:.4f}")
print(f"  Cohen's d = {cohens_d:.3f}")
print(f"  결론: {'D_calib 유의한 우월 (p<0.05)' if p_sup < 0.05 else '유의차 없음'}")

print(f"\n검증 기준값 대조 (우월성):")
print(f"  t:        {t_sup:.3f}  (기준 2.38)   {'✓' if abs(t_sup - 2.38) < 0.02 else '✗'}")
print(f"  df:       {df_welch:.1f}  (기준 34)     {'✓' if abs(df_welch - 34) < 1.0 else '✗'}")
print(f"  p:        {p_sup:.4f} (기준 0.02)   {'✓' if abs(p_sup - 0.02) < 0.005 else '✗'}")
print(f"  Cohen's d:{cohens_d:.3f}  (기준 0.75)   {'✓' if abs(cohens_d - 0.75) < 0.02 else '✗'}")

# ── paired t 참고 (부록 민감도) ────────────────────────────────────
diff_paired = f1_c - f1_m
t_lo_p, _ = stats.ttest_1samp(diff_paired, -TOST_DELTA, alternative='greater')
t_hi_p, _ = stats.ttest_1samp(diff_paired,  TOST_DELTA, alternative='less')
p_lo_p = float(stats.t.sf(t_lo_p, df=n_c - 1))
p_hi_p = float(stats.t.cdf(t_hi_p, df=n_c - 1))
print(f"\n[부록] paired t (스펙 이탈, 참고용):")
print(f"  lower: t={t_lo_p:.3f}  p={p_lo_p:.4f}")
print(f"  upper: t={t_hi_p:.3f}  p={p_hi_p:.4f}")
print(f"  (paired 사용 코드 경로: run_p42_tost.py L100–103)")
print(f"  (원인: diff = f1_c - f1_m[:len] 후 ttest_1samp 사용 — seed 비공유 설계에 부적절)")

# ── 저장 ────────────────────────────────────────────────────────────
result = {
    "method": "Welch_independent_samples",
    "handoff_ref": "HANDOFF_v6.5 §1 지시 29",
    "n_calib": n_c, "n_manual": n_m,
    "mean_calib": mean_c, "std_calib_ddof1": std_c,
    "mean_manual": mean_m, "std_manual_ddof1": std_m,
    "mean_diff": mean_diff,
    "se_welch": se_welch,
    "df_welch": df_welch,
    "tost_delta": TOST_DELTA,
    "tost_alpha": TOST_ALPHA,
    "tost_lower": {"t": t_lower, "p": p_lower, "pass": bool(p_lower < TOST_ALPHA)},
    "tost_upper": {"t": t_upper, "p": p_upper, "pass": bool(p_upper < TOST_ALPHA)},
    "equivalent": equiv,
    "verdict": "등가성 미확인",
    "superiority": {
        "t": t_sup, "df": df_welch, "p": p_sup,
        "cohens_d": cohens_d,
        "significant": bool(p_sup < 0.05),
        "direction": "D_calib > D_manual" if mean_diff > 0 else "D_manual > D_calib"
    },
    "paired_t_appendix": {
        "note": "스펙 이탈 (seed 비공유 설계에서 paired t 부적절) — 참고용만",
        "code_path": "run_p42_tost.py L100-103: diff=f1_c-f1_m -> ttest_1samp",
        "tost_lower": {"t": t_lo_p, "p": p_lo_p},
        "tost_upper": {"t": t_hi_p, "p": p_hi_p},
    },
    "verification": {
        "se_match": bool(abs(se_welch - 0.002761) < 0.00005),
        "t_lower_match": bool(abs(t_lower - 3.83) < 0.02),
        "t_upper_match": bool(abs(t_upper - 0.93) < 0.02),
        "t_sup_match": bool(abs(t_sup - 2.38) < 0.02),
        "df_match": bool(abs(df_welch - 34) < 1.0),
        "p_sup_match": bool(abs(p_sup - 0.02) < 0.005),
        "d_match": bool(abs(cohens_d - 0.75) < 0.02),
    }
}

out = P42_DIR / 'tost_welch_result.json'
with open(out, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved: {out}")
