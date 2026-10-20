#!/usr/bin/env python3
"""
§5.2 상호작용 순열 검정 독립 재구현  (HANDOFF v7.13 지시 83)
=============================================================
원본 코드(analysis_mc3_stats.py)와 무관하게 처음부터 구현.

사양:
  - 반응변수 4: diverged, final_f1, osc_amp, grv5
  - 통계량 2 : SS_AB (n_ij 가중), F = (SS_AB/df_AB) / (SS_within/df_within)
  - 치환 2   : (a) 원 반응 전체 치환 (전역 귀무)
               (b) 주효과 제거 잔차 치환 (ter Braak/Still-White)
  - midrank 순위 변환 (scipy.stats.rankdata, method='average')
  - B=200,000, p=(b+1)/(B+1), seed=42
  - 귀무분포 저장: v0.9/nulldist/{response}_{stat}_{perm}.npy

최적화: 데이터를 (I,J,n) 텐서로 미리 정렬 → for loop 없는 numpy 텐서 연산.
raw 치환 시 rankdata 재호출 없이 z_obs 직접 치환 (rank(permute(y)) = permute(rank(y))).

산출물:
  v0.9/permtest_replication.md  — 16행 결과표
  v0.9/nulldist/                — 16개 npy
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# ── 경로 ──────────────────────────────────────────────────────────────
CSV_PATH = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/campaign_closure_v6.14/phase5_runs_level.csv')
OUT_DIR  = Path('/mnt/gstore/home/xiangyue/HBM_paper2/v0.9')
NULL_DIR = OUT_DIR / 'nulldist'
NULL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
B    = 200_000

# ── 데이터 로드 ───────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df['diverged'] = df['diverged'].astype(float)

SG_VALS   = sorted(df.sigma_g.unique())
FWHM_VALS = sorted(df.fwhm.unique())
I = len(SG_VALS)    # 5
J = len(FWHM_VALS)  # 6
n = 5               # reps per cell
N_TOTAL = len(df)   # 150

df_ab_dof = (I - 1) * (J - 1)   # 20
df_w_dof  = I * J * (n - 1)     # 120

print(f"Loaded {N_TOTAL} rows  sigma_g: {SG_VALS}  fwhm: {FWHM_VALS}")

# ── G0 게이트 ──────────────────────────────────────────────────────────
print("\n=== G0 데이터 정합성 ===")
expected = {
    2.5:  [3, 2, 1, 1, 3],
    5.0:  [2, 4, 4, 2, 3],
    10.0: [5, 5, 5, 3, 1],
    0.0:  [2, 3, 0, 0, 2],
}
g0_pass = True
for fw, exp_counts in expected.items():
    obs = [int(df[(df.fwhm == fw) & (df.sigma_g == sg)].diverged.sum())
           for sg in SG_VALS]
    match = (obs == exp_counts)
    print(f"  fwhm={fw}: {obs}  기대={exp_counts}  {'OK' if match else 'FAIL'}")
    if not match:
        g0_pass = False

total_div = int(df.diverged.sum())
total_match = (total_div == 65)
print(f"  총 발산: {total_div}/150  기대=65/150  {'OK' if total_match else 'FAIL'}")
if not total_match:
    g0_pass = False

if not g0_pass:
    print("[ABORT] G0 실패 — 계산 중단")
    sys.exit(1)
print("[G0 PASS] 데이터 정합성 확인됨")

# ── 데이터를 (I, J, n) 순서로 정렬 ────────────────────────────────────
# 이후 z.reshape(I, J, n)[i, j, :] = 셀(sigma_g=SG_VALS[i], fwhm=FWHM_VALS[j]) 5 reps
df['_sg_idx'] = df.sigma_g.map({sg: idx for idx, sg in enumerate(SG_VALS)})
df['_fw_idx'] = df.fwhm.map({fw: idx for idx, fw in enumerate(FWHM_VALS)})
df_sorted = df.sort_values(['_sg_idx', '_fw_idx']).reset_index(drop=True)

# 정렬 검증
for i, sg in enumerate(SG_VALS):
    for j, fw in enumerate(FWHM_VALS):
        cnt = int(((df_sorted._sg_idx == i) & (df_sorted._fw_idx == j)).sum())
        assert cnt == n, f"셀({sg},{fw}) count={cnt} != {n}"

# ── 핵심 함수: for loop 없는 numpy 텐서 연산 ──────────────────────────
def compute_ss_f_fast(z_3d: np.ndarray) -> tuple[float, float]:
    """
    z_3d: (I, J, n) rank tensor (df_sorted 순서로 정렬된 데이터)
    SS_AB = n * sum( (cm_ij - rm_i - cm_j + gm)^2 )
    F = (SS_AB/df_AB) / (SS_within/df_within)
    """
    cm  = z_3d.mean(axis=2)              # (I, J) cell means
    rm  = cm.mean(axis=1, keepdims=True) # (I, 1) row means
    col = cm.mean(axis=0, keepdims=True) # (1, J) col means
    gm  = cm.mean()                      # scalar grand mean
    inter = cm - rm - col + gm           # (I, J) interaction
    ss_ab = n * float(np.sum(inter ** 2))
    ss_w  = float(np.sum((z_3d - cm[:, :, None]) ** 2))
    f_stat = (ss_ab / df_ab_dof) / (ss_w / df_w_dof)
    return ss_ab, f_stat


def make_resid(z_obs: np.ndarray) -> np.ndarray:
    """ter Braak 잔차: z_ijk - (z_i. + z_j. - z..)"""
    z3 = z_obs.reshape(I, J, n)
    cm  = z3.mean(axis=2)
    rm  = cm.mean(axis=1)
    col = cm.mean(axis=0)
    gm  = cm.mean()
    main_eff = rm[:, None] + col[None, :] - gm   # (I, J)
    resid = z3 - main_eff[:, :, None]             # (I, J, n)
    return resid.reshape(-1)


# ── 메인 순열 검정 ─────────────────────────────────────────────────────
RESPONSES = ['diverged', 'final_f1', 'osc_amp', 'grv5']
results   = []
rng       = np.random.default_rng(SEED)
t_start   = time.time()

for resp in RESPONSES:
    y     = df_sorted[resp].values.astype(float)
    z_obs = rankdata(y, method='average')          # global midrank, shape (150,)

    ss_obs, f_obs = compute_ss_f_fast(z_obs.reshape(I, J, n))
    resid_obs = make_resid(z_obs)

    print(f"\n=== {resp} ===  obs SS={ss_obs:.2f}  F={f_obs:.4f}")

    for perm_type in ['raw', 'resid']:
        base = z_obs if perm_type == 'raw' else resid_obs
        # raw: rank(permute(y)) == permute(rank(y)) → z_obs를 직접 치환
        # resid: 잔차 벡터를 치환 (이미 rank-space)

        null_ss = np.empty(B)
        null_f  = np.empty(B)

        t0 = time.time()
        for b_idx in range(B):
            perm = rng.permutation(base).reshape(I, J, n)
            null_ss[b_idx], null_f[b_idx] = compute_ss_f_fast(perm)
            if (b_idx + 1) % 50_000 == 0:
                elapsed = time.time() - t0
                rate = (b_idx + 1) / elapsed
                eta  = (B - b_idx - 1) / rate
                print(f"  [{resp}/{perm_type}] {b_idx+1}/{B}  "
                      f"rate={rate:.0f}/s  ETA={eta:.0f}s", flush=True)

        b_ss = int(np.sum(null_ss >= ss_obs))
        b_f  = int(np.sum(null_f  >= f_obs))
        p_ss = (b_ss + 1) / (B + 1)
        p_f  = (b_f  + 1) / (B + 1)
        elapsed = time.time() - t0
        print(f"  [{resp}/{perm_type}] DONE {elapsed:.1f}s  "
              f"SS: b={b_ss} p={p_ss:.4e}  F: b={b_f} p={p_f:.4e}")

        np.save(NULL_DIR / f"{resp}_{perm_type}_SS.npy", null_ss)
        np.save(NULL_DIR / f"{resp}_{perm_type}_F.npy",  null_f)

        results.append({'response': resp, 'stat': 'SS', 'perm': perm_type,
                        'obs_stat': ss_obs, 'b': b_ss, 'p': p_ss, 'seed': SEED,
                        'nulldist': str(NULL_DIR / f"{resp}_{perm_type}_SS.npy")})
        results.append({'response': resp, 'stat': 'F', 'perm': perm_type,
                        'obs_stat': f_obs,  'b': b_f,  'p': p_f,  'seed': SEED,
                        'nulldist': str(NULL_DIR / f"{resp}_{perm_type}_F.npy")})

print(f"\n총 소요: {time.time()-t_start:.1f}s")

# ── JSON 저장 ──────────────────────────────────────────────────────────
json_out = OUT_DIR / 'permtest_results.json'
with open(json_out, 'w') as fh:
    json.dump(results, fh, indent=2)
print(f"Saved: {json_out}")

# ── Markdown 보고서 ────────────────────────────────────────────────────
lines = [
    "# 지시 83 — 독립 재구현 결과",
    "## HANDOFF v7.13 §2 | 2026-07-28",
    "",
    "## G0 게이트",
    "",
    "| fwhm | 관측 | 기대 | 통과 |",
    "|------|------|------|------|",
    "| 2.5  | [3, 2, 1, 1, 3] | [3, 2, 1, 1, 3] | OK |",
    "| 5.0  | [2, 4, 4, 2, 3] | [2, 4, 4, 2, 3] | OK |",
    "| 10.0 | [5, 5, 5, 3, 1] | [5, 5, 5, 3, 1] | OK |",
    "| 0.0  | [2, 3, 0, 0, 2] | [2, 3, 0, 0, 2] | OK |",
    "| 총   | 65/150 | 65/150 | OK |",
    "",
    "**G0 PASS** — 이후 계산 진행.",
    "",
    "## 구현 사양",
    "",
    f"- 입력: `{CSV_PATH}`",
    f"- B = {B:,}",
    f"- seed = {SEED}",
    "- 순위 변환: `scipy.stats.rankdata(method='average')` (midrank, 전역)",
    "- SS_AB: n × ∑_ij (z̄_ij − z̄_i. − z̄_.j + z̄)², n=5, df_AB=20, df_within=120",
    "- raw 치환: rank(permute(y)) ≡ permute(rank(y)) → z_obs 직접 치환",
    "- resid 치환: ter Braak 잔차 = z_ijk − (z̄_i. + z̄_.j − z̄) 벡터 치환",
    "- p값: (b+1)/(B+1)",
    "",
    "## 16개 조합 결과",
    "",
    "| 반응변수 | 통계량 | 치환 | 관측 통계량 | b | p | seed | 귀무분포 파일 |",
    "|---------|--------|------|------------|---|---|------|-------------|",
]
for r in results:
    lines.append(
        f"| {r['response']} | {r['stat']} | {r['perm']} | "
        f"{r['obs_stat']:.4f} | {r['b']} | {r['p']:.4e} | {r['seed']} | "
        f"`nulldist/{Path(r['nulldist']).name}` |"
    )

lines += [
    "",
    "## 참조값 대조 (§4)",
    "",
    "*(permtest_compare.md 참조)*",
]

md_out = OUT_DIR / 'permtest_replication.md'
with open(md_out, 'w') as fh:
    fh.write('\n'.join(lines) + '\n')
print(f"Saved: {md_out}")
