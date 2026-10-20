# 지시 82 — 원본 분석 코드 회수 결과
## HANDOFF v7.13 §1 | 2026-07-28

---

## 원본 스크립트 위치

```
diffpipe/phase4/analysis_mc3_stats.py
  (MC3 통계 보강 재분석, HANDOFF v7.6 §3.1 지시 69)
  입력: phase5/phase5_runs_level.csv
  출력: phase4/reanalysis_mc3_mc5.md
```

---

## 5개 항목 (코드 인용 포함)

### 1. 반응변수

**`osc_amp`** (발진 진폭). `diverged`, `final_f1` 은 이 검정에서 사용되지 않음.

```python
# 코드 인용 (analysis_mc3_stats.py:101-105)
# ── (a3) Interaction: osc_amp의 f×σ_g 순열 상호작용
print("\n[a3] Interaction 검정: osc_amp의 fwhm×sigma_g 순열 상호작용")
df_ia = df.copy()
df_ia['rank_osc'] = stats.rankdata(df_ia.osc_amp)
```

---

### 2. 순위 변환

`scipy.stats.rankdata(df_ia.osc_amp)` — 전체 150 run에 대해 일괄 적용.
셀 내부가 아니라 **전체 데이터셋**에 적용. `rankdata` default method='average' = **midrank** (동점은 평균 순위).

이후 셀별 평균을 구함:
```python
# 코드 인용 (analysis_mc3_stats.py:112-117)
cell_stats = df_ia.groupby('cell_id').agg(
    sigma_g=('sigma_g', 'first'),
    fwhm=('fwhm', 'first'),
    mean_rank=('rank_osc', 'mean'),   # 셀 내 run들의 순위 평균
    n=('rank_osc', 'count'),
).reset_index()
```

---

### 3. 검정 통계량

셀 평균 rank의 **비가중 SS_AB** — `n_ij` 를 곱하지 않음.

```python
# 코드 인용 (analysis_mc3_stats.py:123-131)
def interaction_stat(df_c):
    pivot = df_c.pivot_table(values='mean_rank', index='sigma_g', columns='fwhm', aggfunc='mean')
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return float('nan')
    row_means = pivot.mean(axis=1)
    col_means = pivot.mean(axis=0)
    grand     = pivot.values.mean()
    inter     = pivot.values - row_means.values[:,None] - col_means.values[None,:] + grand
    return float(np.nansum(inter**2))   # ← n_ij 없이 inter² 합산
```

**주의**: HANDOFF §2.2 사양의 `∑ n_ij (z̄_ij − z̄_i. − z̄_.j + z̄)²` 와 달리 n_ij 가중치가 없음.
이 차이가 관측 SS 값(원본 8962.16 vs. 사양 기반 독립 구현)에 영향을 줌.

---

### 4. 치환 방식

셀 단위 fwhm 라벨 치환 — **전역 귀무 근사**. 주효과 제거 잔차 치환이 아님.

```python
# 코드 인용 (analysis_mc3_stats.py:119-138)
# Permutation test: under H0 (no interaction), shuffle fwhm assignments
N_PERM = 5000
np.random.seed(42)
...
perm_stats = []
for _ in range(N_PERM):
    shuffled = cell_stats.copy()
    shuffled['fwhm'] = np.random.permutation(cell_stats['fwhm'].values)  # 25셀 fwhm 라벨 치환
    perm_stats.append(interaction_stat(shuffled))
```

25개 셀(5 σ_g × 5 fwhm, fwhm=0 제외 가능성)의 fwhm 값을 치환.
원 반응변수 전체 치환(run 150개)이 아니라 **셀 집계 후 fwhm 라벨 치환**임.

---

### 5. p값 산출식

```python
# 코드 인용 (analysis_mc3_stats.py:139-142)
perm_stats = [s for s in perm_stats if not np.isnan(s)]   # NaN 제거
pval_ia = float(np.mean([s >= obs_stat for s in perm_stats]))
print(f"  Obs interaction SS (rank-transform) = {obs_stat:.2f}")
print(f"  Permutation p (N={N_PERM}) = {pval_ia:.4f}")
```

**`b / len(valid_perms)`** 방식 (`(b+1)/(B+1)` 아님).
- b = 관측 통계량 이상인 치환 수
- 분모 = NaN 제거 후 유효 치환 수 (명목 5000 이하)
- **b=0이면 p=0.0이 출력 가능** — 원고 "p < 10^{-4}" 표기의 원인

---

## 원본 스크립트 실제 출력

출처: `diffpipe/phase4/reanalysis_mc3_mc5.md`

```
#### (a3) 상호작용 검정 (순열 기반, 순위 변환)

- 관측 상호작용 SS (rank-transform) = 8962.16
- 순열 p 값 (N = 5000) = 0.0000
```

---

## 원고 기재와의 관계

원고 §5.2: `p < 10^{-4} (N_perm = 5000, rank-transform SS)`

- `p = 0.0000` → `b/B = 0/5000 = 0` → 원고에서 `< 10^{-4}` 로 표기
- **이 표기가 성립하는 이유**: 원본이 `(b+1)/(B+1)` 이 아닌 `b/B` 사용.
  `b/B` 방식에서 b=0이면 p=0.0으로 출력 가능.
- HANDOFF §0의 "셋 중 하나" 중 **3번** 해당: p값 산출식이 `b/B`
- 단, `b/B` 방식은 보수성 없음 (유효 p의 하한이 0으로 떨어짐).
  통계 관행상 `(b+1)/(B+1)` 권장 (Phipson & Smyth 2010).

---

## 원본 구현과 §2.3 독립 구현의 주요 차이점

| 항목 | 원본 | §2.3 독립 구현 |
|------|------|----------------|
| 반응변수 | osc_amp 1개 | 4개 (diverged, final_f1, osc_amp, grv5) |
| SS_AB 가중 | n_ij 없음 | n_ij 가중 |
| 치환 단위 | 셀 집계 후 fwhm 라벨 | run 수준 전체 치환 / 잔차 치환 |
| p값 | b/B (+1 없음) | (b+1)/(B+1) |
| B | 5000 | 200,000 |
| 시드 | np.random.seed(42) | 고정 (보고서 참조) |
