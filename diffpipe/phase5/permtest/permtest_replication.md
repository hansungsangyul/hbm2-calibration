# 지시 83 — 독립 재구현 결과
## HANDOFF v7.13 §2 | 2026-07-28

## G0 게이트

| fwhm | 관측 | 기대 | 통과 |
|------|------|------|------|
| 2.5  | [3, 2, 1, 1, 3] | [3, 2, 1, 1, 3] | OK |
| 5.0  | [2, 4, 4, 2, 3] | [2, 4, 4, 2, 3] | OK |
| 10.0 | [5, 5, 5, 3, 1] | [5, 5, 5, 3, 1] | OK |
| 0.0  | [2, 3, 0, 0, 2] | [2, 3, 0, 0, 2] | OK |
| 총   | 65/150 | 65/150 | OK |

**G0 PASS** — 이후 계산 진행.

## 구현 사양

- 입력: `/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/campaign_closure_v6.14/phase5_runs_level.csv`
- B = 200,000
- seed = 42
- 순위 변환: `scipy.stats.rankdata(method='average')` (midrank, 전역)
- SS_AB: n × ∑_ij (z̄_ij − z̄_i. − z̄_.j + z̄)², n=5, df_AB=20, df_within=120
- raw 치환: rank(permute(y)) ≡ permute(rank(y)) → z_obs 직접 치환
- resid 치환: ter Braak 잔차 = z_ijk − (z̄_i. + z̄_.j − z̄) 벡터 치환
- p값: (b+1)/(B+1)

## 16개 조합 결과

| 반응변수 | 통계량 | 치환 | 관측 통계량 | b | p | seed | 귀무분포 파일 |
|---------|--------|------|------------|---|---|------|-------------|
| diverged | SS | raw | 37125.0000 | 26391 | 1.3196e-01 | 42 | `nulldist/diverged_raw_SS.npy` |
| diverged | F | raw | 1.6230 | 11786 | 5.8935e-02 | 42 | `nulldist/diverged_raw_F.npy` |
| diverged | SS | resid | 37125.0000 | 6950 | 3.4755e-02 | 42 | `nulldist/diverged_resid_SS.npy` |
| diverged | F | resid | 1.6230 | 11776 | 5.8885e-02 | 42 | `nulldist/diverged_resid_F.npy` |
| final_f1 | SS | raw | 28285.4533 | 159427 | 7.9714e-01 | 42 | `nulldist/final_f1_raw_SS.npy` |
| final_f1 | F | raw | 2.4149 | 381 | 1.9100e-03 | 42 | `nulldist/final_f1_raw_F.npy` |
| final_f1 | SS | resid | 28285.4533 | 151 | 7.6000e-04 | 42 | `nulldist/final_f1_resid_SS.npy` |
| final_f1 | F | resid | 2.4149 | 382 | 1.9150e-03 | 42 | `nulldist/final_f1_resid_F.npy` |
| osc_amp | SS | raw | 44810.7867 | 49349 | 2.4675e-01 | 42 | `nulldist/osc_amp_raw_SS.npy` |
| osc_amp | F | raw | 1.4436 | 23376 | 1.1688e-01 | 42 | `nulldist/osc_amp_raw_F.npy` |
| osc_amp | SS | resid | 44810.7867 | 15093 | 7.5470e-02 | 42 | `nulldist/osc_amp_resid_SS.npy` |
| osc_amp | F | resid | 1.4436 | 23115 | 1.1558e-01 | 42 | `nulldist/osc_amp_resid_F.npy` |
| grv5 | SS | raw | 5704.4933 | 200000 | 1.0000e+00 | 42 | `nulldist/grv5_raw_SS.npy` |
| grv5 | F | raw | 1.1230 | 67044 | 3.3522e-01 | 42 | `nulldist/grv5_raw_F.npy` |
| grv5 | SS | resid | 5704.4933 | 51617 | 2.5809e-01 | 42 | `nulldist/grv5_resid_SS.npy` |
| grv5 | F | resid | 1.1230 | 66936 | 3.3468e-01 | 42 | `nulldist/grv5_resid_F.npy` |

## 참조값 대조 (§4)

*(permtest_compare.md 참조)*
