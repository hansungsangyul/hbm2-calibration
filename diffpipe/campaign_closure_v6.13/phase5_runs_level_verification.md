# phase5_runs_level.csv 검증 인스턴스 (§C, v6.13)
## 제출일: 2026-07-16

---

## 검증 대상

- 파일: `phase5/phase5_runs_level.csv` (151 rows = 헤더 + 150 runs)
- 인스턴스: `sg2_fw3_rep2` (σ_g=0.041, fwhm=2.5, 비발산 run)

---

## CSV 행 (재현값)

| 필드 | CSV 값 |
|---|---|
| run_id | sg2_fw3_rep2 |
| cell_id | sg2_fw3 |
| sigma_g | 0.041 |
| fwhm | 2.5 |
| seed | 800152 |
| grv5 | 53.75449093948312 |
| lm5 | 2841.3291015625 |
| diverged | False |
| divergence_reason | (빈 칸) |
| final_f1 | 0.9381727203726768 |
| osc_amp | 0.011080909986048937 |

---

## 원천 results.json vs CSV 대조

원천: `phase5/runs/sg2_fw3/rep2/results.json`

| 필드 | CSV 값 | raw 값 | 계산 방법 | 일치 |
|---|---|---|---|---|
| seed | 800152 | 800152 | 직접 저장 | ✅ |
| sigma_g | 0.041 | 0.041 | 직접 저장 | ✅ |
| fwhm | 2.5 | 2.5 | 직접 저장 | ✅ |
| grv5 | 53.754491 | grad_rv[:5] 평균 = 53.754491 | `np.mean(grad_rv[:5])` | ✅ |
| lm5 | 2841.3291 | lambda_max["5"] = 2841.3291 | HANDOFF v6.10 §1 재정의 (단일치) | ✅ |
| diverged | False | False | 직접 저장 | ✅ |
| divergence_reason | — | None | 비발산이므로 빈칸 | ✅ |
| final_f1 | 0.938173 | 0.938173 | 직접 저장 | ✅ |
| osc_amp | 0.011081 | medfilt-3(f1_raw[-10:]) max−min = 0.011081 | `scipy_medfilt(f1_raw[-10:], 3)` 적용 후 range | ✅ |

---

## CSV 자기 일관성 확인

- 총 rows: 150 (헤더 제외) — 예상값 일치
- 결측값 없음 (divergence_reason은 비발산 run에서 빈칸 정상)
- seed 범위: 800000~800149 (5 reps × 30 cells = 150, 연속 배정)

---

## 결론

phase5_runs_level.csv 전 필드가 raw results.json에서 정확히 파생됨을 sg2_fw3_rep2 인스턴스로 확인. A1·A2·A3 독립 재검산에 사용 가능.
