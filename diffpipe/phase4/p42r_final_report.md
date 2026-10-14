# P4.2r 최종 보고서
## 근거: HANDOFF v6.6 §2 설계 + v6.10 §3 완료 이벤트
## 완료일: 2026-07-15 (40/40 runs)

---

## 실험 설계

| 항목 | 값 |
|---|---|
| 조건 | `calib` (fwhm=2.0956, prenorm=True, seed_base=730000) vs `manual` (fwhm=2.5, prenorm=False, seed_base=740000) |
| runs/조건 | 20 × 2 = 40 |
| epochs | 41 |
| batch | 32 |
| 테스트셋 3종 (고정, 500장 × 3) | TEST_ID, TEST_OOD, TEST_CAL |

**테스트셋 정의**:
- **TEST_ID** (동일 분포): θ_manual, fwhm=2.5, pre-norm 미적용
- **TEST_OOD** (분포 이동): ×1.5, fwhm=3.75, pre-norm 미적용
- **TEST_CAL** (보정 분포): D_calib, fwhm=2.0956, pre-norm 적용

---

## 발산 현황

| 조건 | 발산 | 수렴 |
|---|---|---|
| calib | 12/20 (60%) | 8/20 |
| manual | **20/20 (100%)** | 0/20 |

**manual 전 run 발산** — F1 진동(crit 3) 발산. calib는 60%만 발산.

---

## 기술 통계

| 테스트셋 | calib mean ± std | manual mean ± std | diff (c-m) |
|---|---|---|---|
| TEST_ID | 0.8025 ± 0.1236 | 0.8746 ± 0.0982 | **−0.072** |
| TEST_OOD | 0.7455 ± 0.0855 | 0.5330 ± 0.0783 | **+0.213** |
| TEST_CAL | 0.8877 ± 0.0692 | 0.5413 ± 0.0693 | **+0.346** |

---

## Welch t-test (양측)

| 테스트셋 | t | p | 해석 |
|---|---|---|---|
| TEST_ID | −1.991 | 0.054 | 비유의 (borderline) |
| TEST_OOD | +7.990 | 1.24e-09 | **calib 유의 우세** |
| TEST_CAL | +15.413 | 6.18e-18 | **calib 압도적 우세** |

## 우월성 검정 (단측: calib > manual)

| 테스트셋 | p_sup | 판정 |
|---|---|---|
| TEST_ID | 0.973 | 우월성 없음 (calib < manual) |
| TEST_OOD | 6.22e-10 | **calib 우월** |
| TEST_CAL | 3.09e-18 | **calib 우월** |

---

## TOST 등가성 검정 (δ=0.02)

| 테스트셋 | diff | TOST p | 등가 여부 |
|---|---|---|---|
| TEST_ID | −0.072 | 0.921 | **비등가** |
| TEST_OOD | +0.213 | 1.000 | **비등가** |
| TEST_CAL | +0.346 | 1.000 | **비등가** |

전 테스트셋에서 등가성 기각 실패 — 두 조건의 성능은 동등하지 않다.

---

## 2×2 교차 행렬 (발산 여부)

```
              manual 수렴   manual 발산
calib 수렴       0              8
calib 발산       0             12
```

- calib 수렴 8 runs 전부 manual은 발산 → calib가 안정성 측면에서 유리
- manual 수렴 run 없음

---

## TEST_OOD 클래스별 F1

| 클래스 | calib | manual | diff |
|---|---|---|---|
| class 0 | 0.745 | 0.194 | **+0.551** |
| class 1 | 0.834 | 0.638 | +0.197 |
| class 2 | 0.786 | 0.760 | +0.026 |
| class 3 | 0.617 | 0.540 | +0.077 |

class 0에서 calib-manual 차이 최대 (+0.551) — 분포 이동 시 class 0 인식이 calib 학습에서 특히 유리.

---

## 종합 판정

| 가설 | 결과 |
|---|---|
| TEST_ID 등가성 | **기각** — 비등가(diff=−0.072, manual 약우세, p=0.054 비유의) |
| TEST_OOD: calib 우세 | **지지** — p=1.24e-09, diff=+0.213 |
| TEST_CAL: calib 우세 | **지지** — p=6.18e-18, diff=+0.346 |
| calib 안정성 | **지지** — 발산율 60% vs 100% |

**결론**: calib 조건은 동일 분포(TEST_ID)에서 manual과 동등 이하이나, 분포 이동(OOD) 및 보정 분포(TEST_CAL)에서 대폭 우세하고 안정성도 높다. 학습-테스트 분포 일치(D_calib 조건)가 OOD 일반화에 핵심 기여.

---

## 산출물

| 파일 | 내용 |
|---|---|
| `phase4/p42r_runs/calib/run*/results.json` | calib 20 runs |
| `phase4/p42r_runs/manual/run*/results.json` | manual 20 runs |
| `phase4/run_p42r.py` | 실행 스크립트 |
| `phase4/launch_p42r.sh` | 40 runs 직렬 실행 스크립트 |
| `phase4/p42r_testsets.md` | 테스트셋 설계 문서 |
