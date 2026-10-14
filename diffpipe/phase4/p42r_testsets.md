# P4.2r 테스트셋 명세 (평가 프로토콜 1페이지)
## HANDOFF v6.6 §2.1 | 생성일: 2026-07-12

---

## 개요

P4.2r (재실행) 평가를 위한 사전 고정 테스트셋 3종.
**온-더-플라이 열화 금지** — ε 시드 고정으로 1회 생성, 전 모델이 동일 파일로 평가.

---

## 테스트셋 명세

| 셋 | 생성 파이프라인 | pre-norm | fwhm | 용도 |
|---|---|---|---|---|
| TEST_ID | θ_manual 전체 | **미적용** | 2.5 | D_manual 주 평가, 스펙 v1.0 §4.2 사전 등록 |
| TEST_OOD | 전작 R1-3 ×1.5 | **미적용** | 3.75 | OOD 강건성 평가 |
| TEST_CAL | D_calib 파이프라인 | **적용** | 2.0956 | D_calib 주 평가, 교차 행렬 부 평가 |

### θ_manual (TEST_ID 기준)
| 파라미터 | 값 |
|---|---|
| N | 300.0 |
| sigma_g | 0.0408 |
| p | 0.001 |
| fwhm | 2.5 |
| alpha | 0.35 |
| in_low / in_high | 0.05 / 0.90 |
| out_low / out_high | 0.10 / 0.90 |
| ring_strength / ring_n | 0.12 / 8.0 |
| sigma_post | 0.0 |

### TEST_OOD pre-norm 미적용 근거 (HANDOFF v6.7 §2 확정)

테스트셋은 **배포 환경 대리물(deployment proxy)**이다. 실제 추론 시점의 입력은 카메라/스캐너에서 획득한 이미 열화된 실이미지이며, pre-norm은 *합성 열화 생성 단계의 내부 장치*로서 배포 환경에 존재하지 않는다. 따라서 TEST_OOD에 pre-norm을 적용하지 않는 것이 배포 조건의 충실한 모사(faithful simulation)다.

D_calib 모델이 TEST_OOD로 평가될 때 발생하는 pre-norm 불일치는 제거 대상이 아니라 **"캘리브레이션 절차가 감수하는 실측 비용"(calibration-procedure cost)** — 즉 OOD 강건성 분석의 측정 대상이다. TEST_CAL(pre-norm 적용)이 교차 평가 행렬에서 이 효과를 D_calib in-distribution 조건과 분리한다.

### TEST_OOD 변경분 (나머지 θ_manual 동일)
| 파라미터 | 원값 | OOD값 | 배율 |
|---|---|---|---|
| fwhm | 2.5 | **3.75** | ×1.5 |
| sigma_g | 0.0408 | **0.0612** | ×1.5 |
| ring_strength | 0.12 | **0.18** | ×1.5 |

### TEST_CAL 변경분 (나머지 θ_manual 동일)
| 파라미터 | 값 |
|---|---|
| fwhm | **2.0956** (ESF 보정, HANDOFF v6.3) |
| pre-norm | **적용** (A=0.501834, B=0.173320) |
| sigma_post | 0.0 |

---

## 소스 데이터

| 항목 | 값 |
|---|---|
| 소스 | `test_data/HBM_clean/` (원본 클린 이미지 500장, indices 1501-2000) |
| 라벨 | `test_data/HBM_labels/` (기존 JSON 파일 참조, 변경 없음) |
| 규모 | **500장** (HANDOFF 명세 1000장 vs 실제 test_data 500장 불일치 — 아래 참조) |

**규모 확정 (HANDOFF v6.7 §1 승인)**: 500장 사용. val_data(500)는 P4.2r 학습 중 F1 모니터링에 사용되므로 테스트셋 오염 방지를 위해 제외. test_data 500장으로 고정.

전작 R1-3 "1000장" 확인 결과 (v6.7 §1 §2 조사): 전작 `gen_canonical.py`의 `NTEST=1000`은 클린 이미지 500장 × ε 2 실현이 아닌, 서로 다른 템플릿에서 생성된 1000장 독립 이미지. 동일 방식 재생성 불가 (전작 클린 이미지 템플릿 미공유). **500장 유지 확정. 논문에 n=500 명기.**

---

## 생성 세부

| 항목 | 값 |
|---|---|
| ε 시드 기저 | TEST_ID=500000, TEST_OOD=510000, TEST_CAL=520000 |
| 이미지당 시드 | `seed_base + i` (i = 이미지 정렬 순서, 0-499) |
| 전처리 | `cv2.copyMakeBorder(img, 3,3,0,0) → [:, 4:516]` (훈련과 동일) |
| 저장 형식 | `.npy` (float32, shape [H_crop, W_crop]) |
| 생성 스크립트 | `phase4/generate_p42r_testsets.py` |

---

## 출력 경로

```
phase4/p42r_testsets/
  test_id/
    image_{N}.npy   (500개)
    meta.json
  test_ood/
    image_{N}.npy   (500개)
    meta.json
  test_cal/
    image_{N}.npy   (500개)
    meta.json
  manifest.json
```

---

## 평가 프로토콜 (사전 등록)

### 주 분석 (스펙 v1.0 §4.3)
- 테스트셋: **TEST_ID** (공통 고정, 스펙 준수)
- 지표: mean F1 (Crack/Void/Open/Background macro-average)
- 검정: Welch TOST (δ=0.004, α=0.05) + Welch 우월성 검정
- 가설: TEST_ID에서 D_manual 분포 일치 이점 → 동등 또는 D_manual 근소 우위 예상

### 부 분석 (교차 행렬)
| | TEST_ID | TEST_CAL |
|---|---|---|
| D_calib 모델 (pre-norm학습) | 교차 평가 | In-distribution |
| D_manual 모델 (no pre-norm) | In-distribution | 교차 평가 |

→ pre-norm 효과 vs fwhm 보정 효과 분리

### OOD 분석
- 테스트셋: **TEST_OOD** (×1.5 강도)
- 가설: D_calib의 fwhm 보정이 강건성 효과를 가지면 TEST_OOD에서 격차 발현 예상

### per-class F1
- Crack / Void / Open / Background 클래스별 F1 수집 의무
- run별 results.json에 `f1_by_class` 키로 저장

---

## 학습 사양 (P4.2r)

| 항목 | 값 |
|---|---|
| 조건 | D_calib (pre-norm + fwhm 2.0956) vs D_manual (no pre-norm, fwhm 2.5) |
| runs | 20 + 20 |
| seeds | calib 730000~730019 / manual 740000~740019 |
| batch_size | **32** (전작 G4 준수). OOM 시 16으로 낮추고 양 조건 동일 적용 |
| epochs | 41 |
| 저장 의무 | final.pth + per-class F1 + mean F1 + results.json |
| 착수 조건 | Phase 5 완료 후 |

---

*검증 인스턴스 확인: testsets.md 커밋 보고로 갈음 (HANDOFF v6.6 §2.1)*
