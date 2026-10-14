# HANDOFF v7.8 §2 — 실험 2 재실행 보고서

**생성일**: 2026-07-25 02:13
**CSV**: exp2_results_v2.csv (600행)
**오버레이**: overlays_v2/ (calib+manual 대표 모델 × 5매 × 수정전/후)

## 전처리 수정 내역

| 조건 | 학습 prenorm | 추론 전처리 (v7→v8) |
|------|------------|-------------------|
| calib | ✓ | 원본→prenorm 수정 |
| manual | ✗ | 원본 유지 |
| ab1 | ✓ | 원본→prenorm 수정 |
| ab2 | ✗ | 원본 유지 |
| ab3 | ✗ | 원본 유지 |
| exp3 (σ_post) | ✓ | 원본→prenorm 수정 (신규 포함) |

## 결과

| 조건 | median_pixel_fpr | median_n_false |
|------|----------------|---------------|
| calib | 0.9815 | 1.0 |
| manual | 0.9812 | 1.0 |
| ab1 | 0.9821 | 1.0 |
| ab2 | 0.9812 | 1.0 |
| ab3 | 0.9812 | 1.0 |
| exp3 | 0.9822 | 1.0 |

**FPR 정합성 문턱 (< 0.5) 충족**: False

## Mann-Whitney 검정

| 비교 | calib/exp3 중앙값 | manual 중앙값 | p-value | 판정 |
|------|----------------|-------------|---------|------|
| calib vs manual | 1.0 | 1.0 | 1.0000 | **EQUIVALENT** |
| exp3  vs manual | 1.0 | 1.0 | 0.0827 | **EQUIVALENT** |

## 최종 판정

**FPR_ABOVE_THRESHOLD**
  → FPR ≥ 0.5: "합성 학습 모델의 실영상 일반화 실패" — §6 한계 승격

*HANDOFF v7.8 §2 완료*
