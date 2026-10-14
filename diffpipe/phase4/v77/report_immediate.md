# HANDOFF v7.7 즉시군 통합 보고서

**생성일**: 2026-07-22 09:20:32  
**포함 실험**: 실험 1(추론 정규화), 실험 2(실영상 FPR), 실험 5(ISO 프로파일), 실험 7(ESF 완화 민감도)

---

## 실험 1 — 추론 시 정규화 재평가 (M4)

**판정표 (HANDOFF v7.7 §1)**:
- gap = calib_ID − manual_ID (정규화 모드별)
- gap > −0.01  → **ID_RESOLVED** (ID 분포이동 문제 없음)
- −0.03 < gap ≤ −0.01 → **ID_PARTIAL** (부분 트레이드오프)
- gap ≤ −0.03 → **TRADEOFF_REAL** (실제 트레이드오프)

| 정규화 모드 | calib ID F1 | manual ID F1 | gap | 판정 |
|------------|-------------|--------------|-----|------|
| orig     | 0.8757 | 0.9269 | -0.0512 | **TRADEOFF_REAL** |
| zscore   | 0.2001 | 0.1849 | 0.0152 | **ID_RESOLVED** |
| prenorm  | 0.9120 | 0.6021 | 0.3098 | **ID_RESOLVED** |

**핵심 결론**: prenorm 모드 판정 = **ID_RESOLVED**
  → "Self-Calibrating 프레임 유지: prenorm 추론 시 ID 분포이동 문제 없음"

---

## 실험 2 — 실영상 무결함 위양성률 (M5)

**판정표 (HANDOFF v7.7 §2)**:
- Mann-Whitney U, calib vs manual (n_false_components)
- calib < manual & p<0.05  → **CALIB_LOWER** (캘리브 모델이 FPR 낮음)
- p ≥ 0.05                 → **EQUIVALENT** (차이 없음)
- calib > manual & p<0.05  → **CALIB_HIGHER** (캘리브 모델이 FPR 높음)

| 지표 | calib (mean) | manual (mean) | p-value | 판정 |
|------|--------------|---------------|---------|------|
| pixel_fpr | 0.9812 | 0.9812 | — | — |
| n_false_comp | 1.00 | 1.00 | 0.1583 | **EQUIVALENT** |

**핵심 결론**: **EQUIVALENT**

---

## 실험 5 — ISO 프로파일 (M6)

**판정표 (HANDOFF v7.7 §5)**:
- fwhm(σ_g=0.0205) < fwhm(σ_g=0.041) → **MONOTONE** (ISO 단조성 확인)
- 역전 또는 차이 없음 → **FLAT** (ISO 단조성 불확인)

| 항목 | 값 |
|------|-----|
| version | exp5_v77 |
| experiment | 5_iso_profile |
| per_f | [{'fwhm': 1.4, 'rep_losses': [0.09416386485099792, 0.09664811193943024, 0.09684543311595917], 'mean_loss': 0.0958858033021291, 'std_loss': 0.0012202562391535473}, {'fwhm': 1.6, 'rep_losses': [0.06744217872619629, 0.06928392499685287, 0.07006903737783432], 'mean_loss': 0.0689317137002945, 'std_loss': 0.0011009499510237798}, {'fwhm': 1.8, 'rep_losses': [0.058723971247673035, 0.060781124979257584, 0.06173645332455635], 'mean_loss': 0.06041384985049566, 'std_loss': 0.001256962033645971}, {'fwhm': 2.0, 'rep_losses': [0.05839599296450615, 0.06075991690158844, 0.06213575601577759], 'mean_loss': 0.06043055529395739, 'std_loss': 0.0015444127732679382}, {'fwhm': 2.2, 'rep_losses': [0.06285549700260162, 0.06546767801046371, 0.06746681034564972], 'mean_loss': 0.06526332845290501, 'std_loss': 0.0018880981159532253}] |
| spread_metric | {'max_mean_loss': 0.0958858033021291, 'min_mean_loss': 0.06041384985049566, 'relative_spread': 0.587149362916412, 'threshold': 0.25} |

**핵심 결론**: **F_COUPLING_WEAK**

---

## 실험 7 — ESF 완화 기준 민감도 (M9, CPU)

**판정표 (HANDOFF v7.7 §7)**:
- R²·min_fits 완화 전 기준에서 f_ESF ∈ [1.9, 2.3] → **ROBUST**
- 이탈 기준 존재 → **SENSITIVE** + 이탈 조건 목록

| 항목 | 값 |
|------|-----|
| 기준 FWHM (R²=0.95, min_fits=20) | 2.0569 |
| 허용 범위 | [1.9, 2.3] |
| 범위 내 조건 수 | 0/9 |
| 판정 | **SENSITIVE** |

이탈 조건:
  - R²=0.9, min_fits=20: f_ESF=1.8908
  - R²=0.9, min_fits=10: f_ESF=1.8350
  - R²=0.9, min_fits=5: f_ESF=1.8350
  - R²=0.85, min_fits=20: f_ESF=1.8036
  - R²=0.85, min_fits=10: f_ESF=1.8036
  - R²=0.85, min_fits=5: f_ESF=1.8036
  - R²=0.8, min_fits=20: f_ESF=1.7749
  - R²=0.8, min_fits=10: f_ESF=1.7749
  - R²=0.8, min_fits=5: f_ESF=1.7749

**핵심 결론**: **SENSITIVE**
  → "ESF 민감도 존재 — 이탈 조건 검토 필요"

---

## 즉시군 종합 판정 요약

| 실험 | 상태 | 판정 | 의미 |
|------|------|------|------|
| Exp1 (정규화 재평가) | 완료 | ID_RESOLVED | prenorm 추론 시 ID 분포이동 |
| Exp2 (실영상 FPR)    | 완료 | EQUIVALENT | calib 모델 위양성 비교 |
| Exp5 (ISO 프로파일) | 완료 | F_COUPLING_WEAK | FWHM 단조성 확인 |
| Exp7 (ESF 민감도)   | 완료 | SENSITIVE | ESF 측정 신뢰도 |

---
*이 보고서는 자동 생성됩니다. PENDING 항목은 해당 실험 완료 후 재실행 하십시오.*
