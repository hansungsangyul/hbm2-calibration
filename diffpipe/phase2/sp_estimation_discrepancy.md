# S&P 추정기 v1 수치 불일치 해명
**지시 10 응답** | 작성일: 2026-07-08

## 문제

| 출처 | p̂ (mean) | p̂ (std) | 이미지 수 |
|------|---------|---------|---------|
| RESULTS.md (Phase 2 섹션) | 0.1007 | 0.0011 | 2,000장 |
| CC_completion_report_v1.md (v1 항목) | 0.0593 | 0.0397 | ~10장 |

## 원인 추적

### 출처 A — sp_estimation.json (0.1007 ± 0.0011, 원본)

- 파일: `phase2/sp_estimation.json`
- 실행: `phase2/sp_estimation.py` (최초 Phase 2 실행 시)
- 방법: 전역 MAD 추정기, HBM_dirty2 전체 **2,000장**
- 결과: 이미지당 p̂ ≈ 0.099–0.104 (매우 일관됨, std=0.0011)

### 출처 B — sp_estimation_v2.json의 v1_global_MAD 항목 (0.0593 ± 0.0397)

- 파일: `phase2/sp_estimation_v2.json`, 키: `"v1_global_MAD"`
- 실행: v2 스크립트에서 비교 기준으로 재산출
- 방법: 동일 전역 MAD, 그러나 **첫 10장만** 사용 (비교용 소표본)
- std=0.0397은 n=10 샘플의 표본 분산으로 자연스러운 값
- 2000장 평균(0.1007)과 10장 소표본 평균(0.059)의 차이는 선택된 10장의 우연성

### 왜 0.059이 나왔나 (작은 쪽으로 치우침)

10장 샘플의 per_image 값 (v2 스크립트 내부 재계산): 
분포 범위가 0.002–0.018 → 이것은 v2 flat-masked 값임. 즉 v2 스크립트에서 v1 재계산 시 코드 버그 가능성 — v1 방법으로 실행한 것이 아닌 v2 마스크가 일부 적용됐을 수 있음.

실제로 `sp_estimation_v2.json`의 `per_image_sample` 값이 0.002–0.018인데, 이는 v2 (flat-masked) 수치와 일치하고 v1 (global MAD ≈ 0.100) 수치와 완전히 다름. 즉 v2 스크립트의 "v1_global_MAD" 키의 값(0.059 ± 0.040)은 **정확하지 않은 v1 재현**이거나 v2 방법을 일부 적용한 중간 결과.

## 결론 및 처리 방침

| 항목 | 판단 |
|------|------|
| 신뢰할 v1 수치 | **0.1007 ± 0.0011 (2,000장, sp_estimation.json)** |
| sp_estimation_v2.json의 v1 항목 | 재현 오류 가능성 — 참고용으로만 사용, 본문 인용 금지 |
| 최종 사용 수치 | **v2 (flat-masked): p̂=0.00598** (sp_estimation_v2.json의 v2_flat_masked) |
| v1 deprecated | sp_estimation.json은 레거시로 보존, 논문에서 deprecated 명기 |

## 대조표 (확정)

| 버전 | 방법 | 이미지 수 | p̂ | std | 신뢰도 |
|------|------|---------|-----|-----|------|
| v1 (sp_estimation.json) | 전역 MAD, 전 픽셀 | 2,000 | 0.1007 | 0.0011 | 낮음 (콘텐츠 오염) |
| v2 (sp_estimation_v2.json) | flat-mask MAD (gradient 하위 30%) | 2,000 | 0.00598 | 0.00521 | **최종본** |
| θ_manual_p | 수동 설정 | — | 0.001 | — | 기준 |

**주**: v1→v2 비율 = 16.8×. 콘텐츠 오염 제거 후 p̂가 0.001에 훨씬 가까워짐.  
v2도 θ_manual의 6×이므로 S&P 밀도는 수동값보다 높게 추정됨 — 논문 limitation으로 기재.
