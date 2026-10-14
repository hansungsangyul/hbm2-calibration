# campaign_closure_v6.14 — 실험 캠페인 최종 산출물
## 기준: HANDOFF v6.14 §1·§2·§3 이행 완료 (2026-07-16) | 커밋: 8042b71

---

## 파일 목록

| 파일 | 역할 |
|---|---|
| `CC_diagnostic_report_v6.12.md` | **메인 진단 보고서** — §1~§7 전체. §7에 §4-A(P4.2r v2) 사후 판정 결과 전체 수록 |
| `p42r_final_report_v2.md` | **Table 3 원천 문서** — 40 runs 발산율·성능·통계 검정 결과. spread 열 v6.14 정정 완료 |
| `p42r_v2_perclass_f1.csv` | **per-class F1 원자료** — calib/manual × 20 runs × 3 testsets = 120 rows |
| `phase5_final_report.md` | **Phase 5 그리드 보고서** — σ_g × fwhm 30셀 × 5 reps. B-3 프레이밍 정정(v6.14) 포함 |
| `phase5_runs_level.csv` | **Phase 5 run-level CSV** — 150 rows. A1·A2·A3 통계 독립 재검산 원천 |
| `phase5_runs_level_verification.md` | **검증 인스턴스** — sg2_fw3_rep2 전 필드 raw JSON 대조 확인 |
| `p42r_v2_runs/` | **40 runs raw 데이터** — calib/manual 각 20 run의 results.json (symlink) |

---

## 핵심 결과

| 항목 | 결과 |
|---|---|
| 발산율 (calib / manual) | 5/20 (25%) / 4/20 (20%) — v1 대비 −35%p / −80%p |
| TOST δ=0.004 | p=0.6203 → **동등성 미확인** |
| Mann-Whitney U | p=0.0013 → manual val F1 유의하게 높음 |
| TEST_ID | manual +0.051 |
| TEST_OOD | calib +0.191 (p<0.0001) |
| TEST_CAL | calib +0.331 (p<0.0001) |
| 발산율 차이 | Fisher p=1.00 → 비유의 |
| 사전 등록 예측 | 2/2 적중 |
| **최종 판정** | 동등성 기각. 트레이드오프 구조 |

---

## 정정 이력 요약

| HANDOFF | 항목 | 내용 |
|---|---|---|
| v6.13 §A | Welch SE 정정 | 0.01529 → 0.0211 |
| v6.13 §B-1 | 발산 정의 재규정 | 사후 판정 아님 — 롤링 창 온라인 판정+break |
| v6.13 §B-3 | 베이스 구성 명기 | 신 θ + prenorm=True |
| v6.14 §1 | spread 열 통일 | raw 범위 혼용 제거 → 전체 궤적 medfilt 기준으로 전면 재계산. 불일치 12건 정정 |
| v6.14 §2 | B-3 프레이밍 정정 | "문서화 오류" → "v6.1 §7.2 사양으로부터의 이탈" |
| v6.14 §3 | CSV 제출 | phase5_runs_level.csv 추가 |
