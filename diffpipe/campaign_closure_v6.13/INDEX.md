# campaign_closure_v6.13 — 이번 세션 산출물 모음
## 생성일: 2026-07-16 | 근거 커밋: 5d7452b, 0bfd255

---

## 배경

§4-A (P4.2r v2, calib/manual 각 20 run) GPU 계산 완료 후 HANDOFF v6.13의 §A·§B·§C를
전부 이행하여 실험 캠페인 공식 종결. 아래 파일이 이번 세션에서 신규 생성 또는 수정됨.

---

## 파일 목록

| 파일 | 유형 | 역할 |
|---|---|---|
| `CC_diagnostic_report_v6.12.md` | 수정 (§7 추가) | 메인 진단 보고서. §7.0~§7.9에 §4-A 사후 판정 전 결과 수록 |
| `p42r_final_report_v2.md` | 신규 (§A) | Table 3 원천 문서. A-1~A-5 보완 5건·확정 결론문 포함 |
| `p42r_v2_perclass_f1.csv` | 신규 (§A) | 조건·run·테스트셋별 per-class F1 (120 rows) |
| `phase5_final_report.md` | 수정 (§B) | Phase 5 최종 보고서. B-1~B-4 정정 묶음 반영 |
| `phase5_runs_level_verification.md` | 신규 (§C) | sg2_fw3_rep2 검증 인스턴스 (전 필드 raw JSON 대조) |
| `phase5_runs_level.csv` | 신규 (§3) | Phase 5 run-level CSV (151줄: 헤더+150 rows) — A1·A2·A3 독립 재검산 원천 |
| `p42r_v2_runs/` | 링크 → phase4/p42r_v2_runs | 40 runs raw 데이터 (calib 20 + manual 20, 각 results.json) |

---

## 핵심 결과 요약

- **발산율**: calib 5/20(25%), manual 4/20(20%) — v1(60%/100%) 대비 대폭 감소
- **TOST δ=0.004**: p=0.6203 → 동등성 미확인
- **Mann-Whitney U=81**: p=0.0013 → manual이 val F1 기준 유의하게 높음
- **테스트셋 분리**: manual TEST_ID +0.051, calib TEST_OOD +0.191 / TEST_CAL +0.331
- **Fisher exact p=1.00**: 발산율 차이 비유의
- **확정 판정**: 동등성 기각. 트레이드오프 구조. 사전 등록 예측 2/2 적중.

---

## 정정 이력 (이번 세션)

| 항목 | 내용 |
|---|---|
| Welch SE 표기 | "pooled SE 0.01529" → SE=0.0211 (A-1) |
| 발산 정의 | 사후 판정 → 롤링 창 온라인 판정+break 재규정 (B-1) |
| run별 TEST_OOD/CAL 값 | 추정치 → 실측값으로 전체 교체 |
| Phase 5 베이스 구성 | 신θ+prenorm=True 명기, G2★/G4▲ 캐비앳 갱신 (B-3) |
