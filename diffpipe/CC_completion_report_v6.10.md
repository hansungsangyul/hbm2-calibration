# CC 완료 보고서 v6.10
## 근거: HANDOFF v6.10 §1 (run-level CSV) + §2 (보고서 §2.1-§2.4) + P4.2r 완료
## 최초: 2026-07-14 | 갱신: 2026-07-15

---

## 요약

| 항목 | 상태 | 산출물 |
|---|---|---|
| Phase 5 최종 보고 (150 runs) | **완료** | `phase5/phase5_final_report.md` + `phase5/phase5_final_150runs.csv` |
| heatmap 4종 | **완료** | Artifact (https://claude.ai/code/artifact/764d1ae9-6675-4311-bd54-286301428c6d) |
| A1–A3 사전 등록 분석 | **완료** | 보고서 내 |
| §1 run-level CSV | **완료** | `phase5/phase5_runs_level.csv` (150행) |
| §1 자기 일관성 검사 | **전 통과** | `phase5/build_runs_level.py` |
| §2.1-§2.4 추가 해석 | **완료** | `phase5/phase5_final_report.md` 추가 해석 절 |
| P4.2r (40/40 runs) | **완료** | `phase4/p42r_final_report.md` |

---

## Phase 5 핵심 결과

| 항목 | 값 |
|---|---|
| 완료 runs | 150 / 150 |
| 총 발산 | **65 / 150** (43.3%) |
| 발산 유형 | **전부 crit 3** (F1 진동) — crit 1/2 **0건** |
| G4 파일럿 셀 (sg2_fw3) | 1/5 발산, F1=0.890 |

### A3 — blur 축 단조성

| 지표 | 결과 |
|---|---|
| GRV Spearman r (각 σ_g 행) | **1.000** (전 5행) |
| Pooled GRV r, p | 0.878, **p=4.0e-41** (n=125) |
| Pooled λ_max r, p | 0.877, **p=5.8e-41** |

### A1 — 사전 예측 가능성

| 지표 | r_pb | p | AUC |
|---|---|---|---|
| GRV 초기5ep (연속값) | 0.376 | 2.1e-06 | **0.684** |
| λ_max (ep5+ep10 평균) | 0.397 | 5.0e-07 | — |

**§1.3 A1 정의**: AUC=0.684의 예측 변수는 grv5 연속값 (`roc_auc_score(diverged, grv5)`). λ_max r_pb=0.397은 lm5_mean(ep5+ep10 평균) 기준 — run-level CSV의 lm5는 ep5 단일치(HANDOFF v6.10 §1 재정의).

### A2 — 경계 불연속성

fwhm=1.25 vs 5.0: **4/5 σ_g 행** MW p < 0.05 유의.

---

## §1 자기 일관성 검사 통과 로그

```
[OK] 행 수: 150 == 150
[OK] cell_id 30종 × 각 5행
[OK] seed 150개 모두 고유
[OK] diverged ↔ divergence_reason 상호 일치
[OK] run→셀 집계 재생성 전 수치 일치 (30 cells × 9 항목)
[PASS] 전 검사 통과
```

---

## §2 추가 해석 4건

### 2.1 최고 노이즈에서의 발산 역전
fwhm=10.0 행 발산 수 = **[5, 5, 5, 3, 1]** (σ_g 오름차순). sg4_fw5 발산 1/5 (F1=0.661) — blur-noise 상호작용이 그리드 전역 현상임을 확인.

### 2.2 안정 계곡의 노이즈 의존성
fwhm=1.25 행 발산 수 = **[0, 0, 1, 3, 2]** — 계곡은 σ_g ≤ 0.041에서만 성립, sg3 이상에서 침식.

### 2.3 안정성–성능 분리
- sg4_fw5: 발산 1/5 ↔ F1=0.661 (최하위권)
- sg0_fw0: 발산 2/5 ↔ F1=0.968 (최상위)
→ 낮은 발산율 ≠ 높은 성능. 두 heatmap 병치 필수.

### 2.4 G2★ 마킹 등가성 한계
전작 G2 = Stage 1-3 (Poisson+Gaussian+S&P) only (`generate_all_groups.py` line 213). Phase 5 fwhm=0은 ring/vignetting/contrast 포함 → 스테이지 구성 상이. **"G2★는 최근접 등가 마킹이며 정확한 재현이 아님"** 보고서 명기 완료.

---

## P4.2r 결과 (40/40 완료)

### 발산 현황
| 조건 | 발산 |
|---|---|
| calib | 12/20 (60%) |
| manual | **20/20 (100%)** |

### 핵심 수치

| 테스트셋 | calib | manual | diff | Welch p |
|---|---|---|---|---|
| TEST_ID | 0.8025 ± 0.124 | 0.8746 ± 0.098 | −0.072 | 0.054 (n.s.) |
| TEST_OOD | 0.7455 ± 0.086 | 0.5330 ± 0.078 | **+0.213** | **1.24e-09** |
| TEST_CAL | 0.8877 ± 0.069 | 0.5413 ± 0.069 | **+0.346** | **6.18e-18** |

TOST (δ=0.02): 전 테스트셋 **비등가**.

### 2×2 교차 행렬
```
              manual 수렴  manual 발산
calib 수렴       0             8
calib 발산       0            12
```

### TEST_OOD 클래스별 F1
class 0 차이 최대: calib=0.745 vs manual=0.194 (diff=+0.551)

### 종합 판정
- TEST_ID: 비등가, manual 약우세 (p=0.054, 비유의)
- TEST_OOD·TEST_CAL: calib 압도적 우세
- 안정성: calib 우위 (60% vs 100% 발산)
- **결론**: calib 조건이 분포 이동 일반화 및 안정성에서 명확히 우세

---

## 커밋 목록

| 커밋 | 내용 |
|---|---|
| `6718f0c` | Phase 5 최종 보고 + P4.2r 착수 config |
| `037e278` | Phase 5 중간 집계 CSV |
| `95e89ee` | CC 완료 보고서 v6.10 초안 |
| (본 커밋) | run-level CSV + §2.1-§2.4 + P4.2r 완료 보고 |

---

*다음 이벤트: P4.2r 결과 검증 인스턴스 검토 → Phase 6 또는 논문 작성 착수 판단*
