# CC 완료 보고서 v6.5
## 근거: HANDOFF_v6.5_p42_ruling.md
## 날짜: 2026-07-12

---

## 요약

| 지시 | 상태 | 결과 |
|---|---|---|
| 지시 29: Welch TOST 재계산 (스펙 준수) | **완료** | 검증값 전항목 일치 ✓ |
| 지시 30 Q1: 테스트셋 명세 | **완료** | 스펙 §4.2 이탈 확인 → 즉시 보고 |
| 지시 30 Q2: pre-norm 적용 여부 | **완료** | 적용됨 (코드 경로 확인) |
| 지시 30 Q3: batch_size 전작 G4 부합 여부 | **완료** | 불일치 (4 vs 32) — 내적 타당성 유지 |
| 지시 30 Q4: per-class F1 + 체크포인트 | **완료** | 체크포인트 미보존 → OOD·per-class 불가 |
| 지시 31: OOD 1.5× 재평가 | **즉시 보고** | 체크포인트 미보존 → 재학습 없이 불가 |
| §4 ablation 비용 견적 | **완료** | Q2=적용 → ~24h (착수 대기) |

---

## 1. 지시 29 — Welch TOST 재계산

**① 상태**: 완료

**② 산출물 경로**:
- `phase4/p42_tost_welch.py` (신규)
- `phase4/p42_runs/tost_welch_result.json` (신규)

**③ paired t 사용 경위 (코드 경로 인용)**:

```
run_p42_tost.py L100: diff = np.array(f1_c) - np.array(f1_m[:len(f1_c)])
run_p42_tost.py L102: t_lo, p_lo = stats.ttest_1samp(diff, -TOST_DELTA, ...)
```

calib와 manual의 seed 기반이 다름(700000~700019 vs 710000~710019) — 공유 무작위 인자 없음. `ttest_1samp(diff, ...)` 호출이 paired t로 작동했으나 쌍 구조 부당.

**④ Welch TOST 재계산 결과** (`p42_tost_welch.py`, ddof=0 — 원본 tost_result.json 기준값과 동일):

| 항목 | 계산값 | HANDOFF 기준값 | 일치 |
|---|---|---|---|
| SE (Welch) | 0.002764 | 0.002761 | ✓ |
| t_lower | 3.823 | 3.83 | ✓ |
| p_lower | 0.0003 | — | PASS |
| t_upper | 0.928 | 0.93 | ✓ |
| p_upper | 0.8201 | — | **FAIL** |
| 등가성 판정 | | | **미확인** |

**⑤ 우월성 검정 결과** (사전 등록 보너스 분기):

| 항목 | 계산값 | HANDOFF 기준값 | 일치 |
|---|---|---|---|
| t (Welch 양측) | 2.375 | ≈2.38 | ✓ |
| df (Welch–Satterthwaite) | 33.9 | ≈34 | ✓ |
| p | 0.0233 | ≈0.02 | ✓ |
| Cohen's d | 0.751 | ≈0.75 | ✓ |
| **결론** | **D_calib 유의한 우월** (p<0.05) | | |

**⑥ [부록] paired t (참고용, 스펙 이탈)**:

| | t | p |
|---|---|---|
| lower | 3.399 | 0.0015 |
| upper | 0.825 | 0.7903 |

---

## 2. 지시 30 — 해명 4건

### Q1. 평가 테스트셋 명세 【최우선 — 스펙 §4.2 이탈 확인】

**테스트셋 실제 구성**:
- 파일: `val_data/HBM_clean/` 원본 클린 이미지 500장 (인덱스 1001~1500, `split_data.py` 기준)
- 라벨: `val_data/HBM_labels/`
- **열화 방식: 런타임 온-더-플라이** — `P42Dataset.__getitem__`에서 각 조건의 파이프라인 적용

```python
# run_p42_tost.py L220-222
img_norm = PRENORM_A * img + PRENORM_B
x = torch.from_numpy(img_norm)
y = self.pipe(x, eps)      # ← 각 조건 파이프라인 (D_calib 또는 D_manual)
```

**스펙 §4.2 이탈 확인**:

| 항목 | 스펙 요구 | 실제 |
|---|---|---|
| 테스트셋 | θ_manual 기반 공통 고정 테스트셋 | **각 조건이 자기 파이프라인으로 런타임 생성** |
| D_calib val | θ_manual 파이프라인(fwhm=2.5) 열화 | **D_calib 파이프라인(fwhm=2.0956) 열화** |
| D_manual val | θ_manual 파이프라인(fwhm=2.5) 열화 | D_manual 파이프라인(fwhm=2.5) 열화 ✓ |

**영향 분석**:
- D_manual은 자기 분포 테스트에서 F1=0.9481 (HANDOFF §1 지적: "강한 결과")
- D_calib는 자기 분포 테스트에서 F1=0.9546
- 단, 두 테스트셋의 열화 파라미터가 다르므로 F1 차이(+0.0066)에 "D_calib 분포 자체가 학습·평가 모두 더 쉬운 효과" 혼입 가능성 배제 불가
- 공통 테스트셋(θ_manual)으로 재평가 필요 시 **체크포인트 미보존으로 불가 (Q4 참조)**

**결론**: 테스트셋 스펙 이탈 사실 보고. 즉시 대응 불가 (체크포인트 없음).

---

### Q2. D_calib 생성 시 pre-norm 적용 여부

**답변: 적용됨**

```python
# run_p42_tost.py L221 (D_calib 조건 포함 양 조건 공통)
PRENORM_A = 0.501834
PRENORM_B = 0.173320
img_norm = PRENORM_A * img + PRENORM_B    # ← pre-norm 적용
x = torch.from_numpy(img_norm)
y = self.pipe(x, eps)                     # ← 파이프라인 입력
```

PRENORM_A=0.501834, PRENORM_B=0.17332 는 `d_calib_config.json`의 `prenorm` 항목과 동일.

**귀속 함의**: 우월성 원인 후보 = {pre-norm, fwhm 2.0956} 결합.
→ HANDOFF §4에 따라 D_prenorm-only ablation 필요. 비용 견적은 §4 참조.

---

### Q3. batch_size=4의 전작 G4 프로토콜 부합 여부

**전작 G4 batch_size**:
```
/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation/config.py  L25:
    BATCH_SIZE = 32
```

| | 전작 G4 | P4.2 | Phase 5 |
|---|---|---|---|
| batch_size | **32** | **4** | 16 |
| 근거 | `config.py BATCH_SIZE=32` | `launch_p42.sh --batch_size 4` | OOM 회피 (ac76b9c) |

**불일치 확인 (4 vs 32)**.

- **내적 타당성 (조건 간 비교)**: 유지 — calib/manual 양 조건 동일 설정(batch=4)
- **전작 절대 수치 비교**: 불가 — 논문에 명시 필요 ("P4.2는 batch=4로 수행, 전작 G4(batch=32)와 절대 F1 직접 비교 불가")

---

### Q4. per-class F1 + 체크포인트 보존 여부

**per-class F1**:

```python
# run_p42_tost.py L242-253: compute_f1 반환값
def compute_f1(preds, targets, n_classes=4):
    ...
    return float(np.mean(f1s))   # ← 평균만 반환, 클래스별 미저장
```

Crack/Void/Open별 F1은 내부에서 계산되나 results.json에 기록되지 않음.

**체크포인트 보존 여부**:

```bash
find phase4/p42_runs -name "*.pth"   → 0개
```

각 run의 산출물: `results.json` 만 존재. 학습 완료 후 모델 가중치 미저장.

**결론**:
- per-class F1: 체크포인트 없어 재추론 불가 → 제출 불가
- OOD 재평가 (§3): 체크포인트 없어 재평가 불가 → 재학습 협의 필요

---

## 3. 지시 31 — OOD 1.5× 재평가 【즉시 보고】

**체크포인트 미보존** (Q4에서 확인: 0개).

P4.2 40개 모델 전원의 가중치가 존재하지 않음 → OOD 재평가 불가.

**재학습 없이 수행 불가. 재학습 여부 협의 요청.**

참고 — 재학습 시 비용 견적:
- 40 runs × ~72min/run = **~48시간** (Phase 5와 병렬 불가 시)
- 체크포인트 저장 로직 추가 필요 (`torch.save(model.state_dict(), ...)`)

전작 R1-3의 OOD 설정 (참고):
- ID: `test/g4_full` 1000장 (in-distribution)
- OOD: `test_ood/g4_full` 1000장 (PSF 2.5→3.75, σ_bg ×1.5, Ring 0.12→0.18)
- P4.2 적용 시: 동일 OOD 강도(×1.5), 조건별 D_calib/D_manual 기반 OOD 분포 정의 필요

---

## 4. §4 — Ablation 비용 견적 (Q2=적용 확인에 따른 조건부 착수)

Q2=적용(pre-norm 포함) 확인 → HANDOFF §4 조건 충족.

**D_prenorm-only 조건**: pre-norm 적용 + 나머지 전부 θ_manual(fwhm=2.5)

| 항목 | 값 |
|---|---|
| 목적 | pre-norm vs fwhm 보정 기여 분리 |
| runs | 20 |
| 예상 소요 | ~72min/run × 20 = **~24시간** |
| 시드 | 별도 할당 필요 (720000~720019 제안) |
| 착수 조건 | **검증 인스턴스 승인 후** |

---

## 5. 미완료 항목 (검증 인스턴스 판단 대기)

| 항목 | 상태 | 사유 |
|---|---|---|
| Q1 테스트셋 스펙 이탈 처리 | **PENDING_CONSULTATION** | 공통 테스트셋 미사용. 재학습 없이 수정 불가 |
| Q4 per-class F1 | **PENDING_CONSULTATION** | 체크포인트 미보존 → 재추론 불가 |
| 지시 31 OOD 재평가 | **PENDING_CONSULTATION** | 체크포인트 미보존 → 재학습 여부 협의 요청 |
| §4 ablation (D_prenorm-only) | **착수 대기** | 검증 인스턴스 승인 후 진행 |
| Phase 5 50 runs 중간 보고 | **보류** | 50 runs 미도달, 계속 진행 중 |

---

## 6. 커밋 목록

| 커밋 | 내용 |
|---|---|
| (본 보고 후) | `p42_tost_welch.py` + `tost_welch_result.json` + 보고서 v6.5 |

---

*승인 대기 항목 3건: (1) Q1 테스트셋 이탈 처리 방향, (2) OOD 재학습 여부, (3) §4 ablation 착수 승인.*
