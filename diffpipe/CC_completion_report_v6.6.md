# CC 완료 보고서 v6.6
## 근거: HANDOFF_v6.6_p42r_redesign.md
## 날짜: 2026-07-12

---

## 요약

| 지시 | 상태 | 결과 |
|---|---|---|
| §0 철회 처리: p42_runs 이동 + validity 기록 | **완료** | `p42_runs_deprecated_confounded_eval/` 이동, tost_welch_result.json validity 필드 추가 |
| 지시 32: Phase 5 체크포인트 긴급 확인·추가 | **완료** | sg2_fw4/rep4부터 final.pth 저장 (83 runs 미보존 수용) |
| §2.1: 테스트셋 3종 생성 + p42r_testsets.md | **완료** | TEST_ID/OOD/CAL 각 500장 생성, 커밋 |
| 지시 33: P4.2r 재실행 설계 (착수 대기) | **설계 완료** | Phase 5 완료 후 착수 (§2 참조) |
| Phase 5 50 runs 중간 보고 | **보류** | 83 runs 완료 중 (50 runs 초과 → §3 참조) |

---

## 1. §0 — 철회 처리 및 기존 산출물 보존

**① p42_runs 이동**:
```
phase4/p42_runs/  →  phase4/p42_runs_deprecated_confounded_eval/  (삭제 없음)
```

**② tost_welch_result.json validity 필드 추가**:
```json
"validity": "DEPRECATED — confounded evaluation (HANDOFF v6.6 §0):
  each condition evaluated on its own distribution, not a common test set.
  P4.2 결과 Table 3 사용 불가."
```

**③ CC의 이탈 3건 자진 보고 (v6.5) 모범 사례 기록**: 접수.

---

## 2. 지시 32 — Phase 5 체크포인트 긴급 확인·추가

**① 확인 결과**:
- 완료된 83 runs: `results.json`만 존재, `final.pth` **0개** (미저장)
- 현재 실행 중: sg2_fw4/rep3 (cell_index=16, rep=3, epoch ~19/41)

**② 조치**: `run_phase5_grid.py`에 checkpoint 저장 1줄 추가:

```python
# results 저장 직전 (HANDOFF v6.6 §1 지시 32)
torch.save(model.state_dict(), out_dir / 'final.pth')
```

**③ 적용 시작 run**: **sg2_fw4/rep4 (cell=16, rep=4)** — 다음 run부터 적용
- 이전 83 runs 미보존: HANDOFF §1-3 수용 ("이미 완료된 run의 미저장은 그대로 수용")
- 이후 67 runs: results.json + final.pth 저장

**④ MANIFEST 갱신**: Phase 5 MANIFEST.md에 "체크포인트 저장 추가" 항목 기록

---

## 3. §2.1 — 테스트셋 3종 생성

**① 산출물 경로**:
- `phase4/generate_p42r_testsets.py` (생성 스크립트)
- `phase4/p42r_testsets.md` (평가 프로토콜 명세 — 본 커밋의 검증 인스턴스 확인 대상)
- `phase4/p42r_testsets/{test_id,test_ood,test_cal}/` (생성된 이미지 + meta.json)

**② 3종 명세**:

| 셋 | pipeline | pre-norm | fwhm | ε 시드 기저 | 장수 |
|---|---|---|---|---|---|
| TEST_ID | θ_manual | 미적용 | 2.5 | 500000 | **500** |
| TEST_OOD | ×1.5 OOD | 미적용 | 3.75 | 510000 | **500** |
| TEST_CAL | D_calib | **적용** | 2.0956 | 520000 | **500** |

- 소스: `test_data/HBM_clean/` 500장 (indices 1501-2000, 엄격 hold-out)
- 저장: `.npy` (float32), border/crop 전처리 포함
- 라벨: `test_data/HBM_labels/` 기존 JSON 참조

**③ HANDOFF 명세 1000장 vs 실제 500장 불일치 보고**:
- `test_data` 전체: 500장 (split: train 1000 / val 500 / test 500)
- val_data(500)는 P4.2r 학습 모니터링에 사용 → 테스트셋 오염 방지상 제외
- **test_data 500장으로 생성**. 검증 인스턴스 확인 요청: 1000장이 val+test 합산 의도였는지?

**④ TEST_OOD pre-norm 처리**:
- HANDOFF가 TEST_OOD의 pre-norm 여부 미명시 → **미적용** 선택 (TEST_ID와 동일, 전작 R1-3 원칙 준수)
- D_calib 모델을 TEST_OOD로 평가 시 pre-norm 불일치 발생 → 이는 OOD 조건의 일부로 해석

---

## 4. 지시 33 — P4.2r 재실행 설계 (Phase 5 완료 후 착수)

**설계 완료** (`p42r_testsets.md`에 전체 명세). 핵심 사항:

| 항목 | 값 |
|---|---|
| 학습 조건 | D_calib (pre-norm + fwhm=2.0956) vs D_manual (no pre-norm, fwhm=2.5) |
| seeds | calib 730000~730019 / manual 740000~740019 |
| batch_size | **32** (전작 G4 준수) |
| epochs | 41 |
| 저장 의무 | final.pth + per-class F1 + results.json |
| 주 평가셋 | TEST_ID (공통 고정, 스펙 v1.0 §4.3 준수) |
| 착수 조건 | Phase 5 완료 후 (예상: ~5일) |

**사전 등록 가설 (HANDOFF §2.3)**:
- TEST_ID: D_manual 분포 일치 이점 → 동등 또는 D_manual 근소 우위 예상
- TEST_OOD·전이 셀: D_calib fwhm 보정 강건성 효과 발현 시 격차 확대 예상

---

## 5. Phase 5 50 runs 중간 보고 (HANDOFF §6, 이전 §4 요건)

**현재 83/150 runs 완료** — 50 runs 기준 초과 달성. 보고 지연 사유: 50 runs 초과 시점에 P4.2 처리(v6.4~v6.6) 병행 중이었음.

**현황**:
- 완료 셀: sg0_fw0 ~ sg2_fw3 (cell 0~15), sg2_fw4/rep0~rep2 (cell 16 일부)
- 진행 중: sg2_fw4/rep3 (cell=16, rep=3)
- 남은 run: 67 runs (~50h)

**batch 16 전환 영향 1줄**: Phase 5는 OOM 대응으로 batch 32→16 전환 (커밋 ac76b9c). 전작 G4(batch=32) 대비 epoch당 스텝 2배, 수렴 소요 epoch 불변 (파일럿 기준 plateau_epoch=21 유지), 실벽시계 ~63s/epoch (batch=32 기준 미측정 — 직접 비교 불가).

---

## 6. 커밋 목록

| 커밋 | 내용 |
|---|---|
| (본 보고 후) | Phase 5 체크포인트 추가 + 테스트셋 생성·명세 + 보고서 v6.6 |

---

## 7. 검증 인스턴스 확인 요청 사항

| 항목 | 내용 |
|---|---|
| 테스트셋 규모 | HANDOFF 1000장 vs 실제 500장 — val+test 합산(1000) 의도인지 확인 요청 |
| TEST_OOD pre-norm | 미적용으로 처리 — 검토 요청 |
| P4.2r 착수 시점 | Phase 5 완료 후 (~5일). 검증 인스턴스 추가 지시 없으면 자동 착수 예정 |

---

*Phase 5 완료 후 P4.2r 자동 착수 예정 (별도 HANDOFF 없을 시). 테스트셋 1000장 여부만 확인 요청.*
