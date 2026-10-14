# CC 진단 보고서 v6.11
## 근거: HANDOFF v6.11 §C(지시 40) + §D(지시 41) + §E(지시 42)
## 날짜: 2026-07-15

---

## §C — 경위 보고 (지시 40)

### C-1. TOST δ=0.02 채택 경위

**경위**: HANDOFF v6.10 §3 지시(P4.2r 완료 시 보고)에 따라 Welch TOST를 작성할 때 스펙 v1.0 사전 등록값을 조회하지 않고 관례적 등가성 마진 0.02를 사용했다.

**사전 등록 확인**: HANDOFF v6.6 §2 line 63:
> "주 분석 (사전 등록 유지): TEST_ID에서 Welch TOST (δ=0.004)"

**δ=0.004으로 재계산**:

| δ | diff(calib-manual) | TOST_p | 결론 |
|---|---|---|---|
| 0.020 (오기재) | −0.0721 | 0.921 | 비등가 |
| **0.004** (사전 등록) | −0.0721 | 0.966 | **비등가** |

결론 변화 없음. 두 δ 모두 p >> 0.05 → 비등가. **단, 보고서에 사전 등록값 0.004 기준으로 재기재 필요.**

### C-2. "2×2 교차 행렬 (발산 여부)" 생성 경위

v6.6 §2.3이 요구한 것: "학습 조건 × 평가 분포 F1 행렬" (2조건 × 3테스트셋 기술 통계).

내가 생성한 것: 조건별 발산 여부를 독립 seed pairs로 교차표 작성 — 지정물 오해. 독립 seed 간 행별 짝짓기는 의미 없는 구조다.

**올바른 지정물 (v6.6 §2.3 기술 통계 행렬)**:

| 테스트셋 | calib mean ± std | manual mean ± std |
|---|---|---|
| TEST_ID | 0.8025 ± 0.1236 | 0.8746 ± 0.0982 |
| TEST_OOD | 0.7455 ± 0.0855 | 0.5330 ± 0.0783 |
| TEST_CAL | 0.8877 ± 0.0692 | 0.5413 ± 0.0693 |

발산 현황 (별도 기재): calib 12/20, manual 20/20.

---

## §D — 진단 6건 (지시 41)

### D-1. 설정 대조표

**코드 출처**:
- 파일럿: `phase5/run_phase5_pilot.py`
- sg2_fw3: `phase5/run_phase5_grid.py`
- 구판 manual: `phase4/run_p42_tost.py`
- P4.2r: `phase4/run_p42r.py`
- 공통 config: `hbm_augmentation/config.py`

| 항목 | 파일럿 | sg2_fw3 | 구판 manual | P4.2r calib | P4.2r manual |
|---|---|---|---|---|---|
| **batch** | 32 (cfg.BATCH_SIZE) | **16** (arg default) | **8** (arg default; HANDOFF 4 표기는 재확인 필요) | 32 (arg default) | 32 (arg default) |
| **LR** | 1e-4 (cfg) | 1e-4 (cfg) | 1e-4 (cfg) | 1e-4 (cfg) | 1e-4 (cfg) |
| **optimizer** | Adam | Adam | Adam | Adam | Adam |
| **weight_decay** | 1e-5 (cfg) | 1e-5 (cfg) | 1e-5 (cfg) | 1e-5 (cfg) | 1e-5 (cfg) |
| **grad_clip** | 1.0 (cfg) | 1.0 (cfg) | 1.0 (cfg) | 1.0 (cfg) | 1.0 (cfg) |
| **epochs** | 100 (arg) | 41 (arg default) | 41 (arg default) | 41 (arg default) | 41 (arg default) |
| **steps/epoch** | 32 | 63 | 125 | 32 | 32 |
| **총 steps(완료)** | 1,824 (57ep) | **2,583** (41ep) | **5,125** (41ep) | **1,312** (41ep) | **640** (20ep 발산) |
| **prenorm** | **True** | **True** | **없음** | **True** | **False** |
| **sigma_g** | 0.041 | 0.041 | 0.0408 | 0.041 | 0.041 |
| **fwhm** | 2.5 | 2.5 | 2.5 | 2.0956 | 2.5 |
| **alpha** | **0.525 (구 θ)** | 0.35 (신 θ) | 0.35 (신 θ) | 0.35 (신 θ) | 0.35 (신 θ) |
| **ring_strength** | **0.009 (구 θ)** | 0.12 (신 θ) | 0.12 (신 θ) | 0.12 (신 θ) | 0.12 (신 θ) |
| **in_low** | **0.003 (구 θ)** | 0.05 (신 θ) | 0.05 (신 θ) | 0.05 (신 θ) | 0.05 (신 θ) |
| **val 방식** | on-the-fly, 고정 seed | on-the-fly, 고정 seed | on-the-fly, 고정 seed | on-the-fly, 고정 seed | on-the-fly, 고정 seed |
| **ckpt 저장** | O (pilot_checkpoint.pth) | O (final.pth) | O | O (final.pth) | O (final.pth) |

**주요 발견**:
1. 파일럿은 구 θ_manual(alpha=0.525, ring_strength=0.009)을 사용 — 나머지 4 실험과 비교 불가한 독립 변수 존재
2. P4.2r manual은 prenorm=False, 나머지(파일럿·sg2_fw3)는 True — 학습 데이터 분포 차이
3. 총 스텝 수: 구판(5,125) > sg2_fw3(2,583) > 파일럿(1,824) > P4.2r calib(1,312) > P4.2r manual(640, 발산)

### D-2. 학습 곡선 감별

**P4.2r manual 발산 3 runs**:

| run | 발산 ep | train_loss(ep1→ep10→ep_div) | val F1 range (최후10ep) | 감별 |
|---|---|---|---|---|
| manual/run0 | **ep21** | 1.7507 → 0.2393 → 0.0954 | [0.848~0.964] range=0.116 | ▼ |
| manual/run1 | **ep20** | 1.7588 → 0.2655 → 0.1011 | [0.561~0.962] range=0.401 | ▼ |
| manual/run2 | **ep20** | 1.7985 → 0.3219 → 0.1207 | [0.574~0.962] range=0.388 | ▼ |

**P4.2r calib 수렴 2 runs**:

| run | 완료 ep | train_loss(ep1→ep10→ep41) | val F1 range (최후10ep) |
|---|---|---|---|
| calib/run0 | 41 | 1.8704 → 0.2782 → 0.0378 | [0.874~0.946] range=0.072 |
| calib/run2 | 41 | 1.8337 → 0.3365 → 0.0435 | [0.851~0.949] range=0.098 |

**감별 결과**:
- **train_loss**: 전 run 매끄럽게 단조 하강 — 학습 자체는 진행됨
- **val F1 진동**: manual 발산 run은 range 0.11~0.40, calib 수렴 run은 range 0.07~0.10
- **발산 시점**: manual 전 run이 **epoch 20 또는 21**에서 발산 — crit 3 활성화 시점(epoch≥20) 직후
- **H2(평가 노이즈) 기각**: val 열화는 고정 seed(seed_i=idx+100000) → 동일 epoch에 동일 val 이미지 → val 열화 노이즈 없음. 진동은 모델 가중치 불안정에 기인.
- **H1(과소학습) 지지**: batch=32로 ep20까지 640 gradient steps만 수행 — 수렴 전 초기 진동 구간에서 crit 3 트리거

### D-3. val 모니터링 방식 확정

**P4.2r** (`run_p42r.py` line 140-141):
```python
seed_i = idx + (0 if self.is_train else 100000)
torch.manual_seed(seed_i)
eps = {'eps1': torch.randn(H,W), 'eps2': torch.randn(H,W), 'eps3': torch.rand(H,W)}
```
→ val degradation: **고정 seed (on-the-fly이지만 sample idx당 결정론적)** — epoch 간 동일 입력

**Phase 5 grid** (`run_phase5_grid.py` line 166-167): 동일 패턴 (`seed_i = idx + 100000`)

→ 두 실험 val 모니터링 방식 **동일**. val 열화는 양쪽 모두 고정.

### D-4. 총 gradient 스텝 수

훈련 데이터: 1,000장 (`train_data/HBM_clean`)
val 데이터: 500장 (`val_data/HBM_clean`)

| 실험 | batch | steps/ep | 완료 ep | 총 스텝 |
|---|---|---|---|---|
| 파일럿 | 32 | 32 | 57 | 1,824 |
| sg2_fw3 | 16 | 63 | 41 | **2,583** |
| 구판 manual | 8 (default) | 125 | 41 | **5,125** |
| P4.2r calib | 32 | 32 | 41 | **1,312** |
| **P4.2r manual** | 32 | 32 | ~20 | **~640** |

→ P4.2r manual은 발산 시점 기준 640 steps — 구판(5,125)의 1/8, sg2_fw3(2,583)의 1/4
→ **H1 지지**: 스텝 수 부족으로 crit 3 활성화 시점(ep20)에 아직 수렴 전

### D-5. D_manual 데이터 생성 경로 검증

**P4.2r manual 학습 데이터**: `train_data/HBM_clean` 1,000장 → DegradationPipeline(θ_manual, fwhm=2.5, prenorm=False) on-the-fly

**전작 G4 학습 데이터**: `generate_all_groups.py` `apply_degradation(img, params, max_stage=6)` → Stage 1-6 전체 (θ_manual 구버전, prenorm 없음)

**경로 비교**:
- P4.2r manual: 신 θ_manual (alpha=0.35, ring_strength=0.12), prenorm=False
- 전작 G4: 구 θ_manual (스펙 상이), max_stage=6
- 파이프라인 구조: 동일 (Stage 1-6)

**생성 스크립트 상이**: P4.2r=`DegradationPipeline` 클래스 (differentiable), 전작 G4=`apply_degradation` 함수. 기능적으로 동등하나 θ 파라미터 값 차이 존재. 단, P4.2r manual과 구판 manual은 동일 `DegradationPipeline` 사용 → 데이터 생성 결함(H4) 근거 없음.

**H4 잠정 기각**: 구판 manual(동일 구조)이 F1≈0.949 달성했으므로 데이터 생성 경로 자체에 결함 없음.

### D-6. 파일럿 체크포인트 → TEST_ID 평가 (GPU)

**실행**: `eval_pilot_on_testid.py`
**결과**:

```
pilot_epoch=57 (발산 시점), TEST_ID 500장
mean_F1 = 0.6313
per_class = [0.876, 0.672, 0.944, 0.033]
판정 기준: F1 >= 0.92
형식 판정: H5 부상
```

**해석 주의사항 (중요)**:

본 테스트는 다음 혼입 변수를 포함:
1. **파일럿 θ 불일치**: 파일럿은 구 θ_manual(alpha=0.525, ring_strength=0.009, in_low=0.003)로 학습, TEST_ID는 신 θ_manual(alpha=0.35, ring_strength=0.12) 기준 생성 → 학습-테스트 분포 불일치
2. **발산 체크포인트**: 파일럿은 ep57 발산 직후 체크포인트 (val F1=0.833, 최고값 0.943) → 최적 상태 아님
3. **prenorm 불일치**: 파일럿 학습=prenorm True, TEST_ID 생성=prenorm False

class3(background) F1=0.033은 모델 품질 문제(분포 불일치)로 해석됨.

**교차 검증**: P4.2r calib 수렴 runs(N=8)이 TEST_ID에서 mean_F1=0.87~0.94를 달성 → TEST_ID 자체는 정상 생성된 이미지에 대해 합리적 점수 산출. TEST_ID 결함(H5) 근거 없음.

**최종 D-6 판정**: 파일럿 체크포인트 테스트는 **혼입 변수로 결정적이지 않음**. P4.2r calib 결과로부터 TEST_ID 무결성 간접 확인. H5 기각 (간접 근거), P4.2r 학습 측 문제로 좁혀짐.

---

## 가설별 지지/기각 표

| 가설 | 판정 | 근거 |
|---|---|---|
| H1 과소학습 (batch 32 × 41ep) | **지지** | 총 스텝 수 640 (발산 시) — crit 3 활성화 직후(ep20-21) 전건 발산. sg2_fw3(2,583 steps)는 1/5만 발산 |
| H2 평가 노이즈 | **기각** | val 열화 고정 seed(D-3 확인). val F1 진동은 모델 가중치 불안정에 기인 |
| H3 lr-batch 불일치 | **보조적 지지** | batch=32는 구판(8)의 4배 → 동일 lr에서 epoch당 스텝 감소. 단, LR 스케일링 미조정이 직접 원인인지 불분명 |
| H4 데이터 생성 결함 | **기각** | 동일 스크립트 사용 구판이 F1≈0.949 달성. 생성 경로 결함 없음 |
| H5 테스트셋 결함 | **기각 (간접)** | calib 수렴 runs가 TEST_ID F1=0.80-0.94 달성. 파일럿 D-6는 혼입으로 결정적이지 않음 |

**종합 진단**: **H1 주요 원인. H3 보조 기여 가능.**
- P4.2r batch=32, 41ep = 1,312 steps (calib 수렴) / 640 steps (manual 발산)
- Phase 5 grid batch=16은 동일 조건에서 2,583 steps → sg2_fw3 1/5 발산
- 핵심 비교: **스텝 수 절반 감소(2,583→1,312)가 발산율 20%→60%(calib) 기여. prenorm=False(manual)가 추가로 100% 발산 기여.**
- manual이 calib보다 발산율 높은 이유: prenorm=False → 대비 매핑 전 원본 [0,1] 범위 사용 → 학습 이미지 분포 경도 증가 → 수렴 느림 → ep20 crit 3 트리거 확률 높음

---

## §E — v6.10 이행 상태 (지시 42)

| 지시 | 이행 상태 | 커밋 |
|---|---|---|
| 지시 38 (run-level CSV 150행 + 자기 일관성 검사 3종) | **완료** | `cb6755b` |
| 지시 39 (phase5_final_report.md §2.1-§2.4) | **완료** | `cb6755b` |

자기 일관성 검사 통과 로그: `[PASS]` (행 수 150, cell_id 30종×5행, seed 고유, div↔reason 일치, 셀 집계 전 수치 일치)

---

## 결론 및 권고

**P4.2r 결과 수리 보류 유지**. 주요 원인 H1 확인 완료.

**권고 재실행 설정** (검증 인스턴스 승인 후):
- batch=8 (구판 수준) 또는 batch=16 (Phase 5 수준)
- 또는 epochs=100 with batch=32 (총 스텝 3,200으로 증가)
- manual 조건: prenorm=False 유지 (사전 설계 반영)
- 기존 final.pth 체크포인트 보존됨 (재실행 대신 추가 학습 가능 여부는 검증 인스턴스 판단)

**어떤 수정·재실행도 검증 인스턴스 판정 전 착수 금지** (HANDOFF v6.11 §D 말미).
