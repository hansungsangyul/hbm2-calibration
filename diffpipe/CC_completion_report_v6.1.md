# CC Completion Report v6.1
## 작성: 2026-07-09 | 근거: HANDOFF_verification_feedback_v6.1.md

---

## §0. 집행 요약

HANDOFF v6.1 모든 지시 (§2, 지시 22, 23, 24, 25, 26) 집행 완료.  
Phase 5 150-run grid sweep 착수.  
§6 판정표 입력값 수집 완료.

**⚠️ 핵심 결과: 지시 25 Q5 §M 트리거 발동 → §6 Row 1 → D_calib 전체 PENDING_CONSULTATION**

- out_low/out_high 가 d_calib에는 manual 값(0.10/0.90)이나 v5 최적화 중 실제값은 0.252/1.413
- pixel-wise MAE 비율 = **8.48×** (§M 임계값 1.5× 훨씬 초과)
- D_calib 재발행 불가 — 검증 인스턴스 결정 필요 (out_* 처리 방침: §5.Q5 참조)

---

## §1. §2 즉시 집행: D_calib REVOKE (2회차)

**대상**: `phase4/d_calib_config.json`  
**상태 변경**: `"FINAL"` → `"REVOKED"` (2회차)

```json
{
  "status": "REVOKED",
  "revoke_reason": "HANDOFF v6.1 §2 — ring 협의 트리거 발동 중 FINAL 전이 (v5 ring_strength=0.1012, v1/v2 합의 0.009의 11.2×). v6.1 §0.1 상태기계 위반."
}
```

**근거**: v5 ring_strength=0.1012는 v1/v2 합의값 0.009의 11.2배. §O 협의 조건 충족 상태에서 v5 스크립트가 자동으로 FINAL 전이 → §0.1 상태기계 위반.

커밋 해시: `45bb3b71...` (2026-07-09 04:14:29 UTC)

---

## §2. 지시 22: configs/param_registry.json 생성

**출력**: `configs/param_registry.json`  
**상태**: `BOOKED` (지시 완료, 검증 대기)  
**커밋**: `45bb3b71`

등재된 파라미터 전체 목록 (15 entries):

| # | 이름 | 유형 | constraints | d_calib 포함 | Phase5 fixed |
|---|------|------|-------------|------------|------------|
| 1 | N_raw (→N) | raw/sigmoid_box | [50,1000] | YES | YES (=300) |
| 2 | sigma_g_raw (→σ_g) | raw/softplus | >0 | YES | YES (=0.041) |
| 3 | p | direct/no_grad | ≥0 | YES | YES (=0.001) |
| 4 | fwhm_raw (→fwhm) | raw/softplus | >0 | YES (=1.4248) | NO (varies) |
| 5 | alpha_raw (→α) | raw/softplus | >0 | YES | YES (=0.35) |
| 6 | min_brightness | class constant | 0.50 | — | — |
| 7 | in_low | direct/unconstrained | [0,1] nominal | YES | YES (=0.05) |
| 8 | in_high | direct/unconstrained | [0,1] nominal | YES | YES (=0.90) |
| 9 | out_low | direct/unconstrained | unbounded | YES (=manual) | YES (=0.10) |
| 10 | out_high | direct/unconstrained | unbounded | YES (=manual) | YES (=0.90) |
| 11 | ring_strength_raw | raw/softplus | >0 | YES | YES (=0.12) |
| 12 | ring_n_raw (→ring_n) | raw/softplus | >0 | YES | YES (=8.0) |
| 13 | prenorm_a | external constant | — | YES | — |
| 14 | prenorm_b | external constant | — | YES | — |
| 15 | _psf_k | int constant (init-fixed) | odd≥3 | — | — |

**핵심 발견**: `out_low`, `out_high`는 v5 최적화됐으나 PARAMS_TIER2에서 누락 → d_calib에 manual값(0.10/0.90) 저장됨 (지시 25 상세 참조).

---

## §3. 지시 23 (U-1): ESF PSF 측정

**스크립트**: `phase4/run_esf_psf.py`  
**출력**: `phase4/esf_psf_measurement.json` + `phase4/esf_plots/*.png`  
**커밋**: `45cfec2` (bilinear fix + final results)

### 방법론
- Sobel top-5% edge 검출 → connected component (len≥20, RMS<1px)
- ESF fit: `A·erf((x-x₀)/(√2·σ))+B`, R²>0.95 필터
- FWHM = 2.3548·σ

### Self-validation 결과 (fwhm=2.5 합성 블러) — v3 최종 (bilinear 프로파일 수정)

| 이미지 | n_valid | median_FWHM | 통과 |
|--------|---------|-------------|------|
| selfval_00 | 81 | 2.300 | OK |
| selfval_01 | 85 | 2.435 | OK |
| selfval_02 | 39 | 2.475 | OK |
| selfval_03 | 29 | 2.473 | OK |
| selfval_04 | 8 | 2.501 | INSUFFICIENT (n<20) |
| **종합 (4/5)** | **234** | **2.454** | **PASS ✅** |

- 목표 범위 [2.25, 2.75] → 실측 2.454 = **PASS**
- 커밋 `45cfec2`: "ESF v3 bilinear fix — self-val PASS (2.454)"

### Real dirty2 측정 결과 (v3)

| 이미지 | n_valid | median_FWHM | 통과 |
|--------|---------|-------------|------|
| dirty2_img00 | 149 | 2.057 | OK |
| dirty2_img01 | 19 | 1.203 | INSUFFICIENT (n<20) |
| dirty2_img02 | 0 | — | INSUFFICIENT |
| dirty2_img03 | 475 | 2.452 | OK |
| dirty2_img04 | 169 | 1.905 | OK |
| **종합 median (3/5 이미지)** | — | **2.057** | — |

- 3개 유효 이미지 medians: [2.057, 2.452, 1.905] → 전체 median = **2.057**
- ESF=2.057: [1.1,1.7] 외부, [2.2,2.8] 외부 → Row 5 후보 (비고: §6 Row 1이 우선 적용됨)
- 2/5 이미지 불충분: dirty2 이미지 일부에 적합한 직선 엣지 부족

---

## §4. 지시 24 (U-2): v5b Ring-Frozen 재교정

**스크립트**: `phase4/run_phase4_v5b.py`  
**출력**: `phase4/theta_real_v5b.json`  
**커밋**: `154a2d8`

### v5b 설계
- Tier 1 동결: N=300, σ_g=0.0408, p=0.001
- Ring 추가 동결: ring_strength=0.009, ring_n=8.6 (v1/v2 합의값)
- Tier 2 자유: fwhm, alpha, in_low, in_high
- Seeds: 955000, 956000, 957000

### v5b 결과 (3 reps, N_STEPS=2000) — 완료

| rep | fwhm | alpha | in_low | in_high | out_low | out_high |
|-----|------|-------|--------|---------|---------|----------|
| 0 (seed 955000) | 1.3705 | 0.4163 | 0.0228 | 0.7346 | 0.0473 | 1.1123 |
| 1 (seed 956000) | 1.3721 | 0.4181 | 0.0215 | 0.7357 | 0.0489 | 1.1107 |
| 2 (seed 957000) | 1.3684 | 0.4181 | 0.0218 | 0.7356 | 0.0474 | 1.1117 |
| **mean** | **1.3703** | **0.4175** | **0.0220** | **0.7353** | ≈0.048 | ≈1.112 |
| std | 0.0015 | 0.00087 | 0.00055 | 0.00052 | — | — |

### v5/v5b 비교표 (최종)

| 파라미터 | θ_manual | v5 mean | v5b mean | Δ (v5b−v5) | Δ% | 해석 |
|---------|---------|---------|---------|-----------|-----|------|
| fwhm | 2.5 | 1.4248 | **1.3703** | −0.0545 | **−3.8%** | ✅ INDEPENDENT (Δ<20%) |
| alpha | 0.35 | 0.4973 | 0.4175 | −0.0798 | −16% | 약간 이동 |
| in_low | 0.05 | 0.0039 | 0.0220 | +0.018 | — | 이동 (ring 동결 효과) |
| in_high | 0.90 | 0.7021 | 0.7353 | +0.033 | — | 이동 |
| out_low | 0.10 (manual) | ≈0.252 (실제) | ≈0.048 | **−0.204** | **−81%** | ⚠️ ring↔out_* 얽힘 |
| out_high | 0.90 (manual) | ≈1.413 (실제) | ≈1.112 | **−0.301** | **−21%** | ⚠️ ring↔out_* 얽힘 |
| ring_strength | 0.12 | 0.1012 | 0.009 | — | 동결 | 동결 |
| ring_n | 8.0 | 7.987 | 8.6 | — | 동결 | 동결 |

### 결론
- **fwhm 얽힘 판정: INDEPENDENT** (Δ=3.82% < 20%) — ring 동결 여부와 무관하게 fwhm≈1.37~1.42 수렴
- **추가 발견**: out_low/out_high가 ring_strength와 강하게 얽혀 있음 (v5: 0.252/1.413 → v5b ring 동결 시: 0.048/1.112)
  → §M 트리거(Q5)의 추가 근거: ring 값이 달라지면 out_* 최적값도 대폭 변화
- 출처: `phase4/theta_real_v5b.json` (해석: `fwhm_shift_analysis.interpretation`)

---

## §5. 지시 25 (U-3): out_low / out_high 6문항 답변

상세 분석: `phase4/out_low_out_high_analysis.md`

### Q1. 정의
Stage 6 contrast 선형 재스케일링 출력 범위 하한/상한.  
`pipeline.py:218-219`: `r=(x-in_low)/(in_high-in_low+1e-8)`, `clamp(r*(out_high-out_low)+out_low, 0, 1)`

### Q2. 기원
`pipeline.py:85-86` _INIT dict에서 초기화 (out_low=0.10, out_high=0.90).  
`pipeline.py:140-145`: unconstrained `nn.Parameter`, `requires_grad=True`.

### Q3. v1–v5 값 (모든 버전에서 optimizer에는 자유, theta_real_mean 미집계, rep 평균)

| 버전 | out_low (rep 평균) | out_high (rep 평균) |
|------|-----------------|-----------------|
| v1 | ≈0.178 | ≈0.812 |
| v2 | ≈0.178 | ≈0.812 |
| v3 | ≈0.120 | ≈0.974 |
| v4 | ≈0.075 | ≈1.079 |
| **v5 mean** | **0.2515** | **1.4133** |
| v5 std | 0.00037 | 0.00056 |

출처: theta_real_{v1-v5}.json `theta_hat_all`/`theta_hat_main` 필드 (PARAMS_TIER2 미포함으로 mean 미수록)  
d_calib에는 0.10/0.90 (manual 하드코딩) — v1-v5 실제 optimizer값과 불일치

### Q4. 유효성
YES. `pipeline.py:219` `clamp(..., 0.0, 1.0)` — 출력 항상 [0,1].  
out_high=1.4133 → r>0.63 구간에서 포화(white clamp). 물리적으로 유효.

### Q5. 등가성 검증 (§M 기준선: 1.5 × d_realization)

**비교**: Config A(v5 실제 out_low=0.252, out_high=1.413) vs Config B(manual 0.10/0.90)

| 지표 | 값 |
|------|----|
| d_AB pixel-wise MAE (10쌍 평균) | **0.2867** |
| d_realization (동일 θ, 다른 ε) | 0.0338 |
| ratio = d_AB / d_real | **8.48** |
| threshold (§M) | 1.5 |
| **§M 트리거** | ⚠️ **YES (8.48 >> 1.5)** |

CalibrationLoss 4개 통계량:

| 통계량 | Config A (v5 실제) | Config B (manual out) | Δ |
|--------|-----------------|----------------------|---|
| L_nps | 0.030 | 0.503 | **+0.473 (17×)** |
| L_vm | 0.021 | 0.032 | +0.011 |
| total | 0.061 | 0.543 | +0.482 (9×) |

**결론**: §M 트리거 발동 → **§6 Row 1: 전체 PENDING_CONSULTATION**  
상세: `phase4/out_equivalence_test.json`

### Q6. 스펙 §1.1에 없는 은닉 자유도
**out_low, out_high 2개만** 해당. 나머지 optimizer 파라미터 (fwhm, alpha, in_low, in_high, ring_strength, ring_n) 모두 §1.1 Tier 2 명시. 다른 은닉 자유도 없음.  
상세: `phase4/out_low_high_clarification.md`

---

## §6. §6 판정표 적용

### 입력 값

| 입력 | 값 | 비고 |
|------|-----|------|
| 지시 25 Q5 §M ratio | **8.48** (임계값 1.5) | ⚠️ 트리거 발동 |
| ESF median (v3, 3/5 이미지) | **2.057** | self-val PASS |
| ESF self-val 결과 | **PASS** (2.454 for true 2.5) | v3 bilinear fix |
| v5b fwhm mean | **1.3703 ± 0.0015** (3 reps) | 완료 (theta_real_v5b.json) |

### 판정표 적용 결과

HANDOFF v6.1 §6: "위에서부터 순서대로 검사, 최초로 참인 행 하나만 적용"

| 행 | 조건 | 충족? | 결과 |
|----|------|-------|------|
| **Row 1** | 지시 25 트리거 발동 | **⚠️ YES** | **→ 전체 PENDING_CONSULTATION** |
| Row 2 | ESF 자기검증 실패 (2회) | NO (v3 PASS) | 적용 안 됨 |
| Row 3 | ESF ∈[1.1,1.7] AND v5b∈[1.28,1.56] | 미확인 | Row 1 이미 발동 |
| Row 4 | ESF ∈[2.2,2.8] | NO (2.057) | Row 1 이미 발동 |
| Row 5 | 위 외 | YES | Row 1 이미 발동 |

### 최종 판정: ⚠️ **§6 Row 1 — 전체 PENDING_CONSULTATION**

- D_calib 재발행 불가
- P4.2 착수 불가
- 필요 결정사항: out_low/out_high 처리 방침
  - 옵션 A: out_* 를 PARAMS_TIER2에 편입 → v5 재실행 + theta_real_mean 재집계 + D_calib 재발행
  - 옵션 B: out_* = requires_grad=False (manual 고정) → v5 재실행
  - 검증 인스턴스(사용자) 결정 필요

---

## §7. 지시 26: Phase 5 Grid Sweep 착수

### 착수 조건 충족

| 조건 | 상태 |
|------|------|
| 지시 22 커밋 | ✅ `45bb3b71` |
| grid_v1.json v6.1 반영 커밋 | ✅ `45bb3b71` |
| run_phase5_grid.py 커밋 | ✅ `66972bc` |
| 150 runs 착수 | ✅ **PID 2049202** (3차 시작; launcher bash PID, worker 2049205) |

### Grid 구성 (HANDOFF v6.1 §7.2)

- `sigma_g` × `fwhm`: [0.01025, 0.0205, 0.041, 0.082, 0.164] × [0, 0.625, 1.25, 2.5, 5.0, 10.0]
- 30 cells × 5 reps = **150 runs**
- `seed = 800000 + cell_index×10 + rep`
- fixed_params = θ_manual
- fwhm=0 → PSF stage skip
- 발산 기준 3: medfilt(k=3) 적용 (§V 개정)
- λ_max at epochs {5, 10, 20, 30, 40}

### 실행 상황

```
Grid launcher PID: 2049202 (bash launch_grid_all.sh)
Grid worker PID:   2049205 (현재 cell=0 rep=0, GPU 30.7 GiB)
Log: phase5/grid_run.log
출력: phase5/runs/sg{i}_fw{j}/rep{k}/results.json
```

**이력**:
- 1차 (PID 1291878): GPU OOM (다른 프로세스 점유) → 즉시 실패
- 2차 (PID 1735026): v5b와 동시 실행 → OOM; 새 launcher(1735023)가 중복 실행하면서 충돌 → 모두 강제 종료
- **3차 (PID 2049202)**: 2026-07-09 05:23 재시작, GPU 독점 확인 후 시작. cell=0 rep=0 정상 실행 중 (30.7GiB 사용)
- 예상 완료: 150 runs × ~70분/run = ~175시간 (순차 실행)

---

## §8. 상태 기계 현황

| 항목 | 상태 | 비고 |
|------|------|------|
| d_calib_config.json | `PENDING_CONSULTATION` | §6 Row 1 (Q5 §M 트리거) → out_* 처리 방침 대기 |
| param_registry.json | `BOOKED` | 검증 대기 |
| Phase 5 grid sweep | **실행 중 (3차)** | launcher 2049202, worker 2049205, cell=0 rep=0, ~175h |
| ESF 측정 | 완료 (v3 bilinear) | sv_median=2.454 PASS; real=2.057 → Row 5 (Row 1 우선) |
| v5b 교정 | ✅ **완료** | mean fwhm=1.3703 (Δ=−3.8% vs v5) → ring/fwhm INDEPENDENT; out_low/out_high↔ring 얽힘 확인 |
| 지시 25 답변 | 완료 | `out_low_high_analysis.md`, `out_equivalence_test.json` (Q5 §M ratio=8.48, TRIGGER) |
| §6 판정 | **Row 1 PENDING** | Q5 트리거 발동, D_calib 전체 대기 |
| D_calib 재발행 | **PENDING_CONSULTATION** | out_* 처리 방침 결정 필요 |

---

## §9. 커밋 이력 (v6.1 기간)

| 해시 | 내용 |
|------|------|
| `45bb3b71` | param_registry, grid_v1 갱신, d_calib REVOKE(2회차), criterion_revision.md |
| `154a2d8` | run_esf_psf.py, run_phase4_v5b.py |
| `66972bc` | run_phase5_grid.py |
| `1a0ed49` | medfilt fix, launch_grid_all.sh, out_low_out_high_analysis.md |
| `97ea9bc` | ESF attempt 2 결과 (self-val FAIL 2.105) |
| `45cfec2` | ESF attempt 3 bilinear fix — self-val PASS (2.454), real=2.057 → §6 확정 |
| `b029761` | CC report §8 d_calib REVOKED→PENDING_CONSULTATION 수정 |
| `828e16e` | v5b 완료 (fwhm=1.3703 INDEPENDENT); ring↔out_* 얽힘 확인; CC report §4 갱신 |

---

## §10. 미완료 항목 및 검증 인스턴스 결정 대기

### 계속 실행 중
1. **Phase 5 grid** (tmux `phase5_grid`, launcher 2049202, worker 2049205): cell=0 rep=0 실행 중 (30.7GiB GPU), 순차 ~175h 예상

### 검증 인스턴스 결정 필요 (§6 Row 1 발동)
- **out_* 처리 방침 결정**: 옵션 A (PARAMS_TIER2 편입 + v5 재실행) vs 옵션 B (manual 고정 + v5 재실행)
- 결정 후 D_calib 상태: PENDING_CONSULTATION → 처리 방침 확정 → BOOKED (재실행 대기)

### v5b 완료 후 수행 완료
- `theta_real_v5b.json` 커밋 + §4 v5b 섹션 업데이트 (fwhm shift 분석 포함)
- D_calib `REVOKED` 상태 유지 (§6 Row 1으로 인해 재발행 불가)
- param_registry.json `BOOKED` → `PENDING_CONSULTATION` (out_* 미확정)

---

*[§6 Row 1 발동: D_calib 전체 PENDING_CONSULTATION — 검증 인스턴스 결정 대기]*
