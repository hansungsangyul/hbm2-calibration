# 지시 25 (U-3): out_low/out_high 전면 해명
## 근거: HANDOFF_v6.1 §5 | 작성: 2026-07-09

---

## Q1. 정의: 수식·적용 스테이지·적용 순서

**스테이지**: Stage 6 (Contrast Rescaling) — 7단계 파이프라인 중 6번째, Vignetting(5) 직후 Ring(7) 직전.

**수식** (`src/pipeline.py:L218-219`):
```python
r = (x - self.in_low) / (self.in_high - self.in_low + 1e-8)
return torch.clamp(r * (self.out_high - self.out_low) + self.out_low, 0.0, 1.0)
```

- 입력 범위 [in_low, in_high]를 출력 범위 [out_low, out_high]로 선형 매핑
- 최종 `clamp(·, 0, 1)` 적용 — 출력은 항상 [0,1]에 존재
- out_low, out_high는 별도 스테이지 아님 — Stage 6 내부 파라미터

**파라미터 선언** (`src/pipeline.py:L141-145`):
```python
self.out_low  = nn.Parameter(torch.tensor(init['out_low'],  dtype=torch.float32))  # L141-143
self.out_high = nn.Parameter(torch.tensor(init['out_high'], dtype=torch.float32))  # L144-145
```
→ 두 파라미터 모두 `unconstrained` (softplus/sigmoid 없음, 제약 없음).

---

## Q2. 기원: 전작 파이프라인 존재 여부 및 도입 시점

**도입 시점**: 본 연구(HBM_paper2/diffpipe) 최초 파이프라인 구현 시점. 전작(HBM_paper1) 파이프라인과는 별개.

**초기값(manual)** (`src/pipeline.py _INIT dict:L85-86`):
```python
'out_low':  0.10,
'out_high': 0.90,
```

**출처**: Stage 6의 출력 범위를 제한하기 위해 신규 도입 (0.10~0.90: 입력 다이나믹 레인지의 일부만 출력에 사용).

---

## Q3. v1~v5 버전별 자유/동결 여부 및 최종값 (rep 평균)

| 버전 | optimizer 포함 | PARAMS_TIER2 포함 | theta_real_mean 저장 | d_calib 값 | out_low (rep 평균) | out_high (rep 평균) |
|------|--------------|-----------------|-------------------|----------|-----------------|-----------------|
| v1   | ✅ 자유 (`tier2_direct`) | ❌ | ❌ | — | ≈0.178 | ≈0.812 |
| v2   | ✅ 자유 (`tier2_direct`) | ❌ | ❌ | — | ≈0.178 | ≈0.812 |
| v3   | ✅ 자유 (`tier2_direct`) | ❌ | ❌ | — | ≈0.120 | ≈0.974 |
| v4   | ✅ 자유 (`tier2_direct`) | ❌ | ❌ | — | ≈0.075 | ≈1.079 |
| v5   | ✅ 자유 (`tier2_direct`) | ❌ | ❌ | **0.10 / 0.90 (manual 하드코딩)** | ≈0.252 | ≈1.413 |

**근거 코드**:
- v5 optimizer: `run_phase4_v5.py:L181-185` — `tier2_direct = [pipe.in_low, pipe.in_high, pipe.out_low, pipe.out_high]`
- v5 d_calib 하드코딩: `run_phase4_v5.py:L475-476` — `out_low=THETA_MANUAL['out_low']` (=0.10)
- v5 실제 값: `theta_real_v5.json` → `theta_hat_main[i]` 필드 (PARAMS_TIER2 제외로 mean 미집계)

**핵심 문제**: v1-v5 전체에서 out_low/out_high는 optimizer에는 자유였지만 PARAMS_TIER2에 없어서 `theta_real_mean`에 저장되지 않았고, d_calib에는 manual 값(0.10/0.90)이 하드코딩됨 → **optimizer가 사용한 out_* 값과 d_calib 출력 값 불일치**.

---

## Q4. out_high = 1.413의 유효성

**결론**: 물리적으로 유효. 이유:

Stage 6 출력에 `clamp(·, 0, 1)` 적용 (`pipeline.py:L219`) → out_high > 1이어도 최종 출력은 [0,1]에 머뭄.

**물리적 의미**:
- v5 실제값: out_low≈0.252, out_high≈1.413 → 출력 = clamp(1.161·r + 0.252, 0, 1)
- r = (x - 0.004)/(0.702 - 0.004) ≈ (x - 0.004)/0.698 (v5 in_low/in_high 사용)
- r=0.71 (x=0.5 근방) → 출력 = clamp(1.161×0.71 + 0.252, 0, 1) = clamp(1.076, 0, 1) = **1.0** (포화)
- 즉: out_high > 1은 출력 상한을 초과하는 공격적 콘트라스트 스트레칭 + 밝은 픽셀 포화 효과

**수치**: v5 3 reps: out_high = 1.4127, 1.4139, 1.4132 (평균 1.4133). 5 reps(perturb 포함): 1.4224, 1.4155 포함 평균 ≈1.416. 매우 일관된 수렴.

---

## Q5. 등가성 검증 (§M 기준선: 1.5 × d_realization)

### 비교 대상
- **Config A** (v5 실제): out_low=0.2515, out_high=1.4133 (v5 rep 0-2 평균)
- **Config B** (manual out_*): out_low=0.10, out_high=0.90 (d_calib 기재값)
- 나머지 파라미터 동일 (v5 θ̂: fwhm=1.4248, alpha=0.4973, in_low=0.0039, in_high=0.7021, ring=0.1012)

### 결과

| 지표 | 값 | 판정 |
|------|-----|------|
| d_AB (pixel-wise MAE, 10쌍 평균) | **0.2867** | — |
| d_realization (same θ, diff ε, 10쌍 평균) | 0.0338 | — |
| ratio = d_AB / d_real | **8.48** | — |
| threshold (§M) | 1.5 | — |
| **트리거 발동** | **YES (8.48 >> 1.5)** | ⚠️ |

### 4개 CalibrationLoss 통계량 비교

| 통계량 | Config A (v5 실제) | Config B (manual out_*) | Δ |
|--------|-----------------|----------------------|---|
| L_vm   | 0.020828 | 0.031954 | +0.011 (54% 증가) |
| L_nps  | 0.029673 | 0.502912 | +0.473 (**17× 증가**) |
| L_rad  | 0.005383 | 0.006579 | +0.001 (22% 증가) |
| L_hist | 0.005342 | 0.001511 | -0.004 (72% 감소) |
| total  | 0.061226 | 0.542956 | +0.482 (**9× 증가**) |

L_nps(NPS 매칭) 17× 증가 — out_* 변경이 노이즈 전력 스펙트럼에 치명적 영향.

### 결론
**§M 트리거 발동** → §6 Row 1 적용: `전체 PENDING_CONSULTATION`

원인: v5 optimizer는 out_low≈0.252, out_high≈1.413으로 fwhm/alpha/ring을 최적화했으나 d_calib에는 0.10/0.90이 기재됨. 이 불일치가 d_realization의 8.5×. 피팅 해 왜곡 확인.

상세: `phase4/out_equivalence_test.json`

---

## Q6. 스펙 §1.1에 없는 추가 자유도

전수 확인 (configs/param_registry.json 기준, v5 optimizer 대조):

| 파라미터 | 스펙 §1.1 | v5 optimizer | 불일치 |
|----------|-----------|--------------|--------|
| fwhm     | ✅ Tier 2 | ✅ tier2_params | 정상 |
| alpha    | ✅ Tier 2 | ✅ tier2_params | 정상 |
| in_low   | ✅ Tier 2 | ✅ tier2_direct | 정상 |
| in_high  | ✅ Tier 2 | ✅ tier2_direct | 정상 |
| ring_strength | ✅ Tier 2 | ✅ tier2_params | 정상 |
| ring_n   | ✅ Tier 2 | ✅ tier2_params | 정상 |
| **out_low**  | ❌ 스펙에 없음 | ✅ tier2_direct | **불일치** |
| **out_high** | ❌ 스펙에 없음 | ✅ tier2_direct | **불일치** |
| N_raw (frozen) | Tier 1 | ❌ 제외 | 정상 |
| sigma_g (frozen) | Tier 1 | ❌ 제외 | 정상 |
| p (frozen) | Tier 1 | ❌ 제외 | 정상 |

**결론**: out_low, out_high 2개만 스펙 §1.1에 미포함 자유도. 다른 추가 은닉 자유도 없음 (pre-norm a/b는 external constant, 최적화 대상 아님).

---

## 트리거 발동 요약

- **Q5 §M 트리거**: 발동 (ratio=8.48 >> 1.5)
- **§6 Row 1 적용**: 전체 PENDING_CONSULTATION
- **의미**: out_* 는 단순 reporting 누락이 아니라 피팅 해 자체를 바꾸는 파라미터 — d_calib 재발행 전에 out_* 처리 방침 확정 필요.
- **CC 권고**: out_low/out_high를 (a) PARAMS_TIER2에 편입 + theta_real_mean 재집계 후 D_calib 재발행, 또는 (b) 수동값으로 고정(requires_grad=False) 후 v5 재실행 중 택일 — 검증 인스턴스(사용자) 결정 필요.

---

**approved_by**: 해당 없음 — PENDING_CONSULTATION 발동, 검증 인스턴스 판단 대기
