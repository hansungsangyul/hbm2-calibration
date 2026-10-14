# 지시 25 (U-3): out_low / out_high 분석 답변
## 작성: 2026-07-09 | 근거: HANDOFF_v6.1 §5.3

---

## Q1. out_low / out_high의 정의

**Stage 6 Contrast** 선형 재스케일링의 출력 범위 하한·상한.

```python
# pipeline.py:218-219
r = (x - self.in_low) / (self.in_high - self.in_low + 1e-8)
return torch.clamp(r * (self.out_high - self.out_low) + self.out_low, 0.0, 1.0)
```

- 입력 x를 `[in_low, in_high]` → `[out_low, out_high]` 로 선형 매핑
- 매핑 후 `[0, 1]` 로 하드 clamp
- `out_low`: 매핑 출력의 최솟값 목표 (clamp 전)
- `out_high`: 매핑 출력의 최댓값 목표 (clamp 전)

---

## Q2. 기원 (선언 위치)

### 초기값 (_INIT dict)
```python
# pipeline.py:85-86
out_low=0.10,
out_high=0.90,
```

### nn.Parameter 선언
```python
# pipeline.py:140-145
self.out_low = nn.Parameter(
    torch.tensor(init['out_low'], dtype=torch.float32)
)
self.out_high = nn.Parameter(
    torch.tensor(init['out_high'], dtype=torch.float32)
)
```

**핵심**: `softplus` / `sigmoid` wrapper 없음 → **unconstrained** (값이 임의의 실수 가능).  
`requires_grad=True` (기본값) → 최적화 시 자유롭게 변동.

---

## Q3. v1–v5 단계별 값

| 버전 | out_low | out_high | 비고 |
|------|---------|----------|------|
| θ_manual | 0.10 | 0.90 | pipeline.py _INIT |
| v1 | 0.10 | 0.90 | 최적화 대상 아님 (v1 PARAMS_TIER2에 없음) |
| v2 | 0.10 | 0.90 | 최적화 대상 아님 |
| v3 | 0.10 | 0.90 | 최적화 대상 아님 |
| v4 | 0.10 | 0.90 | 최적화 대상 아님 |
| **v5** | **0.2515** | **1.4133** | theta_hat_main 3-rep mean |

### v5 상세 (theta_real_v5.json)

| rep | out_low | out_high |
|-----|---------|----------|
| rep0 (seed 950000) | 0.25186 | 1.41274 |
| rep1 (seed 951000) | 0.25113 | 1.41386 |
| rep2 (seed 952000) | 0.25142 | 1.41324 |
| **mean** | **0.25147** | **1.41328** |
| perturb_lo (seed 953000) | 0.25389 | 1.42249 |
| perturb_hi (seed 954000) | 0.25067 | 1.41553 |

rep 간 표준편차: out_low ≈ 0.00037, out_high ≈ 0.00056 → **수렴 양호**.

---

## Q4. 유효성 — out_high > 1.0이 물리적으로 유효한가?

**YES — 유효함.**

근거 (`pipeline.py:219`):
```python
return torch.clamp(r * (self.out_high - self.out_low) + self.out_low, 0.0, 1.0)
```

`out_high=1.4133`이어도 출력은 항상 `[0, 1]`로 clamp됨.  
실제 포화 동작:

| 시나리오 | 선형 매핑 출력 | clamp 후 출력 |
|----------|-------------|------------|
| `r = 0` | `out_low = 0.2515` | 0.2515 |
| `r = 0.5` | `0.2515 + 0.5×(1.4133-0.2515) = 0.8329` | 0.8329 |
| `r = 1` | `out_high = 1.4133` | **1.0** (포화) |
| `r > 0.63` | `> 1.0` | **1.0** (포화) |

의미: v5 최적화 결과는 **입력 r > 0.63 이상 구간을 포화(white clamp)**시키는 고대비 매핑.  
물리적으로는 "실제 dirty2 이미지의 밝은 영역이 이미 포화되어 있다"는 반영.

---

## Q5. 왜 d_calib_config.json에 저장되지 않았나? (PARAMS_TIER2 omission)

`run_phase4_v5.py`의 PARAMS_TIER2 정의:
```python
PARAMS_TIER2 = ['fwhm', 'alpha', 'in_low', 'in_high', 'ring_strength', 'ring_n']
# out_low, out_high 누락
```

결과 연쇄:
1. `theta_real_mean` 계산 시 out_low/out_high 제외
2. `d_calib_config.json` 작성 시 `theta_real_mean` + Tier1 frozen 사용 → out_low/out_high는 pipeline `_INIT` 값 (0.10 / 0.90) 그대로 기록
3. **`theta_hat_main` 배열에는 저장됨** (theta_real_v5.json 내) — 단 d_calib에는 미반영

결론: d_calib의 `out_low=0.10, out_high=0.90`은 v5 최적화값이 아닌 **manual 초기값**.

---

## Q6. 동치 검정 — v5 최적화값과 θ_manual이 동치인가?

**아니오 — 동치 아님.**

### 유효 출력 범위 비교

| 파라미터 세트 | out_low | out_high | 유효 출력 하한 | 유효 출력 상한 | 유효 범위 폭 |
|-------------|---------|----------|-------------|-------------|-----------|
| θ_manual | 0.10 | 0.90 | 0.10 | 0.90 | **0.80** |
| v5 최적화 | 0.2515 | 1.4133 | 0.2515 | 1.00 (포화) | **0.7485** |

### 픽셀 매핑 차이

`r = (x - in_low)/(in_high - in_low)` 기준 실제 매핑값 비교 (r ∈ [0,1]):

| r (정규화 입력) | θ_manual 출력 | v5 출력 (clamp 후) | 차이 |
|--------------|-------------|-----------------|------|
| 0.0 | 0.10 | 0.2515 | **+0.1515** |
| 0.25 | 0.30 | 0.5429 | **+0.2429** |
| 0.50 | 0.50 | 0.8329 | **+0.3329** |
| 0.75 | 0.70 | 1.00 (포화) | **+0.30** |
| 1.00 | 0.90 | 1.00 (포화) | **+0.10** |

### 결론

두 설정은 실질적으로 다른 밝기 매핑을 생성함:
- **θ_manual**: 전체 [0.10, 0.90] 범위로 이미지 밝기 압축/확장
- **v5**: 어두운 영역을 밝게 끌어올리고 (0.10→0.2515), 밝은 영역은 1.0에 포화

단 **현재 Phase 5 grid sweep은 θ_manual (0.10/0.90) 사용** (HANDOFF v6.1 §7.2 fixed_params 명세).  
v5 최적화값은 D_calib §6 판정 완료 후 재평가 대상 (지금은 ring_strength 협의 중).

---

## 요약표

| 항목 | 값/결론 |
|------|--------|
| 선언 위치 | `pipeline.py:140-145` |
| 초기값 | `_INIT` dict L85-86: `out_low=0.10, out_high=0.90` |
| 파라미터 유형 | `nn.Parameter`, unconstrained (no wrapper) |
| v1~v4 | 최적화 안됨 → manual 0.10/0.90 |
| v5 mean | out_low=0.2515, out_high=1.4133 (3-rep mean) |
| out_high>1.0 유효? | YES — L219 clamp(0,1)으로 최종 출력 항상 [0,1] |
| d_calib 누락 이유 | PARAMS_TIER2에 out_low/out_high 미등재 → theta_real_mean 미포함 |
| 동치 검정 | 아님: r=0.5 기준 출력차 ≈ 0.333; 유효 범위 폭 0.80 vs 0.7485 |
| Phase 5 현재 설정 | θ_manual (0.10/0.90) — HANDOFF v6.1 §7.2 |
