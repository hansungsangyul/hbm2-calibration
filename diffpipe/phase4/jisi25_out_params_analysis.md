# 지시 25 — out_low / out_high 미문서화 자유 파라미터 분석
## HANDOFF v6.1 §2 / 지시 25 (U-3) | 작성: 2026-07-09

---

## 문제 요약

v5 교정에서 `out_low`≈0.252, `out_high`≈1.413 으로 수렴 (수동값 0.10/0.90 대비 대폭 이탈).  
이 두 파라미터는 `PARAMS_TIER2`에 포함되지 않아 `theta_real_mean` 및 `d_calib_config.json`에 저장되지 않았음.  
그러나 v5 optimizer에는 명시적으로 포함되어 실제 최적화됨 — **묵시적 자유 파라미터**.

---

## Q1. out_low / out_high 정의

**파이프라인 위치**: S6 Contrast stage (`pipeline.py:216-219`)

```python
def _stage6_contrast(self, x: torch.Tensor) -> torch.Tensor:
    """Linear contrast rescaling."""
    r = (x - self.in_low) / (self.in_high - self.in_low + 1e-8)
    return torch.clamp(r * (self.out_high - self.out_low) + self.out_low, 0.0, 1.0)
```

완전 전개:
```
output = clamp(A·x + B, 0, 1)
  where A = (out_high - out_low) / (in_high - in_low)   ← 유효 기울기
        B = out_low - A · in_low                          ← 유효 절편
```

**해석**:
- `in_low`/`in_high`: 입력 흑점/백점 (입력 동적 범위 정의)
- `out_low`/`out_high`: 출력 흑점/백점 (출력 동적 범위 정의)
- 4개 파라미터가 합쳐서 하나의 affine 클리핑 변환 (A, B)를 정의

**파라미터 타입**: `nn.Parameter`, unconstrained (L134–145):
```python
self.out_low  = nn.Parameter(torch.tensor(init['out_low'],  dtype=torch.float32))
self.out_high = nn.Parameter(torch.tensor(init['out_high'], dtype=torch.float32))
```
in_low/in_high는 `project_constraints` 클리핑이 있으나, out_low/out_high는 **경계 제약 없음**.

---

## Q2. 수동값 출처 (θ_manual: 0.10 / 0.90)

출처: `pipeline.py:85-86` `_INIT` 딕셔너리:
```python
_INIT = dict(
    ...
    out_low=0.10,
    out_high=0.90,
    ...
)
```

**물리적 근거**: 전작 실사 nano-CT 관찰에서  
- 최암부 픽셀값 ≈ 0.10 (완전 흑(0) 미도달)  
- 최명부 픽셀값 ≈ 0.90 (완전 백(1) 미도달, 포화 방지)  

θ_manual은 측정값이 아닌 전작 G2/G4 전문가 판단 값임.  
v1~v5 스크립트에서 `THETA_MANUAL['out_low']`/`THETA_MANUAL['out_high']`로 초기화에 사용.

---

## Q3. v1~v5 버전별 이력 표

| 버전 | out_low optimizer 포함? | 수렴값 out_low | 수렴값 out_high | theta_real_mean 저장? | d_calib 저장? |
|------|------------------------|----------------|----------------|----------------------|---------------|
| v1   | **아니오** (optimizer 미포함) | 0.10 (고정) | 0.90 (고정) | N/A | 0.10 / 0.90 |
| v2   | **아니오** | 0.10 (고정) | 0.90 (고정) | N/A | 0.10 / 0.90 |
| v3   | **아니오** | 0.10 (고정) | 0.90 (고정) | N/A | 0.10 / 0.90 |
| v4   | **아니오** | 0.10 (고정) | 0.90 (고정) | N/A | 0.10 / 0.90 |
| **v5** | **예** (`tier2_direct`에 포함) | **0.252±0.001** | **1.413±0.001** | **아니오** (PARAMS_TIER2 미포함) | **아니오** (누락) |

**v1~v4**: `run_phase4_v{n}.py`에서 out_low/out_high를 optimizer에 전달하지 않음  
→ 초기값 0.10/0.90에서 변화 없음 (gradient 흐르지 않음).

**v5**: `run_phase4_v5.py:181-184`:
```python
tier2_direct = [pipe.in_low, pipe.in_high, pipe.out_low, pipe.out_high]
optimizer = torch.optim.Adam([
    {'params': tier2_params, 'lr': LR_DEFAULT},
    {'params': tier2_direct, 'lr': LR_DEFAULT}
])
```
→ out_low/out_high가 optimizer에 포함됨 (설계 의도인지 실수인지 미기록).  
→ 5개 run 모두 유사한 값으로 수렴 (std < 0.002): **강한 신호, 확률적 노이즈 아님**.

**v5 per-rep 상세**:

| Run | seed | out_low | out_high |
|-----|------|---------|---------|
| rep0 | 950000 | 0.2519 | 1.4127 |
| rep1 | 951000 | 0.2511 | 1.4139 |
| rep2 | 952000 | 0.2514 | 1.4132 |
| perturb-lo | 953000 | 0.2539 | 1.4225 |
| perturb-hi | 954000 | 0.2507 | 1.4156 |
| **평균** | — | **0.252** | **1.414** |

5개 run 표준편차 < 0.002 → **고신뢰 수렴값**.

---

## Q4. out_high > 1.0 물리적 타당성

**직접 답변: 수학적으로 유효, 물리적으로 해석 가능하나 비직관적.**

S6 수식에 `clamp(..., 0, 1)`이 적용되므로:
```
out_high = 1.413 → A·in_high + B = 1.413 → clamp → 1.0
```
즉, `in_high` 이상 입력은 모두 1.0으로 출력 — "밝은 영역 포화(white saturation)".

**유효 대조비 (v5 최적화값 기준)**:
```
A = (1.413 - 0.252) / (0.702 - 0.004) = 1.161 / 0.698 = 1.663
B = 0.252 - 1.663 × 0.004 = 0.245
```
→ 기울기 A = 1.663 > 1: 입력 대비를 66.3% 증폭, 고명부는 포화.

**물리 해석**: 교정된 파이프라인이 실제 dirty2 이미지와 유사한 contrast 분포를 만들려면  
입력 [0.004, 0.702] 범위를 출력 [0.252, 1.0 (포화)] 범위로 증폭해야 함.  
이는 dirty2 이미지의 실제 히스토그램 특성 반영 가능.

**위험 요소**: `out_high > 1` 상태에서 포화 구간 픽셀의 gradient = 0 (clamp 불활성화).  
최적화 도중 안정된 것은 이 gradient zero zone에 진입했기 때문일 수 있음 — false minimum 가능성.

---

## Q5. (in_low, in_high) ↔ (out_low, out_high) 등가 변환 분석

**질문**: v5 최적화값 (in_low=0.004, in_high=0.702, out_low=0.252, out_high=1.413)을  
수동 out값 (out_low=0.10, out_high=0.90)으로 대체하면서 동일한 대조 매핑을 유지하는  
새로운 (in_low', in_high')가 존재하는가?

**조건**: 유효 (A, B) 보존:
```
A_v5 = 1.663,  B_v5 = 0.245

A' = (0.90 - 0.10) / (in_high' - in_low') = 0.80 / (in_high' - in_low')
B' = 0.10 - A' × in_low'

A' = A_v5 = 1.663  →  in_high' - in_low' = 0.80 / 1.663 = 0.481
B' = B_v5 = 0.245  →  in_low' = (0.10 - 0.245) / 1.663 = -0.087
```

**결론**: `in_low' = -0.087` — in_low 제약 [0.0, 0.30] **위반**.  
등가 변환 **불가능** (constraint 하에서).

**실제 d_calib 불일치**: d_calib는 v5 in_low=0.004, in_high=0.702 + 수동 out_low=0.10, out_high=0.90 조합 사용:
```
A_dcalib = (0.90 - 0.10) / (0.702 - 0.004) = 0.80 / 0.698 = 1.146
B_dcalib = 0.10 - 1.146 × 0.004 = 0.095

vs.

A_v5 = 1.663,  B_v5 = 0.245
```
→ 기울기 1.146 vs 1.663 (31% 차이), 절편 0.095 vs 0.245 (157% 차이).  
**D_calib(REVOKED) 상태의 대조 매핑은 v5 최적화 결과와 유의미하게 다름.**

---

## Q6. 기타 미문서화 자유 파라미터 존재 여부

v5 optimizer에 포함된 전체 파라미터 목록 (11개 nn.Parameter 기준):

| 파라미터 | v5 freeze? | PARAMS_TIER2? | optimizer 포함? | 비고 |
|---------|-----------|--------------|----------------|------|
| N_raw | **동결** (Tier 1) | N/A | 아니오 | sigmoid-box |
| sigma_g_raw | **동결** (Tier 1) | N/A | 아니오 | softplus |
| p | **동결** (Tier 1) | N/A | 아니오 | requires_grad=False |
| fwhm_raw | 자유 | **예** | 예 (tier2_params) | softplus |
| alpha_raw | 자유 | **예** | 예 (tier2_params) | softplus |
| ring_strength_raw | 자유 | **예** | 예 (tier2_params) | softplus |
| ring_n_raw | 자유 | **예** | 예 (tier2_params) | softplus |
| in_low | 자유 | **예** | 예 (tier2_direct) | 클리핑 [0,0.3] |
| in_high | 자유 | **예** | 예 (tier2_direct) | 클리핑 [0.7,1.0] |
| **out_low** | 자유 | **아니오** | **예** (tier2_direct) | **미문서화** |
| **out_high** | 자유 | **아니오** | **예** (tier2_direct) | **미문서화** |

**결론**: 미문서화 자유 파라미터는 `out_low`와 `out_high` **두 개뿐**.  
다른 nn.Parameter는 모두 Tier 1 동결이거나 PARAMS_TIER2에 명시적으로 포함됨.

---

## 요약 및 §6 판정표 영향

| 항목 | 내용 |
|------|------|
| v5 out_low 수렴값 | 0.252 (초기값 0.10, 2.5× 이동) |
| v5 out_high 수렴값 | 1.413 (초기값 0.90, 57% 초과; clamp 적용으로 실효 최대 = 1.0) |
| 등가변환 가능 여부 | **불가** (in_low' = -0.087 < 제약 하한 0.0) |
| D_calib(REVOKED) 대조 매핑 오류 | 기울기 31% 차이, 절편 157% 차이 |
| 미문서화 자유 파라미터 수 | **2개** (out_low, out_high만) |
| v5b에서의 처리 | v5와 동일 — tier2_direct에 포함; 결과 비교 후 판단 |

### 권고 (§6 판정표 입력)

출처가 명확하고 5개 run 모두 수렴이 안정적이므로:

1. **out_low/out_high를 PARAMS_TIER2에 추가**하여 공식 교정 대상으로 등록 권장
2. **또는**: out_low=0, out_high=1 고정 (S6를 단순 입력 클리핑으로 단순화), in_low/in_high가 전체 동적 범위를 흡수
3. **의결 전**: v5b 완료 후 v5b의 out_low/out_high 수렴값과 비교하여  
   ring_strength 동결이 out 파라미터에도 영향을 주는지 확인

---

`approved_by`: HANDOFF_v6.1 §2 / 지시 25 완료  
`next_step`: §6 판정표 적용 (지시 23 ESF + 지시 24 v5b 완료 후)
