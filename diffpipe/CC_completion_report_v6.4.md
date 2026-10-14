# CC 완료 보고서 v6.4
## 근거: HANDOFF_v6.4_sigma_post_ruling.md
## 날짜: 2026-07-12

---

## 요약

| 지시 | 상태 | 결과 |
|---|---|---|
| 지시 28-A: σ_post 파이프라인 구조 수정 (S5.5) | **완료** | gradcheck PASS, bit-identical PASS |
| 지시 28-B: 민감도 미니 체크 (σ_post↔ring 분리) | **완료** | pattern_diff=67.19% → PASS |
| 지시 28-C: v5e 재캘리브레이션 | **완료** | §2 행 3 fallback → FINAL |
| D_calib FINAL 발행 | **완료** | θ_manual + fwhm=2.0956 + σ_post=0 |
| P4.2 TOST (D_calib vs D_manual) | **완료** | 등가성 미확인 (TOST FAIL — 아래 §4) |
| Phase 5 | **진행 중** | 계속 실행 중 (무영향) |

---

## 1. 지시 28-A: σ_post 파이프라인 구조 수정

**① 지시 ID·상태**: 지시 28-A (HANDOFF v6.4 §1.1) — **완료**

**② 산출물 경로**:
- `src/pipeline.py` (S5.5 `_stage55_post_noise` 추가)
- `configs/param_registry.json` (sigma_post 항목 추가, total_optimizable 10→11)
- `phase4/test_pipeline_v5e.py` (회귀 테스트)
- 커밋: `ce2f5d0`

**③ 구조 변경 내용**:

추가 스테이지 **S5.5 (Post-blur detector noise)**: vignetting(S5) 후 · contrast(S6) 전

```python
def _stage55_post_noise(self, x, eps_post):
    return x + self.sigma_post * eps_post
```

- `sigma_post`: `nn.Parameter`, bounds [1e-4, 0.1], init 0.01, reparam clamp
- `eps_post`: 신규 독립 채널 (`eps.get('eps_post')`, 없으면 `zeros_like`)
- `manual_value: 0.0` (전작 파이프라인 부재 → σ_post=0이면 구파이프라인과 완전 동치)

**④ 회귀 테스트 결과**:

| 테스트 | 결과 |
|---|---|
| S5.5 단독 gradcheck (float64) | **PASS** |
| σ_post=0 → 기존 파이프라인과 bit-identical (max\|diff\|=0.0) | **PASS** |
| eps_post 누락 시 zeros 대체 동작 | **PASS** |

σ_post=0 bit-identical 확인 → D_manual(구파이프라인)·Phase 5와 비교 정당성 확보.

---

## 2. 지시 28-B: 민감도 미니 체크 (σ_post ↔ ring 분리)

**① 지시 ID·상태**: 지시 28-B (HANDOFF v6.4 §1.2) — **완료**

**② 산출물 경로**:
- `phase4/run_sensitivity_v5e.py`
- `phase4/sensitivity_v5e.json`
- 커밋: `ce2f5d0`

**③ 핵심 수치** (기준점: θ_manual + σ_post=0.02):

| 파라미터 | ±20% 섭동 | L_nps 변화율 범위 | L_rad 변화율 범위 |
|---|---|---|---|
| σ_post | 0.016 ↔ 0.024 | **±20% 이상 (총 69.5%)** | ≈ 0% (0.025%) |
| ring_strength | 0.096 ↔ 0.144 | **2.3%** | **±32~39% (총 72.1%)** |

- σ_post → L_nps 지배 (전 주파수 플로어 이동)
- ring → L_rad 지배 (국소 봉우리)
- pattern_diff = 69.5% − 2.3% = **67.19% → 통과**

**통과 기준 충족** → v5e 착수 승인.

---

## 3. 지시 28-C: v5e 재캘리브레이션

**① 지시 ID·상태**: 지시 28-C (HANDOFF v6.4 §1.3) — **완료**

**② 산출물 경로**:
- `phase4/run_phase4_v5e.py`
- `phase4/theta_real_v5e.json`
- 커밋: `af590f9`

**③ 동결/자유 구성**:

| 구분 | 파라미터 |
|---|---|
| 동결 (Tier 1) | N=300, σ_g=0.0408, p=0.001 |
| 동결 (v6.2 판정) | out_low=0.10, out_high=0.90 |
| 동결 (v6.3 ESF) | **fwhm=2.0956** |
| 자유 | alpha, in_low, in_high, **σ_post**, ring_strength, ring_n |

Seeds: 990000 / 991000 / 992000 | n_steps=2000

**④ v5d↔v5e 대조표**:

| 파라미터 | θ_manual | v5d (fwhm 동결, σ_post=0) | v5e (fwhm+σ_post 자유) | cv (v5e) |
|---|---|---|---|---|
| fwhm | 2.500 | **2.0956 (동결)** | **2.0956 (동결)** | — |
| alpha | 0.350 | 0.0564 | **0.2960 ± 0.0010** | 0.003 ✓ |
| in_low | 0.050 | 0.2702 | **0.1700 ± 0.0008** | 0.005 ✓ |
| in_high | 0.900 | 0.7000 | **0.7000 ± 0.0000** | **고착** ✗ |
| σ_post | 0.0 | 0.0 (없음) | **0.0169 ± 0.0005** | 0.028 ✓ |
| ring_strength | 0.120 | 0.1247 | **0.0301 ± 0.00004** | 0.001 ✓ |
| ring_n | 8.0 | 7.949 | **8.017 ± 0.004** | 0.001 ✓ |

**⑤ loss 비교**:

| | v5d | v5e | Δ |
|---|---|---|---|
| rep mean loss | 0.2067 | **0.06938** | **−66.4%** |

(σ_post 도입으로 loss 대부분 회복)

**⑥ 사전 예측 적중 여부** (HANDOFF v6.4 §1.3 검증 요구):

| 예측 항목 | 예측 | 실제 | 적중 |
|---|---|---|---|
| ring ≤ 0.03 복귀 | ✓ | 0.0301 (0.0001 초과) | **미적중** |
| in_high 내부점 이탈 | ✓ | 0.7000 (여전히 고착) | **미적중** |
| alpha 정상 범위 회복 | ✓ | 0.296 | **적중** |

→ σ_post 도입이 ring·alpha 병리적 보상 일부 해소(ring 0.1247→0.0301)했으나 완전 해소 실패. in_high 고착은 구조적 한계로 남음.

---

## 4. §2 판정표 적용 → D_calib FINAL

**HANDOFF v6.4 §2 판정표 (위에서부터 최초 참인 행)**:

| 행 | 조건 | 실제 값 | 판정 |
|---|---|---|---|
| 1 | σ_post > 0.08 | σ_post=0.0169 (해당 없음) | 미적용 |
| 2 | 전원 내부점 + rep일치 + ring≤0.03 | in_high 고착 + ring=0.0301>0.03 | 미적용 |
| **3** | 그 외 전부 | **해당** | **→ FINAL (fallback)** |

**D_calib 최종 구성** (`approved_by: "HANDOFF_v6.4 §2 행3-fallback"`):

| 파라미터 | 값 | 출처 |
|---|---|---|
| N | 300 | Tier 1 동결 |
| σ_g | 0.0408 | Tier 1 동결 |
| p | 0.001 | Tier 1 동결 |
| **fwhm** | **2.0956** | ESF 보정값 (v6.3) |
| alpha | 0.35 | θ_manual |
| in_low | 0.05 | θ_manual |
| in_high | 0.90 | θ_manual |
| out_low | 0.10 | θ_manual (v6.2 동결) |
| out_high | 0.90 | θ_manual (v6.2 동결) |
| **σ_post** | **0.0** | 전작 파이프라인 부재 (bit-identical 보장) |
| ring_strength | 0.12 | θ_manual |
| ring_n | 8.0 | θ_manual |

논문 스코프: "잔여 파라미터 NPS 캘리브레이션은 모델 구조 한계 잔존으로 미완 (Limitations)"

---

## 5. P4.2 TOST 결과

**① 실행 사양**:

| 항목 | 값 |
|---|---|
| 비교 쌍 | D_calib (fwhm=2.0956, σ_post=0) vs D_manual (fwhm=2.5) |
| runs/condition | 20 |
| seeds | calib: 700000~700019 / manual: 710000~710019 |
| epochs | 41 |
| batch_size | 4 |
| TOST δ | 0.004 |
| TOST α | 0.05 |

**② F1 원자료 (seed별)**:

| run | calib F1 | manual F1 |
|---|---|---|
| 0 | 0.9588 | 0.9544 |
| 1 | 0.9613 | 0.9558 |
| 2 | 0.9607 | 0.9572 |
| 3 | 0.9531 | 0.9495 |
| 4 | 0.9525 | 0.9556 |
| 5 | 0.9555 | 0.9535 |
| 6 | 0.9521 | 0.9325 |
| 7 | 0.9531 | 0.9548 |
| 8 | 0.9303 | 0.9543 |
| 9 | 0.9470 | 0.9564 |
| 10 | 0.9471 | 0.9477 |
| 11 | 0.9570 | 0.9566 |
| 12 | 0.9601 | 0.9169 |
| 13 | 0.9535 | 0.9421 |
| 14 | 0.9532 | 0.9446 |
| 15 | 0.9597 | 0.9502 |
| 16 | 0.9596 | 0.9392 |
| 17 | 0.9608 | 0.9505 |
| 18 | 0.9618 | 0.9352 |
| 19 | 0.9557 | 0.9546 |
| **mean** | **0.9546** | **0.9481** |
| **std** | **0.0071** | **0.0101** |

**③ TOST 검정 결과**:

| 검정 | 통계량 | p값 | 판정 |
|---|---|---|---|
| lower (μ_diff > −δ) | t = 3.399 | p = 0.0015 | **PASS** |
| upper (μ_diff < +δ) | t = 0.825 | p = 0.7903 | **FAIL** |
| **등가성 판정** | | | **미확인** |

```
mean_diff = 0.9546 − 0.9481 = +0.0066 (δ=0.004 초과)
```

**④ 해석**:

- D_calib가 D_manual 대비 F1 **+0.0066** 우위 (δ=0.004 경계 밖)
- "두 조건이 동등하다"를 입증 실패 — 단, 실패 방향이 **D_calib 우위 방향**임
- 즉, 파이프라인 캘리브레이션(fwhm ESF 보정)이 D_manual 대비 분류 성능을 유의미하게 향상시키는 경향 관찰
- HANDOFF v6.4 §2에는 TOST FAIL 시 처리 절차 미규정 → **검증 인스턴스 판단 필요**

---

## 6. 커밋 목록

| 커밋 | 내용 |
|---|---|
| `ce2f5d0` | v6.4 §1.1-§1.2: σ_post(S5.5) 파이프라인 추가 + 테스트 + 민감도 체크 |
| `af590f9` | v6.4 §1.3+§2: v5e 완료 → D_calib FINAL (행3-fallback) |
| `b0c41d7` | P4.2 TOST 스크립트 + 런처 (HANDOFF v6.4 §2) |

---

## 7. 미완료 항목 (검증 인스턴스 판단 대기)

| 항목 | 상태 | 사유 |
|---|---|---|
| P4.2 TOST FAIL 처리 | **PENDING_CONSULTATION** | HANDOFF v6.4 §2에 FAIL 케이스 판정 기준 미규정. D_calib 우위 방향임을 포함해 판단 요청. |
| Phase 5 50 runs 중간 보고 | **보류** | 50 runs 미도달 (진행 중) |

---

*승인 대기: P4.2 TOST FAIL — μ_diff=+0.0066 > δ=0.004, upper test p=0.7903 (FAIL). D_calib 우위 방향. 후속 판정 지침 요청.*
