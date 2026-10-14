# CC 완료 보고서 v6.3
## 근거: HANDOFF_v6.3_esf_ruling.md
## 날짜: 2026-07-09

---

## 요약

| 지시 | 상태 | 결과 |
|---|---|---|
| 지시 27 (v5d: fwhm=2.095 동결 재캘리브) | **완료** | §3 행 1 발동 → PENDING_CONSULTATION |
| D_calib FINAL 전이 | **보류** | §3 행 1 (in_high 경계 고착) + 행 2 (ring>0.03) |
| P4.2 TOST | **미착수** | D_calib FINAL 선행 조건 미충족 |
| Phase 5 | **진행 중** | 계속 실행 중 (D_calib과 무관) |

---

## 1. 지시 27 (v5d) — fwhm ESF 보정값 동결 재캘리브레이션

**① 지시 ID·상태**: 지시 27 (HANDOFF v6.3 §2) — **완료**

**② 산출물 경로**:
- `phase4/run_phase4_v5d.py`
- `phase4/theta_real_v5d.json`
- `phase4/rep_v5d_0/calib_trace.csv`, `rep_v5d_1/`, `rep_v5d_2/`
- `phase4/run_phase4_v5d.log`
- 커밋: `b4be212`

**③ 핵심 수치**:

fwhm 보정 계산:
```
fwhm_frozen = 2.057 × (2.500 / 2.454) = 2.0956
```

| 파라미터 | θ_manual | v5c (fwhm 자유) | v5d (fwhm=2.0956 동결) | 판정 기준 |
|---|---|---|---|---|
| fwhm | 2.500 | 1.3563 | **2.0956 (동결)** | — |
| alpha | 0.350 | 0.3171 | **0.0564 ± 0.0002** | cv=0.003 ✓ |
| in_low | 0.050 | 0.1457 | **0.2702 ± 0.0007** | cv=0.003 ✓ |
| in_high | 0.900 | 0.7000 | **0.7000 ± 0.0000** | **경계 고착** ✗ |
| ring_strength | 0.120 | 0.0284 | **0.1247 ± 0.003** | **> 0.03** ✗ |
| ring_n | 8.0 | 7.9314 | 7.949 ± 0.009 | cv=0.001 ✓ |
| out_low | 0.10 | 0.10 (frozen) | 0.10 (frozen) | — |
| out_high | 0.90 | 0.90 (frozen) | 0.90 (frozen) | — |

Loss 비교:

| | v5c | v5d | Δ |
|---|---|---|---|
| 최종 loss (rep mean) | 0.04013 | 0.20670 | **+0.16657 (+415.1%)** |

(fwhm을 NPS 최적값 1.356에서 ESF값 2.096으로 이동했으므로 loss 악화는 예상 범위)

소요 시간: **65.8분**

**④ 트리거 발동 여부**:

**§3 행 1 발동 (최우선)**:
- `in_high = 0.7000 ± 0.0000` → bound [0.70, 1.00]의 하한 고착 (`|θ̂ − bound| = 0.000 < 0.05×(1.00−0.70)=0.015`)
- `§3 행 2도 해당`: `ring_strength = 0.1247 > 0.03`
- 행 1이 최초 참 → **PENDING_CONSULTATION**

---

## 2. §3 판정표 적용 결과

**아래 표를 위에서부터 검사, 최초 참인 행 적용 (HANDOFF v6.3 §3)**:

| 행 | 조건 | 실제 값 | 판정 |
|---|---|---|---|
| **1** | alpha/in_low/in_high 경계 고착 or rep 불일치 | **in_high=0.7000 (하한 고착)** | **→ PENDING** |
| 2 | ring_strength > 0.03 | ring=0.1247 (해당) | (행 1 우선, 미적용) |
| 3 | 그 외 전부 | — | (미적용) |

**적용 행: 행 1** → `D_calib status: PENDING_CONSULTATION`

---

## 3. 물리 해석 (보고 목적)

fwhm=2.095 동결 시 파라미터 거동:

- **in_high 하한 고착 (0.7000)**: in_high bound를 [0.70, 1.00]으로 설정했는데 최적해가 정확히 하한에 고착. fwhm이 크면 합성 이미지가 더 밝아져 대비 곡선(in_high)을 낮추려는 방향으로 압력이 작용한 것으로 보임. bound 재정의 또는 원인 분석 필요.
- **ring_strength 급등 (0.028→0.125)**: fwhm=2.095의 PSF가 고주파를 많이 suppression하므로 ring 아티팩트를 키워 NPS 고주파 성분 보충 시도. v5(fwhm 자유)에서 ring=0.101이었던 것과 유사한 기전.
- **alpha 붕괴 (0.317→0.056)**: alpha(감마 보정 강도)가 거의 0으로 수렴. 전체 loss 0.207 수준에서 다른 파라미터들이 이미 loss budget 대부분을 차지.

---

## 4. Phase 5 상태 (HANDOFF v6.3 §4 — 계속 진행)

**① 상태**: 진행 중

**② 현재 위치**: sg0_fw2 진행 중 (확인 필요)

**③ OOM 조치 관련 (§4 요구사항)**:
- batch_size 32 → 16 변경 (OOM fix, 커밋 `ac76b9c`)
- 파일럿(batch 32)과 배치 달라짐 → run당 시간·수렴 특성 변화 중간 보고에 포함 예정
- λ_max 고정 배치(32장 인덱스, seed 950000)는 학습 배치와 무관하게 유지 ✓

**④ 50 runs 중간 보고**: 아직 미도달, 도달 시 보고 예정

---

## 5. 커밋 목록

| 커밋 | 내용 |
|---|---|
| `d39669f` | v5c 스크립트, param_registry 업데이트, deprecated 파일 |
| `dc483bf` | v5c-b 완료, §6 행 5 PENDING, 보고서 v6.2 |
| `b4be212` | v5d 완료, §3 행 1 PENDING, d_calib_config 업데이트 |

---

## 6. 미완료 항목 (검증 인스턴스 판단 대기)

| 항목 | 상태 | 사유 |
|---|---|---|
| D_calib FINAL 발행 | **PENDING_CONSULTATION** | §3 행 1: in_high 경계 고착 (v5d) |
| P4.2 TOST | 미착수 | D_calib FINAL 후 착수 (HANDOFF v6.3 §3) |

---

*승인 대기: §3 행 1 PENDING — in_high=0.7000 경계 고착 + ring_strength=0.1247 검증 인스턴스 판단 필요*
