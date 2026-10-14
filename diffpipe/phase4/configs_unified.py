#!/usr/bin/env python3
"""
Minor 7 통합 구성표 (HANDOFF v7.6 §3.4, 지시 69)
===================================================
전 실험(캘리브레이션·P4.2r·P4.3·Phase5)의 config 1표
출력: phase4/configs_unified.md
"""
from pathlib import Path

OUT_MD = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/configs_unified.md')

FWHM_FROZEN = 2.057 * (2.500 / 2.454)

UNIFIED_TEXT = f"""# Minor 7 — 통합 구성표

**실행일**: 자동 생성 (HANDOFF v7.6 §3.4, 지시 69)
**목적**: 전 실험의 config 1표 (batch/epoch/prenorm/시드/평가셋)

---

## 전체 실험 구성 통합표

| 실험 | 버전 | 목적 | batch | epochs/steps | prenorm | fwhm | 시드 | 평가셋 | 스크립트 |
|------|------|------|-------|-------------|---------|------|------|--------|---------|
| 캘리브레이션 v1 | phase4/v1 | 무제약 자유최적화 | 8 | 2000 steps | ✗ (전처리 없음) | Free | 910000/911000/912000 | CalibLoss (50 real) | run_phase4_v2.py |
| 캘리브레이션 v2 | phase4/v2 | σ_g·in_low·in_high 제약 | 8 | 2000 steps | ✗ | Free | 910000/911000/912000 | CalibLoss (50 real) | run_phase4_v2.py |
| 캘리브레이션 v5e | phase4/v5e | σ_post 도입, Tier1 동결 | 8 | 2000 steps | ✓ (a=0.501834, b=0.173320) | {FWHM_FROZEN:.4f} (frozen) | 990000/991000/992000 | CalibLoss (50 real) | run_phase4_v5e.py |
| **v6-free** | phase4/v6free | 전파라미터자유+σ_post (HANDOFF v7.6 §2) | 8 | 2000 steps | ✓ | Free | 910000/911000/912000 | CalibLoss (50 real) | run_phase4_v6free.py |
| P4.2r D_calib | phase4/p42r_v2 | 캘리브 파이프라인 검출기 학습 | 32 | 41 | ✓ (a=0.501834, b=0.173320) | {FWHM_FROZEN:.4f} | 750000–750019 | test_id/ood/cal | run_p42r_v2.py |
| P4.2r D_manual | phase4/p42r_v2 | 수동 파이프라인 검출기 학습 | 32 | 41 | ✗ | 2.5 | 760000–760019 | test_id/ood/cal | run_p42r_v2.py |
| **P4.3 AB-1** | phase4/p43 | 절제: D_manual+prenorm (HANDOFF v7.6 §1) | 32 | 41 | ✓ | 2.5 | 770000–770019 | test_id/ood/cal | run_p43_ablation.py |
| **P4.3 AB-2** | phase4/p43 | 절제: D_manual+f변경 | 32 | 41 | ✗ | {FWHM_FROZEN:.4f} | 780000–780019 | test_id/ood/cal | run_p43_ablation.py |
| **P4.3 AB-3** | phase4/p43 | 절제: D_calib−prenorm | 32 | 41 | ✗ | {FWHM_FROZEN:.4f} | 790000–790019 | test_id/ood/cal | run_p43_ablation.py |
| Phase 5 Grid | phase5 | 5σ_g × 6fwhm 안정성 진단 | N/A | 20 ep | 관련 없음 | 0,0.625,1.25,2.5,5,10 | 셀별 5 reps (셀ID+오프셋) | GRV/λ_max (통계) | run_phase5_grid.py |

---

## 캘리브레이션 파라미터 상세

| 구분 | N | σ_g | fwhm | prenorm a | prenorm b | sigma_post |
|------|---|-----|------|-----------|-----------|------------|
| θ_manual | 300 | 0.0408 | 2.5 | — | — | 0 |
| v5e 동결 (Tier1) | **300** | **0.0408** | **{FWHM_FROZEN:.4f}** | 0.501834 | 0.173320 | 자유 |
| v6-free | 자유 | 자유 | 자유 | 0.501834 | 0.173320 | 자유 |

---

## P4.2r·P4.3 조건 상세

| 조건 | fwhm | prenorm | sigma_post | seed_base | N runs |
|------|------|---------|------------|-----------|--------|
| D_calib  | {FWHM_FROZEN:.4f} | ✓ | 0 | 750000 | 20 |
| D_manual | 2.5    | ✗ | 0 | 760000 | 20 |
| AB-1     | 2.5    | ✓ | 0 | 770000 | 20 |
| AB-2     | {FWHM_FROZEN:.4f} | ✗ | 0 | 780000 | 20 |
| AB-3     | {FWHM_FROZEN:.4f} | ✗ | 0 | 790000 | 20 |

공통 THETA_BASE: N=300, p=0.001, alpha=0.35, in_low=0.05, in_high=0.90,
out_low=0.10, out_high=0.90, ring_strength=0.12, ring_n=8.0, sigma_g=0.0408

---

## Phase 5 그리드 상세

| 차원 | 값 |
|------|----|
| σ_g (5단계) | 0.01025, 0.0205, 0.041, 0.082, 0.164 |
| fwhm (6단계) | 0.0, 0.625, 1.25, 2.5, 5.0, 10.0 |
| 셀 수 | 30 (5 × 6) |
| rep / cell | 5 |
| 총 runs | 150 |
| epochs / run | 20 |
| crit3 임계 | 0.10 (medfilt 3, f1_filt[-10:] max-min) |

---

## 평가셋 정의

| 세트 | 내용 | 설명 |
|------|------|------|
| test_id (in-distribution) | 동일 분포 | 학습 파이프라인과 동일 조건 생성 |
| test_ood (out-of-distribution) | 분포 외 | 다른 degradation 조건 |
| test_cal (calibration) | 교정 검증 | ESF 측정 이미지 기반 |

---

*생성: configs_unified.py (HANDOFF v7.6 §3.4)*
"""

OUT_MD.write_text(UNIFIED_TEXT)
print(f"Saved: {OUT_MD}")
print(f"\n{UNIFIED_TEXT[:200]}...")
