# Calibration v7 — Causal Closure Report

## Setup

v5c + σ_post free (Tier-1 + out_* frozen, same protocol)

| Parameter | Status | Value |
|-----------|--------|-------|
| N | FROZEN | 300.0 |
| σ_g | FROZEN | 0.0408 |
| p | FROZEN | 0.001 |
| out_low/high | FROZEN | 0.1/0.9 |
| fwhm | FREE | init=2.5 |
| σ_post | **FREE (v7 추가)** | init=0.01 |
| other Tier2 | FREE | same as v5c |

## 3-Rep Results

| Rep | seed | fwhm | σ_post |
|-----|------|------|--------|
| 0 | 980000 | 1.3599 | 0.00478 |
| 1 | 981000 | 1.3560 | 0.00010 |
| 2 | 982000 | 1.3638 | 0.00685 |

fwhm median = **1.3599**  (mean=1.3599, std=0.0032)
σ_post median = **0.00478**  (v5e ref: 0.01695)

## 판정

**인과 미폐쇄 — f 수렴점 v5c 근방 잔류**

| 기준 | fwhm 범위 | 판정 | 원고 처리 |
|------|-----------|------|----------|
| ≥ 1.7 | ESF 방향 | 인과 폐쇄 | 기여 2 완성 |
| 1.45-1.7 | 부분 이동 | 부분 이동 | 이동량 보고 |
| < 1.45 | v5c 근방 | 인과 미폐쇄 | L-1(B) 채택 |

v7 fwhm median = **1.3599** → **인과 미폐쇄 — f 수렴점 v5c 근방 잔류**

## M-3 λ Scaling Check

| λ | fwhm | Δ vs λ=(1,1,1,1) |
|---|------|------------------|
| (1,1,1,1) | 1.3599 | baseline |
| (2,2,2,2) | 1.3641 | +0.0042 |
| (0.5,...) | 1.3518 | -0.0081 |
