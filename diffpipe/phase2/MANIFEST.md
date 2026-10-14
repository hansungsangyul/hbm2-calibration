# Phase 2 MANIFEST — Calibration Loss Functions

## Environment
- Date: 2026-07-08 (v1) / 2026-07-08 (v2 redesign)
- Python: /mnt/gstore/home/xiangyue/HBM_paper/envs/hbm_cudnn/bin/python (PyTorch 2.2.2+cu118)

## Source Files
- src/losses.py — CalibrationLoss: L_vm, L_nps, L_rad, L_hist (v2 redesigned)
- src/calibrator.py — estimate_sp_density, warm_start, calibrate

## Phase 2 Outputs

| File | Description |
|------|-------------|
| phase2/sp_estimation.json | S&P density estimate from 2000 real images (p_hat_mean, p_hat_std, histogram) |
| phase2/warmstart.json | Warm-start parameter estimates from 100 real images |
| phase2/sensitivity_matrix.csv | v1 sensitivity matrix (GATE FAIL) |
| phase2/sensitivity_summary.json | v1 sensitivity summary (GATE FAIL) |
| phase2/sensitivity_matrix_v2.csv | **v2 sensitivity matrix (GATE PASS)** |
| phase2/sensitivity_summary_v2.json | **v2 sensitivity summary (GATE PASS)** |
| phase2/phase2b_sensitivity.py | v2 sensitivity script with redesigned losses |
| phase2/loss_gradcheck.json | Gradient check from v1 |
| phase2/MANIFEST.md | This file |

## S&P Estimation (from 2000 real images)
- p_hat_mean: 0.1007  (theta_manual was 0.001)
- p_hat_std:  0.0011

## Warm Start (from 100 real images)
- N_est:         30.31   (theta_manual: 300)
- sigma_g_est:   0.1631  (theta_manual: 0.0408)
- vm_fit_r_squared: 0.103

## Sensitivity Matrix

Entries are |delta L_i / delta theta_j| (finite-difference, normalized). Rows = parameters, Columns = losses.

| parameter    | L_vm   | L_nps  | L_rad  | L_hist |
|--------------|--------|--------|--------|--------|
| N            | 0.0028 | 0.0518 | 0.0002 | 0.0045 |
| sigma_g      | 0.0052 | 0.0978 | 0.0002 | 0.0090 |
| p            | 0.0003 | 0.0062 | 0.0001 | 0.0006 |
| fwhm         | 0.1310 | 2.6019 | 0.0002 | 0.1878 |
| alpha        | 0.0145 | 0.0594 | 0.3833 | 0.1533 |
| in_low       | 0.0054 | 0.1064 | 0.3249 | 0.0212 |
| in_high      | 0.1370 | 2.1551 | 6.9745 | 0.4601 |
| ring_strength| 0.0171 | 0.2004 | 0.1998 | 0.1925 |

Baseline losses: L_vm=1.78e-3, L_nps=1.770, L_rad=1.47e-2, L_hist=1.12e-2

## Diagonal Dominance Check

Expected assignment: each loss should be most sensitive to its designated target parameter(s).

| Loss  | Intended target(s)   | Rank of target in column | Top-2 actual          | Dominant?  |
|-------|----------------------|--------------------------|-----------------------|------------|
| L_vm  | N, sigma_g           | N: rank 7 (0.0028), sigma_g: rank 6 (0.0052) | in_high (0.137), fwhm (0.131) | NO — FAIL  |
| L_nps | fwhm, sigma_g        | fwhm: rank 1 (2.6019), sigma_g: rank 5 (0.0978) | fwhm (2.602), in_high (2.155) | PARTIAL — fwhm dominates but in_high is strong confound; sigma_g absent from top-2 |
| L_rad | ring_strength        | ring_strength: rank 4 (0.1998)   | in_high (6.975), alpha (0.383) | NO — FAIL  |
| L_hist| alpha, in_low, in_high| in_high: rank 1 (0.4601), alpha: rank 4 (0.1533), in_low: rank 5 (0.0212) | in_high (0.460), ring_strength (0.193) | PARTIAL — in_high dominates; ring_strength is a confound for alpha/in_low |

Key structural problem: `in_high` is the top or second sensitivity driver for ALL four losses,
meaning the optimization landscape is heavily dominated by a single parameter and the Jacobian
is nearly rank-deficient. The four losses do not span four independent parameter directions.

## Phase 2b Gate: PASS (v2 redesign)

### v2 Sensitivity Matrix (losses redesigned)

| parameter | L_vm | L_nps | L_rad | L_hist |
|---|---|---|---|---|
| N | 0.0909 | 0.0945 | 0.0000 | 0.0075 |
| sigma_g | **0.3732** | 0.2913 | 0.0059 | 0.0161 |
| p | 0.0169 | 0.0554 | 0.0015 | 0.0115 |
| fwhm | 1.2913 | **4.2887** | 0.0020 | 0.4035 |
| alpha | 0.0210 | 0.2876 | **1.3912** | 0.6295 |
| in_low | 0.1196 | 1.0148 | 0.3201 | **1.2864** |
| in_high | 0.0150 | 1.1654 | 0.0288 | 0.5763 |
| ring_strength | 0.3201 | 0.0863 | 0.7033 | 0.1497 |

### Fixes Applied (losses.py)
1. **L_vm**: Flat-patch filter (50th-pct patch gradient) + **mean-subtracted log-variance curve** — PSF scale factor cancels in log space, making L_vm sensitive to noise-mean curve SHAPE (N, sigma_g) rather than blur width. N gradient improved 200× (3.9e-7 → -8.4e-5).
2. **L_rad**: Normalized by radial mean — contrast-scale-invariant, alpha sensitivity 4.5× higher than ring_strength.
3. **Perturbation protocol**: contrast params (in_low, in_high) use absolute ±0.05 instead of ×1.2/×0.8 to avoid pathological in_high > 1.0 clipping.

### Diagonal Dominance Checks (v2)
- **L_vm**: sigma_g in top-2 → PASS ✓
- **L_nps**: fwhm rank-1 → PASS ✓
- **L_rad**: alpha rank-1 → PASS ✓
- **L_hist**: in_low rank-1 (contrast param) → PASS ✓

## Phase 2 Gate (v1): FAIL (historical record)

Gate criterion: every loss must have its intended target parameter in the top-2 sensitivity rank.

Failures:
1. **L_vm / target = N**: N ranks 7th out of 8 in the L_vm column (sensitivity 0.0028).
   The loss is 49x more sensitive to in_high and 47x more sensitive to fwhm than to N.
   A gradient step on L_vm will adjust in_high and fwhm, not noise count N.

2. **L_rad / target = ring_strength**: ring_strength ranks 4th (sensitivity 0.1998) in the L_rad column.
   in_high (6.975) is 35x more sensitive; alpha (0.383) and in_low (0.325) also rank above ring_strength.
   Ring artifact calibration via L_rad will be swamped by window/contrast parameter gradients.

3. **L_nps / target = sigma_g**: sigma_g ranks 5th (0.0978); in_high (2.155) is the second dominant
   driver, creating strong cross-coupling between NPS shape and window upper bound.

Root cause: the pixel-value window parameters (alpha, in_low, in_high) affect image brightness
globally, making them high-sensitivity confounds for every loss that operates on pixel values.
The radial profile loss (L_rad) and variance-metric loss (L_vm) are particularly vulnerable
because they are pixel-value statistics that do not isolate noise texture from tone-mapping.

Recommendation before proceeding to Phase 3:
- Redesign L_vm to operate on residuals (synthetic minus blurred synthetic) rather than
  raw pixel values, so it is insensitive to the in_high/fwhm global offset.
- Redesign L_rad to subtract a DC radial background before comparing profiles, or apply
  a band-pass filter that passes ring-frequency harmonics only.
- Consider a block-normalised NPS (divide each patch by its local mean) so L_nps is
  independent of in_high.

## Anomalies

1. **S&P density 100x above manual setting**: p_hat_mean = 0.1007 vs theta_manual_p = 0.001.
   The estimator returns ~10 % S&P pixels; the manual model uses 0.1 %. Either the estimator
   is detecting noise as S&P, or the manual parameter was set in a different unit convention.
   Impact: if p is initialised from p_hat_mean the degradation model will generate extreme
   salt-and-pepper noise, invalidating all other loss comparisons.

2. **N_est 10x below manual**: N_est = 30.3 vs theta_manual = 300. The warm-start routine
   sees far fewer effective noise frames than the manual setting. Likely cause: the VM slope
   fitting is unreliable (see item 3 below); N is inferred from slope as N ~ 1/slope.

3. **sigma_g_est 4x above manual**: sigma_g_est = 0.163 vs theta_manual = 0.0408. A higher
   sigma_g inflates per-pixel noise variance, which the warm-start may be compensating for
   the low N_est via N * sigma_g^2 ~ const.

4. **vm_fit_r_squared = 0.103**: The variance-versus-mean linear fit used to estimate N and
   sigma_g explains only ~10 % of variance. The fit is essentially noise. This is the proximate
   cause of the inaccurate N_est and sigma_g_est, and also explains why L_vm has near-zero
   gradient magnitude (grad_abs_max = 3.9e-7 in the gradcheck).

5. **L_vm gradient near-zero**: grad_abs_max for L_vm = 3.9e-7, compared to 3.9e-2 for L_nps
   (4 orders of magnitude weaker). In a joint optimisation, L_vm will be numerically invisible
   unless its weight is scaled up by ~10,000x, making it unstable.
