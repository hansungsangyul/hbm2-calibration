# Phase 4 MANIFEST — Real Image Calibration

## Environment
- Date: 2026-07-08
- Python: /mnt/gstore/home/xiangyue/HBM_paper/envs/hbm_cudnn/bin/python (PyTorch 2.2.2+cu118)
- Commit: 8167a91 "Phase 4 v3 FAIL: N-sigma_g non-identifiability conclusively demonstrated"
- Prior commit: 899aab1 "Initial commit: diffpipe Phase 1-4 results (pre-git retrospective)"

## Source Files
| File | Description |
|------|-------------|
| run_phase4.py | v1 — baseline calibration (no constraints, no pre-norm) |
| run_phase4_v2.py | v2 — σ_g projected gradient constraint [1e-4, 0.042] |
| run_phase4_v3.py | v3 — sigmoid-box N reparameterization + pre-normalization (BUG: N display only) |

## Critical Bug — v3 N Parameterization Mismatch
The v3 script implements a sigmoid-box N reparameterization for DISPLAY purposes:
`N_display = 50 + 950 × sigmoid(N_raw)` — but pipeline.py's forward still uses `N_pipeline = F.softplus(N_raw)`.

**Consequence**: v3 N display (997.7) ≠ pipeline N (≈6.0). The "three-attractor" claim is incorrect. Corrected:
- v1: N_pipeline = 278.8 (genuine third attractor)
- v2: N_pipeline = 5.56 (σ_g clamp drives N down via non-identifiability)
- v3: N_pipeline ≈ 6.0 (same attractor as v2; display only shows sigmoid-box N=997.7)

## v1 Results (seeds 900000/901000/902000)
| Param | θ_manual | θ̂_v1 | rel_err |
|-------|----------|--------|---------|
| N | 300 | 278.8 | 7% |
| σ_g | 0.041 | 0.192 | 370% |
| fwhm | 2.500 | 1.753 | 30% |
| alpha | 0.350 | 0.547 | 56% |
| in_low | 0.050 | -0.058 | NON-PHYSICAL |
| Gate | — | FAIL (non-physical params) |

## v2 Results (seeds 910000/911000/912000) — σ_g clamped to [1e-4, 0.042]
| Param | θ_manual | θ̂_v2 | rel_err |
|-------|----------|--------|---------|
| N | 300 | 5.56 | 98% |
| σ_g | 0.041 | 0.042 (clamped) | 3% |
| fwhm | 2.500 | 1.793 | 28% |
| alpha | 0.350 | 0.545 | 56% |
| Gate | — | PASS (ratio=1.40 @ corrected single-synth measurement) |
**Note**: Original v2 gate measurement used 8-rep averaged synth (ratio=0.57 PASS — artifact). Corrected single-synth measurement: ratio=1.40 PASS.

## v3 Results (seeds 920000/921000/922000) — pre-normalization + sigmoid-box N (BUG)
| Param | θ_manual | θ̂_v3_display | N_pipeline(actual) |
|-------|----------|--------|---------|
| N | 300 | 997.7 (sigmoid-box) | ≈6.0 (softplus, actual pipeline value) |
| σ_g | 0.041 | 0.008 | — |
| fwhm | 2.500 | 2.503 | ✓ converged (pre-norm effect) |
| alpha | 0.350 | 0.525 | identifiable |
| in_low | 0.050 | 0.003 | calibrated |
| in_high | 0.900 | 0.854 | calibrated |
| Gate | — | FAIL (N_raw at upper clamp boundary) |

## Identifiability Analysis
### N-σ_g Non-Identifiability
The calibration loss cannot separate N and σ_g contributions. The loss landscape has a degenerate ridge where any (N, σ_g) satisfying σ²_total(N,x) + σ²_g = const achieves the same loss value.

### Demonstrated Attractors (corrected)
1. **Attractor A** (v1): N≈279, σ_g≈0.192 — high N/high σ_g region
2. **Attractor B** (v2, v3): N_pipeline≈5-6, σ_g≈0.042 (clamped) — low N, driven by σ_g clamp

### Identifiable Parameters
- **fwhm**: with pre-normalization (v3), converges to 2.503 ≈ manual 2.500. Anchor test (perturbed init ±1.0) in progress.
- **alpha**: consistently ≈0.52-0.55 across all versions (identifiable, shifted from manual 0.35 due to domain gap)
- **in_low, in_high**: v3 values (0.003, 0.854) used in D_calib

## Gate Summary
| Version | Gate | Primary Finding |
|---------|------|-----------------|
| v1 | FAIL | Non-physical in_low/in_high |
| v2 | PASS | N=5.6 (physically absurd), σ_g clamped |
| v3 | FAIL | N_raw boundary-pinned (N_display=998, N_pipeline≈6) |

## Corrected Identifiability Trajectory
| Param | θ_manual | v1 | v2 | v3(pipeline) | v1_relE | v2_relE | v3_relE |
|-------|----------|----|----|--------------|---------|---------|---------|
| N | 300 | 278.8 | 5.56 | 6.0 | 7% | 98% | 98% |
| σ_g | 0.041 | 0.192 | 0.042 | 0.008 | 370% | 3% | 80% |
| fwhm | 2.500 | 1.753 | 1.793 | 2.503 | 30% | 28% | 0.1% |
| alpha | 0.350 | 0.547 | 0.545 | 0.525 | 56% | 56% | 50% |

## Diagnostics
- **v2 flat noise re-measurement** (ring_absorption_test.json): ring_strength=0 → flat noise 0.997× (no change). Ring NOT absorbing noise.
- **Content leakage analysis**: template content std in real flat mask regions = 0.006 (vs real noise 0.034) — negligible.
- **PSF-blurred template**: flat region std reduces further to 0.002-0.005 after PSF blur.
- **Residual noise gap**: synth flat noise (0.037) vs physics prediction (0.005) — gap explained by actual N_pipeline≈6 (not 998).

## Output Files
| File | Description |
|------|-------------|
| theta_real.json | v1 deprecated (non-physical params) |
| theta_real_v1_deprecated.json | v1 results (archived) |
| theta_real_v2.json | v2 results + corrected gate (PASS) |
| theta_real_v3.json | v3 results + gate (FAIL) |
| identifiability_trajectory.csv | N/σ_g/fwhm/alpha trajectory across v1/v2/v3 (N corrected) |
| ring_absorption_test.json | ring=0 test + content leakage analysis |
| rep_v3_{0,1,2}/calib_trace.csv | Per-step θ trajectories for v3 reps |
| fwhm_anchor_test.json | fwhm perturbed-init verdict (in progress) |
| run_phase4.log | v1 log |
| run_phase4_v2.log | v2 log |
| run_phase4_v3.log | v3 log |
| fwhm_anchor_test.log | perturbed-init run log |
| MANIFEST.md | This file |

## Notes / Anomalies
- v3 N parameterization bug (sigmoid-box display vs softplus pipeline) renders the "N=997.7 third attractor" claim invalid. Actual v3 uses N_pipeline≈6 (same as v2).
- fwhm convergence to 2.503 is valid: fwhm_raw uses softplus correctly; pre-norm shifts the clean template mean to match real image mean, enabling fwhm convergence.
- The warm-start V-M R²=0.103 (from phase2/warmstart.json) is a pre-registered identifiability indicator: low R² predicts identifiability difficulty (confirmed by Phase 4 failures).
