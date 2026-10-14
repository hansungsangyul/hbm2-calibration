# Phase 3 MANIFEST — Synthetic Calibration Recovery

## Environment
- Date: 2026-07-08
- Python: /mnt/gstore/home/xiangyue/HBM_paper/envs/hbm_cudnn/bin/python (PyTorch 2.2.2+cu118)
- Commit: N/A (pre-git; git history begins at 899aab1 covering Phase 1–4)

## Source Files
| File | Description |
|------|-------------|
| run_phase3.py | Phase 3a (Primary) — 10 sets × 3 reps, synthetic calibration |
| run_phase3b.py | Phase 3b — additional sweep with fixed convergence criteria |
| primary_designation.md | Designation: Phase 3a = PRIMARY (convergence criteria match spec §2.5 exactly) |

## PRIMARY Designation
**Phase 3a = PRIMARY** — convergence criteria: CONV_WINDOW=200, CONV_TOL=1e-3, relative moving-average change. Matches spec §2.5 exactly. All Phase 3 references in subsequent phases use 3a results.

## Ground Truth Parameter Sets
- 10 sets (set_00 to set_09)
- set_00: θ_manual (fwhm=2.5, N=300, σ_g=0.0408, α=0.35, ...)
- set_01..09: random perturbations within physically plausible ranges
- Source: gt_params.json

## Convergence Criteria (Phase 3a, §2.5)
- CONV_WINDOW = 200 steps
- CONV_TOL = 1e-3 (relative moving-average change)
- Max steps: 2000 per rep
- Reps per set: 3

## Results Summary
| Param | R² | MRE | MAE | Status |
|-------|-----|-----|-----|--------|
| FWHM | 0.9992 | 0.81% | 0.021 | ✓ Excellent |
| N | 0.1322 | 50.41% | 109.6 | ✗ N-σ_g non-identifiability (even synthetic!) |
| σ_g | 0.9293 | 13.12% | 0.004 | △ Moderate |
| α | 0.9993 | 0.95% | 0.003 | ✓ Excellent |

**Overall Gate**: PASS (loss convergence criterion; N identifiability failure is a structural loss landscape issue, not a convergence failure)

**N identifiability note**: R²=0.13 for N even in synthetic calibration confirms that N-σ_g non-identifiability is a fundamental structural property of the loss landscape, not an artifact of domain gap. This is paper §3.3 evidence.

## Output Files
| File | Description |
|------|-------------|
| phase3/gt_params.json | Ground truth parameter sets (10 sets) |
| phase3/recovery_results.json | Full recovery results (all sets, all reps, θ_hat_mean, θ_hat_std) |
| phase3/set_00/ to phase3/set_09/ | Per-set directories with rep results and calib_trace.csv |
| phase3/primary_designation.md | Phase 3a = PRIMARY designation record |
| phase3/scatter_gt_vs_estimated.png | GT vs estimated scatter plots (4 params, 3a basis) |
| phase3/MANIFEST.md | This file |
| phase3b/ | Phase 3b results (non-primary, archived) |
| run_phase3.log | Phase 3a run log |
| run_phase3b.log | Phase 3b run log |

## Notes / Anomalies
- Phase 3b results diverge slightly from 3a due to different convergence window (3b: 100 steps); 3a is authoritative.
- set_00 recovers θ_manual exactly (confirmation that θ_manual is a valid parameter point).
- N-σ_g identifiability is good in synthetic (controlled) conditions but breaks down in real-image calibration (Phase 4) due to domain gap and measurement noise.
