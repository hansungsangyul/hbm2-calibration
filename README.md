# hbm2-calibration ??code and summary data for

**Identifiability Limits and Downstream Training Value of Self-Supervised Calibration for Physics-Based Synthetic CT Data: A Diagnostic Case Study on HBM Microbump Nano-CT**
K. Rew, J. Chung, S. Lee ??submitted to IEEE Access (Access-2026-42084, resubmission).

This repository holds the differentiable degradation pipeline, the self-supervised calibrator, every launch script and configuration used in the paper, the run-level results (training trajectories, terminal metrics, per-image confusion counts, calibration traces), the pre-registration/protocol documents, and the analysis scripts that produce the numbers in the manuscript and its supplement.

**What is not included.** The real nano-CT images, the single real defect-free template from which the synthetic training and evaluation images are built, and the synthetic images themselves (which are derived from that template) are confidential and are not deposited. Only per-image summary statistics computed on real images are included. Re-generating the synthetic sets therefore requires the template; everything downstream of the saved results (all tables, figures, and statistics) can be reproduced from this repository alone. Representative trained checkpoints (one per configuration, ~1 GB) are distributed separately as release assets / a Zenodo record (see `checkpoints/README.md`).

## Layout

| Path | Contents | Paper |
|---|---|---|
| `diffpipe/src/` | `pipeline.py` (differentiable seven-stage degradation pipeline, reparameterized stochastic stages, `sigma_post` stage), `losses.py` (variance?뱈ean, NPS, radial-profile, soft-histogram losses), `calibrator.py` (warm start, Adam calibration, prenorm) | Sec. III, App. C |
| `diffpipe/configs/` | `param_registry.json` (parameter bounds, reference values, transforms), `grid_v1.json` | Sec. III, VII |
| `diffpipe/phase1/` | Gradient checks, forward-equivalence tests, Gaussian-surrogate adequacy grid | App. C, Table 8 |
| `diffpipe/phase2/` | Sensitivity matrix (`phase2b_sensitivity.py`, `sensitivity_matrix*.csv`), impulse-density estimate, warm start | Sec. III-B, Fig. 6, App. D |
| `diffpipe/phase3/` | Self-recovery experiment: `gt_params.json` (ten pre-specified truth sets), `run_phase3b.py`, per-set calibration traces, `recovery_results.json`, functional-equivalence results | Sec. IV-A, Table 3 |
| `diffpipe/phase4/` | Real-image calibration regimes v1?뱕6free (`run_phase4_v*.py`, `theta_real_*.json`, `rep_*/` traces), ESF measurement (`run_esf_psf.py`), fixed-f sweep, training campaigns: `p42r_v2_runs/` (D_calib vs D_manual, 20 runs each), `p43_runs/` (AB-1?쪨B-4), `ab1r_runs/`, `ab1rr_runs/` (equal-norm angle sweep), `v77/` (inference-time normalization, real-image FPR, D_v5e, D_sigma_post, z-score, DeepLabV3+ transfer, ESF sensitivity), statistics scripts (`p42_tost_welch.py`, `analysis_mc*.py`) | Sec. IV-B, V, VI, VII-C, App. D?밇 |
| `diffpipe/phase5/` | Noise?밷lur phase map: `run_phase5_grid.py`, `launch_grid_all.sh`, `runs/<cell>/rep*/results.json` (150 runs, epoch-level trajectories), `phase5_final_150runs.csv`, `phase5_runs_level.csv` | Sec. VII, Fig. 3 |
| `diffpipe/provenance_preregistration.md`, `campaign_closure_*`, `CC_*_report_*.md`, `RESULTS.md` | Pre-registration inventory and campaign reports used for the evidentiary audit | App. A, Table 5 |
| `f1_recompute_results/` | Dataset-level re-inference of all archived checkpoints: `counts/per_image_counts.csv` (per-run, per-image confusion counts on the three frozen test sets), `f1_dataset_level_runs.csv` | Sec. VI, Supp. S2 |
| `revision1/` | Revision-stage analyses with their pre-registered protocols (`PREREG_*.md`): `a1_weights` (loss-weight sensitivity), `a2_ident` (Hessian, N-profile, multistart), `a3_esf_low` (AB-4 at f = 1.941, bootstrap), `a4_metrics` (per-class precision/recall, nested bootstrap), `a5_realgap` (matching interventions), `boot_fix`, `holm` (Holm families), `rev2/` (a7/a7fix held-out real-image FPR on 1,750 images, a8 ESF re-measurement on 200 reserved images, a9 tile-level detection, a10 alternative stress sets, a11 profiles/joint starts/curvature) | Sec. IV-C, V, VI, App. A, Supp. S8 |
| `analysis/` | Manuscript-level aggregation scripts and their outputs (per-class F1, predicted-pixel composition, intensity statistics, real-image FPR summaries, pre-registration audit) | Sec. VI, App. D |
| `checkpoints/` | Index and SHA-256 sums of the eight representative checkpoints (binaries distributed separately) | Sec. VI |

Protocol files `revision1/*/PREREG_*.md` were damaged during a server transfer (non-ASCII text lost); the recovered UTF-8 versions are provided alongside as `*_original_utf8.md`, as disclosed in Appendix A of the paper.

## Reproducing the reported numbers

All scripts assume the repository root as working directory and the Python environment of `requirements.txt`. Calibration (phases 2??) and the training campaigns (phase 4/5, revision1) require a CUDA GPU and the confidential template, so they cannot be re-run from this deposit; their complete outputs are included. The statistics in the paper are reproduced from the saved outputs, e.g.

```
python diffpipe/phase4/p42_tost_welch.py                 # Table 6
python f1_recompute_results/analysis/*.py                # dataset-level F1 (Sec. VI, Supp. S2)
python diffpipe/phase4/analysis_mc3_stats.py             # phase-map statistics (Sec. VII-B)
python revision1/holm/holm.py                            # Holm-adjusted families (Supp. S8.7)
python revision1/rev2/a9_analysis.py                     # tile-level detection (Supp. S8.5)
```

Absolute paths of the original server (`/home/xiangyue/...`, `/mnt/gstore/...`) remain in some launch scripts and logs; they document the original runs and need to be adapted to re-run anything.

## License

Code: MIT. Data, logs and protocol documents: CC BY 4.0. See `LICENSE`.

## Citation

See `CITATION.cff`. Please cite the paper; a Zenodo DOI for this repository is given in the release notes.
