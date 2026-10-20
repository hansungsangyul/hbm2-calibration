[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23368374.svg)](https://doi.org/10.5281/zenodo.23368374)

# hbm2-calibration — code and summary data for

**Identifiability Limits and Downstream Training Value of Self-Supervised Calibration for Physics-Based Synthetic CT Data: A Diagnostic Case Study on HBM Microbump Nano-CT**
K. Rew, J. Chung, S. Lee — submitted to IEEE Access (Access-2026-52990, corrected resubmission of Access-2026-42084).

This repository holds the differentiable degradation pipeline, the self-supervised calibrator, every launch script and configuration used in the paper, the run-level results (training trajectories, terminal metrics, per-image confusion counts, calibration traces), the pre-registration/protocol documents, and the analysis scripts for the manuscript and its supplement.

**Correction (v1.1.0, 2026-10-16).** The image set that release v1.0.0 and the resubmitted manuscript described as measured, defect-free nano-CT images (the "real images" of the calibration statistics, the ESF anchor and the false-positive evaluation) is synthetic: it was produced by the MATLAB data-generation scripts of the prior work from a bank of ten measured template crops, with procedurally inserted defects in 72 % of its tiles, and degraded by a separately implemented pipeline. The corrected manuscript calls it the reference set R and treats the calibration as calibration under degradation-model mismatch; the former real-image evaluation is replaced by a pre-registered, content-matched paired evaluation. Everything added for the correction is under `revision2/` (see `revision2/README.md` and `revision2/reference_set_R/R_SPECIFICATION.md`). File and variable names containing `real` elsewhere in this repository refer to R.

**What is not included.** The ten measured template crops and the synthetic images derived from them (the training pool and the images of R) are confidential and are not deposited. The tile-level label files of R, per-image summary statistics on R, and all run-level results are included. Re-generating the synthetic sets therefore requires the templates; the deposited run records, per-image metrics, calibration traces and analysis outputs support recomputation of the reported statistics and tables; figure panels that display images require the withheld images. Eight representative trained checkpoints, one per configuration in Table 9 (about 1 GB in total), are distributed separately as assets of GitHub release v1.0.0 (see `checkpoints/README.md`).

## Layout

| Path | Contents | Paper |
|---|---|---|
| `diffpipe/src/` | `pipeline.py` (differentiable seven-stage degradation pipeline, reparameterized stochastic stages, `sigma_post` stage), `losses.py` (variance–mean, NPS, radial-profile, soft-histogram losses), `calibrator.py` (warm start, Adam calibration, prenorm) | Sec. III, App. C |
| `diffpipe/configs/` | `param_registry.json` (parameter bounds, reference values, transforms), `grid_v1.json` | Sec. III, VII |
| `diffpipe/phase1/` | Gradient checks, forward-equivalence tests, Gaussian-surrogate adequacy grid | App. C, Table 10 |
| `diffpipe/phase2/` | Sensitivity matrix (`phase2b_sensitivity.py`, `sensitivity_matrix*.csv`), impulse-density estimate, warm start | Sec. III-C, Fig. 8, App. D |
| `diffpipe/phase3/` | Self-recovery experiment: `gt_params.json` (ten pre-specified truth sets), `run_phase3b.py`, per-set calibration traces, `recovery_results.json`, functional-equivalence results | Sec. IV-A, Table 4 |
| `diffpipe/phase4/` | Reference-set calibration regimes v1–v6free (`run_phase4_v*.py`, `theta_real_*.json`, `rep_*/` traces), ESF estimation (`run_esf_psf.py`), fixed-f sweep, training campaigns: `p42r_v2_runs/` (D_calib vs D_manual, 20 runs each), `p43_runs/` (AB-1–AB-4), `ab1r_runs/`, `ab1rr_runs/` (equal-norm angle sweep), `v77/` (inference-time normalization, whole-image predicted-defect fraction on R (Supp. S8.8), D_v5e, D_sigma_post, z-score, DeepLabV3+ transfer, ESF sensitivity), statistics scripts (`p42_tost_welch.py`, `analysis_mc*.py`) | Sec. III-D, IV-B, V, VI, VII-B–C, App. D–E |
| `diffpipe/phase5/` | Noise–blur phase map: `run_phase5_grid.py`, `launch_grid_all.sh`, `runs/<cell>/rep*/results.json` (150 runs, epoch-level trajectories), `phase5_final_150runs.csv`, `phase5_runs_level.csv`, `permtest/` (interaction permutation tests with their null distributions) | Sec. VII, Fig. 7, Supp. S5.3 |
| `diffpipe/provenance_preregistration.md`, `campaign_closure_*`, `CC_*_report_*.md`, `RESULTS.md` | Pre-registration inventory and campaign reports used for the evidentiary audit | Sec. III-E, Table 3, App. A, Supp. S8.12 |
| `f1_recompute_results/` | Dataset-level re-inference of all archived checkpoints: `counts/per_image_counts.csv` (per-run, per-image confusion counts on the three frozen test sets), `analysis/f1_dataset_level_runs.csv` | Sec. VI, Supp. S2 |
| `revision1/` | Revision-stage analyses and their protocol files (`PREREG_*.md`; Supplementary S8.12 gives the timing evidence for each): `a1_weights` (loss-weight sensitivity), `a2_ident` (Hessian, N-profile, multistart), `a3_esf_low` (AB-4 at f = 1.941, bootstrap), `a4_metrics` (per-class precision/recall, nested bootstrap), `a5_realgap` (matching interventions), `boot_fix`, `holm` (Holm families), `rev2/` (a7/a7fix whole-image predicted-defect fractions on subsets of R (Supp. S8.8), a8 ESF re-estimation on the 200 reserved images of R, a9 tile-level detection, a10 alternative stress sets, a11 profiles/joint starts/curvature) | Sec. IV-C, V, VI, App. A, Supp. S8 |
| `revision2/` | Correction-stage analyses: tile-label metrics, paired cross-pipeline evaluation (pre-registered), reference-set specification and labels | Sec. III-B, VI-E, Supp. S8.13, S9 |
| `revision3/` | Analyses added after the pre-submission review (release v1.1.2): stage-wise interventions between model A and a Python implementation of the reference pipeline (pre-registered, with an outcome-informed amendment), label-assisted support sensitivity of the calibration objective, outcome-informed decomposition of the prenorm shift gain (A14) | Sec. IV-C, VI-B, VI-E, App. A, Supp. S8.14–S8.16 |
| `analysis/` | Manuscript-level aggregation scripts and their outputs (per-class F1, predicted-pixel composition, intensity statistics, whole-image summaries on R, pre-registration audit) | Sec. III-E, VI, App. A, D, Supp. S8.8, S8.12 |
| `checkpoints/` | Index and SHA-256 sums of the eight representative checkpoints (binaries distributed separately) | Sec. VI |

File and directory names use run codes; the paper uses display names: AB-1 = Prenorm only; AB-2 and AB-3 = Blur only 1 and Blur only 2 (pooled as Blur only); AB-4 = the lower-endpoint configuration (f = 1.941 with prenorm); D_v5e = D_full; D_sigma_post = D_σp; calibration regime v5c = Frozen noise. Supplementary Section S3.2a gives the complete map.

The deposited protocol files of the five follow-up analyses designated A1–A5 in Supplementary S8.12 contain partial protocol text; for the count re-analysis (A3), timing support comes from file timestamps without commit verification. Appendix A and Supplementary S8.12 distinguish rules supported by committed text from rules supported only by companion archival records with earlier timestamps. The working-repository commit identifiers reported in the manuscript refer to the private analysis repository. The public deposit contains protocol-file copies and the audit summary in `analysis/prereg_audit.csv`; the private repository's commit history is not included, so those identifiers cannot be resolved through the public repository's Git history.

## Reproducing the reported numbers

The archived scripts use the dependencies listed in `requirements.txt`; their input and output paths must be checked and adapted to the local checkout before execution. Calibration, detector training and image-level evaluation require the confidential template crops or the images derived from them, so they cannot be re-run from this deposit. The deposited run records, per-image metrics, calibration traces and analysis outputs support recomputation of the reported statistics, for example:

```
python analysis/recompute_f1_dataset_level.py --counts-csv f1_recompute_results/counts/per_image_counts.csv --out out
                                                         # dataset-level F1 and Welch contrasts (Table 6, Sec. VI; Supp. S2)
python revision1/holm/holm.py                            # Holm-adjusted families (Supp. S8.7)
python revision1/rev2/a9_analysis.py                     # tile-level detection (Supp. S8.5)
python revision2/scripts/a8_aggregate.py                 # Table 9 (adjust input paths; see revision2/README.md)
python diffpipe/phase5/permtest/permtest_replication.py  # phase-map interaction permutation tests (Sec. VII-B, Supp. S5.3; adjust the paths at the top)
```

The phase-map run-level summary is `diffpipe/phase5/phase5_runs_level.csv`, and the threshold re-adjudication of Supplementary S4 is summarized in `diffpipe/phase4/reanalysis_mc3_mc5.md`. The seed-42 implementation in `diffpipe/phase5/permtest/` reproduces the permutation p-values reported in Section VII-B and Supplementary S5.3 to the reported precision. This directory contains its script, results and 2 × 10^5-draw null distributions. The reported analysis was checked with two independent implementations using seeds 1 and 42; this release contains the seed-42 implementation. The permutation p-values printed by `diffpipe/phase4/analysis_mc3_stats.py` and `reanalysis_mc3_mc5.md` are affected by a missing-value defect and are superseded by these results.

Absolute paths of the original server (`/home/xiangyue/...`, `/mnt/gstore/...`) remain in some launch scripts and logs; they document the original runs and need to be adapted to re-run anything.

## License

Code: MIT. Data, logs and protocol documents: CC BY 4.0. See `LICENSE`.

## Citation

See `CITATION.cff`. Please cite the paper. Archived releases: v1.0.0 (DOI 10.5281/zenodo.23368375), v1.1.0 (correction-stage material), v1.1.2 (post-review analyses, `revision3/`, with documentation aligned to the corrected manuscript), v1.1.3 (documentation update: wording and run-code map aligned with the final manuscript) and v1.1.4 (README corrections and the phase-map permutation-test files in `diffpipe/phase5/permtest/`); a release v1.1.1 published on 2026-10-20 was withdrawn the same day because it contained example images derived from the confidential template crops; concept DOI for all versions 10.5281/zenodo.23368374 (https://doi.org/10.5281/zenodo.23368374).
