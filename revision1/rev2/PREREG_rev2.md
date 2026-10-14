# Revision-1 round 2 experiments A7?밃11 (pre-specified 2026-09-26, committed before any run)
Source of agreement: Claude?밅odex consensus plan (C:\claude\HBM2_revision\CONSENSUS_PLAN_20260926.md). No new training.

## dirty2 split (fixed now, before any look at new outputs)
0001??050 = calibration statistics set (existing) 쨌 0051??250 = ESF reserve (A8) 쨌 0251??000 = FPR held-out (A7). Same acquisition campaign; specimen/volume grouping and view overlap unknown.

## A7 real-image FPR, all 8 configurations 횞 all runs (a7_fpr.py)
Preprocessing per configuration as in Table 7 (prenorm: D_calib, AB-1, D_v5e, D_?p; z-score: z-score; none: D_manual, AB-2, AB-3). Pixel FPR over 314횞512 valid pixels. Report per config 횞 set: median, p5, min FPR, mean background px, fraction of images with FPR < 0.9.
Decision: none (descriptive). The paper's statement "every configuration median FPR 1.00" is retained only if every config's held-out median ??0.995; otherwise the exact values replace it.

## A8 ESF on reserved block 0051??250 (a8_esf.py = run_esf_psf.py with only the image list and output path changed)
Frozen v3 rules: 짹8 px profile, ??0 valid edge fits, component length ??0, RMS ??.0, R짼 0.95 with the script's existing relaxation to 0.90; bias factor from self-validation (2.5/2.454). Report per-image medians, rejection rate, pooled median, image-level bootstrap interval, heterogeneity (IQR of per-image medians).
Pre-commitment: downstream anchor remains f = 2.096 regardless of A8. A8 is a measurement-heterogeneity check; estimates outside [1.941, 2.5] are disclosed with the statement that their downstream effect was not tested.

## A9 tile-level detection (a9_tiles.py, a9_analysis.py)
Unit: microbump tile (20횞5 grid, label generator assigns one class per tile). Configs D_calib, D_manual, AB-1; TEST_ID, TEST_OOD; 20 runs each. Tile score_c = mean softmax prob of class c over the tile's valid pixels.
Detection for defect class c := score_c ???_c (independent of argmax). Report per class: AP (one-vs-rest), max recall with FP tiles/image ??0.01 and ??0.05 (ties admitted/excluded together), operating point recomputed within each bootstrap replicate; nested bootstrap B=1000 over runs 횞 images. Coverage terciles fixed from labels only (a9_coverage_bins.json, computed before any prediction): crack [0.0099, 0.01417], hole [0.02101, 0.02967], open [0.04688, 0.06415]; recall by tercile at argmax assignment (descriptive). All results synthetic-label only; no real-defect sensitivity is established.

## A10 alternative-simulator stress sets (a10_stress.py)
Same 500 originals, 罐_manual. (a) test_tailpsf: PSF = 0.8쨌Gaussian(FWHM 2.5) + 0.2쨌Cauchy-type tail (same FWHM scale), 31횞31, seeds 530000+i. (b) test_postnoise: ?_post = 0.017 with 罐_manual, seeds 540000+i. Evaluate all 8 configs 횞 20 runs, dataset-level 4-class and defect-only macro F1. Descriptive; question: does the ordering D_calib vs D_manual under TEST_OOD persist under a differently-shaped perturbation? 1.5횞 is labelled an unvalidated stress level.

## A11 identifiability robustness
A11a (a11_profile.py): N-profile N ??{50,100,300,600,1000}, ?g쨌f쨌Tier2 free, seed 970000, weights W1 = (1, 2.841, 22.79, 233.3) and W2 = (1, 0.01574, 15.74, 37.02) (W2 = A1's gradient-norm weights of seed 971000, the first logged). Decision: profile flatness (range/min < 0.5) and ?g compensation direction under each weighting, reported with the baseline.
A11b (a11_profile.py --free_N): joint multistart in the free regime, 8 starts N_init {100,800} 횞 ?g_init {0.02,0.08} 횞 f_init {1.5,3.0}, seed 970000, default weights. Report reached (N, ?g, f), loss.
A11c (a11c_kkt_hessian.py): at v5c, v6-free and the 8 A11b points: finite-difference gradient and projected-gradient norm (active N bounds 50/1000), Hessian in (log N, log ?g, log f) at h ??{0.025, 0.05, 0.1} with common random numbers. Curvature described as local objective geometry at reached points; eigenvalue stability across h reported.
