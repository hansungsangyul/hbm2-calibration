# A14 — Decomposition of the prenorm shift gain: affine shift (M) × pre-blur noise law (Q)

Status: outcome-informed follow-up. Specified after the original 2×2 ablation (Table 7: Prenorm only vs D_manual, +0.219 TEST_OOD four-class F1) and the exploratory equal-norm sweep (Appendix E) were known, after inspecting the prespecified generation-only self-test and image diagnostics, and before training any C1/C2 detector or inspecting any C1/C2 detector outcome. Registered by commit of this file and the scripts below before the launch.

## Question
Prenorm applies x' = a x + b (a = 0.501834, b = 0.173320) to clean training inputs before degradation. The same noise stages then act on a signal whose contrast is scaled by a. Does the TEST_OOD gain of Prenorm only over D_manual arise from the affine shift of the signal entering stages 3–7 (M), from the pre-blur noise law relative to the signal (Q), or from both?

## Conditions (f = 2.5, σ_post = 0, all other θ as THETA_BASE of phase4/run_p43_ablation.py)
With n0(x) = sqrt(x/N) ε1 + σg ε2 and n1(x) = [sqrt((a x + b)/N) ε1 + σg ε2]/a, the stage-3 input is
| Condition | M | Q | stage-3 input | runs |
|---|---|---|---|---|
| D = D_manual (historical, p42r_v2_runs/manual, seeds 760000+i) | no | baseline | x + n0(x) | 20 existing |
| P = Prenorm only / AB-1 (historical, p43_runs/ab1, seeds 770000+i) | yes | prenorm | a[x + n1(x)] + b | 20 existing |
| C1 "affine shift with baseline-relative noise" | yes | baseline | a[x + n0(x)] + b | 20 new, seeds 830000+i |
| C2 "prenorm-relative noise without affine shift" | no | prenorm | x + n1(x) | 20 new, seeds 840000+i |
Stages 3–7 (S&P replacement, PSF blur, vignetting, contrast window with clamp, ring) are unchanged; they transmit both factors to the final image. The factors are defined at the stage-3 input; final-image moments and SNR are reported as diagnostics and are not claimed to be matched.

Reproduction check (not gating, reported): dman_r run0 and ab1_r run0 retrain D and P with the A14 code path and the original seeds.

Protocol identical to P4.3/P4.2r §4-A: U-Net, batch 32, 41 epochs, no early stopping, post hoc crit3, same training pool, per-image fixed noise realisations (seed = image index), frozen TEST_ID/TEST_OOD/TEST_CAL files, no inference-time normalisation. Script: a14_train.py. Self-test and diagnostics before training: a14_selftest_diag.py (bit-exact equality of the A14 path with DegradationPipeline.forward for D and P; closed-form check of the C1/C2 stage-2 outputs; 50-image final-image mean/SD, stage-6 pre-clamp fractions, effective noise SD and SNR). Launch only if the self-test passes.

## Outcomes and analysis (a14_counts.py, a14_analysis.py)
Dataset-level per-class F1 from pooled TP/FP/FN (same logic as analysis/gen_counts_csv.py; historical counts from f1_recompute_results/counts/per_image_counts.csv).
Primary: TEST_OOD four-class macro F1. Secondary: TEST_OOD defect-only macro F1, TEST_OOD background F1, TEST_ID four-class macro F1. Exploratory: post hoc crit3 after 41 epochs (range of the median-filtered validation F1 over the last 10 epochs > 0.1), for all four conditions.
Contrasts: ΔM = C1 − D, ΔQ = C2 − D, I = P − C1 − C2 + D, C1 − C2, P − C1, P − C2; ratios rM = ΔM/(P − D), rQ = ΔQ/(P − D).
Inference: unequal-variance (Welch–Satterthwaite) t tests for ΔM, ΔQ, C1 − C2, and I, with Holm adjustment across these four contrasts on the primary outcome; P − C1 and P − C2 are reported as estimates with bootstrap confidence intervals. Bootstrap: runs resampled independently within each of the four conditions (B = 10000, seed 20261019), percentile 95% CIs for all contrasts and for rM, rQ with the denominator resampled jointly. P − D is an observed historical contrast and is not re-tested.
Descriptive classification on the primary outcome (reported with CIs; no claim beyond it):
- "mainly M": lower CI of rM ≥ 0.5, upper CI of rQ < 0.5, and C1 > C2 with Holm p < 0.05;
- "mainly Q": the mirror condition;
- "each factor alone reproduces at least half": lower CIs of rM and rQ ≥ 0.5;
- "neither factor alone reproduces half": upper CIs of rM and rQ < 0.5;
- otherwise "undetermined".
Interaction is interpreted only through I and its CI. All 40 new runs are reported whatever the result (the counts and analysis scripts stop if any run, epoch count, seed or test image is missing); no run, checkpoint or criterion is selected after results are seen.

## Reporting
Section VI-B (one paragraph and a small table) and a new supplementary section with the full table, diagnostics, reproduction check and statistics. The abstract is unchanged unless all three parties agree otherwise.
