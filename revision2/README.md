# revision2 - correction-stage analyses (release v1.1.0)

Added after the provenance of the reference set R was established (manuscript Section III-B; Supplementary S8.13 and S9). The reference set previously described as measured defect-free images is synthetic (see `reference_set_R/R_SPECIFICATION.md`).

| Path | Contents | Paper |
|---|---|---|
| `scripts/A8_PAIRED_PREREG.md` | Pre-registration of the paired cross-pipeline evaluation, written after the whole-image analyses of R (Supp. S8.8) and before any A-degraded result was inspected (commit 2e3e610 of the working repository) | Sec. VI-E, App. A, Supp. S8.12 |
| `scripts/a7_labels_eval2.py` | Tile-label metrics (normal-tile false-alarm rate, defect-tile recall, class-correct recall at tau = 0.25/0.5/0.75; normal-tile pixel FPR) on TEST_ID, R 0001-0050 and R 0251-2000 for all 8 configurations x 20 runs | Sec. VI-E, Supp. S8.13 |
| `scripts/a8_paired_gen.py` | Generation of the A-degraded counterpart of R's held-out clean images (images not deposited): each configuration's own training theta and prenorm placement, epsilon seeded per image (900000 + n) | Sec. VI-E |
| `scripts/a8_paired_eval.py` | Evaluation of all checkpoints on the A-degraded arm | Sec. VI-E |
| `scripts/a8_aggregate.py` | Per-run medians over images -> median/IQR over runs; paired differences; Table 9 and Supp. S8.13 | Table 9 |
| `scripts/a8_fig2_pair.py` | Fig. 5 (image_0251; runs selected by the median A-side false-alarm rule) | Fig. 5 |
| `results/a7_labels_v2_per_image.csv.gz` | Per image x run x configuration x input mode metrics (552,000 rows) | Table 9, Supp. S8.13 |
| `results/a8_paired_per_image.csv.gz` | A-degraded arm, per image x run x configuration (280,000 rows) | Table 9, Supp. S8.13 |
| `results/a8_summary.json`, `results/a8_summary.md` | Aggregates (declared input mode: raw for all configurations, per-image z-score for the z-score configuration) | Table 9, S8.13 |
| `results/fig2_pair.json`, `results/fig2_runs.json`, `results/a8_gen_meta.json` | Fig. 5 composition, selected runs, generation metadata (theta per configuration) | Fig. 5 |
| `reference_set_R/R_tile_labels.zip` | Tile-level label files of R (2,000 JSON files; no image content) | Sec. III-B |
| `reference_set_R/R_SPECIFICATION.md` | Generator, template weights, degradation stages and per-image parameter distributions of R | Sec. III-B, Supp. S9 |

Reproduce Table 9 from the deposited per-image files:
```
python revision2/scripts/a8_aggregate.py   # adjust the input paths at the top of the script to results/*.csv.gz
```
Re-running the evaluations requires the confidential images and the archived checkpoints. Commit 2e3e610 identifies a record in the private analysis repository. The public deposit provides the protocol-file copy; the private commit history is not included in the public repository. Appendix A and Supplementary S8.12 describe the chronology evidence and its limits.
