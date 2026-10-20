# Representative checkpoints (not part of the git tree)

One terminal U-Net checkpoint (run 0 of 20) per training configuration of Section VI of the paper,
with the run's `results.json`. Files are PyTorch `state_dict`s (~124 MB each); see `SHA256SUMS.txt`.

| File | Configuration (paper) | Source run directory |
|---|---|---|
| D_manual_run0_final.pth | D_manual | diffpipe/phase4/p42r_v2_runs/manual/run0 |
| D_calib_run0_final.pth | D_calib | diffpipe/phase4/p42r_v2_runs/calib/run0 |
| AB-1_run0_final.pth | AB-1 (Prenorm only) | diffpipe/phase4/p43_runs/ab1/run0 |
| AB-2_run0_final.pth | AB-2 (Blur only 1) | diffpipe/phase4/p43_runs/ab2/run0 |
| AB-3_run0_final.pth | AB-3 (Blur only 2) | diffpipe/phase4/p43_runs/ab3/run0 |
| D_v5e_run0_final.pth | D_v5e (D_full, full v5e output) | diffpipe/phase4/v77/exp3b_runs/run0 |
| D_sigma_post_run0_final.pth | D_sigma_post (D_σp) | diffpipe/phase4/v77/exp3_runs/run0 |
| zscore_run0_final.pth | z-score | diffpipe/phase4/v77/exp4_runs/run0 |

These are distributed as release assets / a separate Zenodo record because of their size.
