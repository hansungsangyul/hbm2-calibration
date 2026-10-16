# Specification of the reference set R (manuscript Section III-B; Supplementary S9)

R consists of 2,000 synthetic images of 314 x 520 pixels generated on 2025-10-09 by the MATLAB data-generation scripts of the prior work (Lee et al., IEEE Access 2026), in two steps. The images themselves are not deposited (they embed ten measured template crops); the tile-level label files are (`R_tile_labels.zip`, `HBM_labels/label_0001.json` ... `label_2000.json`).

## 1. Clean images and labels
- Each image is a 5 x 20 grid of microbump tiles (tile 62.8 x 26 px after the x0.1 downscale of a 3140 x 5200 canvas; the evaluation code uses 62 x 26 tiles with the last row/column absorbing the remainder).
- Every tile is drawn from a bank of ten measured template crops (normal1.png ... normal10.png, 628 x 260 px each) with fixed weights w_n = exp(-0.3 (n-1)) assigned in the lexicographic file order of the bank (normal1, normal10, normal2, ..., normal9): 27.3 %, 20.2 %, 15.0 %, 11.1 %, 8.2 %, 6.1 %, 4.5 %, 3.3 %, 2.5 %, 1.8 %. Per-image tile counts are the rounded products of these weights with 100, shuffled at random.
- Each tile is assigned one of {normal, crack, hole, open} with equal nominal probability; a defective tile is produced by the framework's defect simulator with per-tile random parameters recorded in the label file; a tile whose insertion fails the simulator's acceptance test after a bounded number of retries reverts to normal.
- Over the 2,000 images: normal 27.85 %, crack 24.74 %, open 22.33 %, hole 25.08 % of tiles; every image contains at least one defective tile.
- Label file format: a JSON list of 100 objects {"label": "normal"|"crack"|"open"|"hole", "row": 1..5, "col": 1..20, "params": {...}} (params present for defect tiles only).
- The clean training pool of the manuscript (hbm_augmentation train/val/test, generated 2026-01-27) comes from the same scripts, template bank and defect settings as an independent random draw (tile-class agreement at equal image index 20-32/100 on four inspected images; chance level about 25 %).

## 2. Degradation operator (batch script, intermediate level)
Applied in order to each clean image:
1. MATLAB `imnoise(..., 'poisson')` on double-precision input (numerically negligible);
2. additive Gaussian noise of variance 1.5 g;
3. salt-and-pepper noise of density d;
4. additive Gaussian noise of standard deviation 0.6 g, then clipping to [0, 1];
5. motion blur with the MATLAB `fspecial('motion', l, phi)` kernel, circular convolution;
6. radial vignetting max(1 - s r^2 / r_max^2, b), r = distance from the image centre;
7. piecewise linear contrast mapping `imadjust` from [t_lo, t_hi] to [o_lo, o_hi];
8. JPEG encoding at quality q and decoding;
9. resampling to kappa times the image size and back to 314 x 520 (pixel coordinates preserved).

Per-image parameter draws (independent across images and parameters; N(mean, sd), clipped to the range; realized values were not stored):

| Stage | Parameter | Distribution | Range |
|---|---|---|---|
| Gaussian noise | g (variance 1.5 g; second-stage sd 0.6 g) | N(0.04, 0.012) | [0, inf) |
| Impulse noise | density d | N(0.015, 0.005) | [0, 0.05] |
| Motion blur | length l (px, rounded) | N(6, 2) | [1, inf) |
| | angle phi (deg) | N(45, 30) | - |
| Vignetting | strength s | N(0.6, 0.2) | [0, 1] |
| | floor b | N(0.6, 0.1) | [0.3, 1] |
| Contrast | t_lo, t_hi | N(0.1, 0.03), N(0.9, 0.03) | [0, 0.3], [0.7, 1] |
| | o_lo, o_hi | N(0.2, 0.05), N(0.8, 0.05) | [0, 0.4], [0.6, 1] |
| JPEG | quality q (rounded) | N(60, 10) | [40, 95] |
| Resampling | scale kappa | N(1.04, 0.015) | [0.98, 1.08] |

## 3. Splits used in the manuscript
- 0001-0050: calibration statistics (Section III-C) and the first ESF estimates (Section V).
- 0051-0250: reserved block for the revision ESF estimate (Section V); counted as calibration evidence.
- 0251-2000: held out from every calibration step; used only for the paired cross-pipeline evaluation (Section VI-E).

## 4. Attribution evidence
The attribution of R to these scripts rests on the archived scripts and file modification times (clean images and labels written 2025-10-09 07:44-10:30; the level-1 and level-2 degraded outputs written in succession 12:48-12:52; the batch degradation script last saved 12:53, set to its next level) and on the image and label files themselves (normal tiles match the ten downscaled templates with best correlation 0.994, min 0.972; template-usage frequencies agree between R and the training pool). The generation logs were not kept.
