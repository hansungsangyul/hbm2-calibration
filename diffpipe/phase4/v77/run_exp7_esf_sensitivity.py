#!/usr/bin/env python3
"""
HANDOFF v7.7 실험 7 — ESF 완화 기준 민감도 (M9, CPU)
====================================================
R² 임계 {0.90, 0.85, 0.80} × 최소 피팅 수 {20, 10, 5} = 9 조건
dirty2 이미지 5매 × 9 조건 → 이미지별 중앙값 FWHM + 집계

기준 (R²=0.95, min_fits=20) 현재 결과와 비교:
  전 기준에서 f_ESF ∈ [1.9, 2.3] → ROBUST
  이탈 기준 존재                  → SENSITIVE + 이탈 조건 목록

출력:
  phase4/v77/exp7_sensitivity_table.csv
  phase4/v77/exp7_summary.json
"""
from __future__ import annotations
import json, sys, csv
from pathlib import Path
from itertools import product

import numpy as np
import cv2
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.ndimage import label as nd_label, map_coordinates

# ── Paths ──────────────────────────────────────────────────────────────────────
DIFFPIPE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
REAL_DIR     = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2')
PHASE4_DIR   = DIFFPIPE_DIR / 'phase4'
V77_DIR      = PHASE4_DIR / 'v77'

V77_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV  = V77_DIR / 'exp7_sensitivity_table.csv'
OUT_JSON = V77_DIR / 'exp7_summary.json'

if OUT_CSV.exists():
    print(f'[SKIP] {OUT_CSV} already exists')
    sys.exit(0)

# ── ESF constants (from run_esf_psf.py) ───────────────────────────────────────
PROFILE_HALF  = 8       # pixels each side of edge pixel
COMP_LEN_MIN  = 20      # min connected component length
COMP_RMS_MAX  = 1.0     # max RMS deviation from straight line
FWHM_SIGMA    = 2.3548  # FWHM = 2.3548 * sigma

# ── Experiment parameter grid ──────────────────────────────────────────────────
R2_THRESHOLDS = [0.90, 0.85, 0.80]
MIN_FITS_LIST = [20, 10, 5]
# 기준 조건
R2_BASELINE   = 0.95
MIN_FITS_BASE = 20

# ── ESF model ─────────────────────────────────────────────────────────────────
def esf_model(x, A, x0, sigma, B):
    return A * erf((x - x0) / (np.sqrt(2.0) * sigma)) + B


def fit_esf_profile(profile: np.ndarray, r2_thresh: float) -> tuple[float | None, float]:
    """Fit ESF erf model to 1-D intensity profile.
    Returns (FWHM, R²) if fit passes r2_thresh, else (None, r2).
    """
    n = len(profile)
    xs = np.arange(n, dtype=np.float64) - n // 2

    A0   = (profile[-1] - profile[0]) / 2.0
    x0_0 = 0.0
    lo, hi = profile.min(), profile.max()
    if abs(hi - lo) < 1e-6:
        return None, 0.0
    idx_lo = np.argmin(np.abs(profile - (lo + 0.1 * (hi - lo))))
    idx_hi = np.argmin(np.abs(profile - (lo + 0.9 * (hi - lo))))
    sigma0 = max(0.5, abs(int(idx_hi) - int(idx_lo)) / 2.5)
    B0 = (profile[0] + profile[-1]) / 2.0

    try:
        popt, _ = curve_fit(
            esf_model, xs, profile.astype(np.float64),
            p0=[A0, x0_0, sigma0, B0],
            bounds=([-2, -n/2, 0.1, -1], [2, n/2, n/2, 2]),
            maxfev=2000,
        )
        fitted = esf_model(xs, *popt)
        ss_res = np.sum((profile - fitted) ** 2)
        ss_tot = np.sum((profile - profile.mean()) ** 2)
        r2     = 1.0 - ss_res / (ss_tot + 1e-12)
        sigma  = abs(popt[2])
        if sigma < 0.1 or sigma > PROFILE_HALF:
            return None, 0.0
        fwhm = float(sigma * FWHM_SIGMA)
        if r2 >= r2_thresh:
            return fwhm, float(r2)
        return None, float(r2)
    except (RuntimeError, ValueError):
        return None, 0.0


def _collect_edge_fwhms(img: np.ndarray, Gx, Gy, labeled: np.ndarray,
                         n_comp: int, r2_thresh: float) -> list[float]:
    """Extract valid FWHM estimates from all accepted edge components."""
    H, W = img.shape
    valid_fwhms = []
    for comp_id in range(1, n_comp + 1):
        ys, xs_px = np.where(labeled == comp_id)
        if len(ys) < COMP_LEN_MIN:
            continue
        if len(ys) >= 3:
            poly  = np.polyfit(xs_px.astype(float), ys.astype(float), 1)
            y_pred = np.polyval(poly, xs_px.astype(float))
            rms   = float(np.sqrt(np.mean((ys - y_pred) ** 2)))
        else:
            rms = 0.0
        if rms > COMP_RMS_MAX:
            continue
        for r, c in zip(ys, xs_px):
            gx_pix = Gx[r, c]
            gy_pix = Gy[r, c]
            gnorm  = np.hypot(gx_pix, gy_pix) + 1e-12
            nx = gx_pix / gnorm
            ny = gy_pix / gnorm
            d_coords   = np.arange(-PROFILE_HALF, PROFILE_HALF + 1, dtype=np.float64)
            row_coords = r + d_coords * ny
            col_coords = c + d_coords * nx
            if (row_coords.min() < 0 or row_coords.max() >= H or
                    col_coords.min() < 0 or col_coords.max() >= W):
                continue
            profile = map_coordinates(
                img.astype(np.float64), [row_coords, col_coords], order=1, mode='nearest'
            )
            if len(profile) < PROFILE_HALF:
                continue
            fwhm, _ = fit_esf_profile(np.array(profile, dtype=np.float64), r2_thresh)
            if fwhm is not None:
                valid_fwhms.append(fwhm)
    return valid_fwhms


def measure_fwhm_image(img: np.ndarray, r2_thresh: float, min_valid_edges: int
                        ) -> tuple[float | None, int]:
    """
    img: float32 [0,1] grayscale (H, W)
    Returns (median_fwhm or None, n_valid).
    """
    # Sobel gradients
    img_u8 = (img * 255).clip(0, 255).astype(np.uint8)
    Gx = cv2.Sobel(img_u8, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(img_u8, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(Gx, Gy)

    # Top-5% threshold
    thresh = np.percentile(mag, 95.0)
    edge_bin = (mag >= thresh).astype(np.uint8)

    # Connected components
    struct = np.ones((3, 3), dtype=np.int32)
    labeled, n_comp = nd_label(edge_bin, structure=struct)

    valid_fwhms = _collect_edge_fwhms(img, Gx, Gy, labeled, n_comp, r2_thresh)
    n_valid = len(valid_fwhms)

    if n_valid < min_valid_edges or n_valid == 0:
        return None, n_valid

    return float(np.median(valid_fwhms)), n_valid


# ── Load dirty2 images ─────────────────────────────────────────────────────────
dirty2_paths = sorted(REAL_DIR.glob('*.png'))[:5]
print(f'[exp7] dirty2 images: {[p.name for p in dirty2_paths]}')

dirty2_imgs = []
for p in dirty2_paths:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    dirty2_imgs.append((img, p.name))
    print(f'  {p.name}: shape={img.shape}')

# ── Main sensitivity loop ──────────────────────────────────────────────────────
rows = []
grid = list(product(R2_THRESHOLDS, MIN_FITS_LIST))
total_combos = len(grid)

for combo_idx, (r2_thresh, min_fits) in enumerate(grid):
    print(f'\n[exp7] [{combo_idx+1}/{total_combos}] r2_thresh={r2_thresh}  min_fits={min_fits}')
    for img, img_name in dirty2_imgs:
        fwhm_med, n_valid = measure_fwhm_image(img, r2_thresh=r2_thresh, min_valid_edges=min_fits)
        fwhm_val = fwhm_med if fwhm_med is not None else float('nan')
        print(f'  {img_name}: fwhm_median={fwhm_val:.3f}  n_valid={n_valid}')
        rows.append({
            'r2_thresh': r2_thresh,
            'min_fits':  min_fits,
            'img_name':  img_name,
            'fwhm_median': fwhm_val,
            'n_valid': n_valid,
        })

# ── Also compute baseline (R²=0.95, min_fits=20) ──────────────────────────────
print(f'\n[exp7] Baseline: r2_thresh={R2_BASELINE}  min_fits={MIN_FITS_BASE}')
baseline_rows = []
for img, img_name in dirty2_imgs:
    fwhm_med, n_valid = measure_fwhm_image(img, r2_thresh=R2_BASELINE, min_valid_edges=MIN_FITS_BASE)
    fwhm_val = fwhm_med if fwhm_med is not None else float('nan')
    print(f'  {img_name}: fwhm_median={fwhm_val:.3f}  n_valid={n_valid}')
    baseline_rows.append({
        'r2_thresh': R2_BASELINE,
        'min_fits':  MIN_FITS_BASE,
        'img_name':  img_name,
        'fwhm_median': fwhm_val,
        'n_valid': n_valid,
    })

# ── Save CSV (grid + baseline) ─────────────────────────────────────────────────
all_rows = rows + baseline_rows
fieldnames = ['r2_thresh', 'min_fits', 'img_name', 'fwhm_median', 'n_valid']
with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)
print(f'\n[exp7] Saved CSV: {OUT_CSV}')

# ── Summary aggregation ────────────────────────────────────────────────────────
# strict-R²-only 중앙값: 각 조건에서 n_valid >= min_fits인 이미지들의 fwhm_median 중앙값
ESF_LO = 1.9
ESF_HI = 2.3

FWHM_ACCEPT_RANGE = (ESF_LO, ESF_HI)

def agg_fwhm_for_condition(condition_rows):
    """strict-only: n_valid >= min_fits인 행들의 fwhm_median 집계 중앙값."""
    valid_fwhms = [r['fwhm_median'] for r in condition_rows
                   if not np.isnan(r['fwhm_median'])]
    if not valid_fwhms:
        return float('nan')
    return float(np.median(valid_fwhms))


summary = {'conditions': [], 'baseline': {}, 'verdict': '', 'deviant_conditions': []}

# Baseline
baseline_agg = agg_fwhm_for_condition(baseline_rows)
baseline_valid_count = sum(1 for r in baseline_rows if not np.isnan(r['fwhm_median']))
summary['baseline'] = {
    'r2_thresh': R2_BASELINE,
    'min_fits':  MIN_FITS_BASE,
    'aggregate_fwhm_median': baseline_agg,
    'n_valid_images': baseline_valid_count,
    'in_range': bool(ESF_LO <= baseline_agg <= ESF_HI) if not np.isnan(baseline_agg) else False,
}

deviant = []
for r2_thresh, min_fits in grid:
    cond_rows = [r for r in rows if r['r2_thresh'] == r2_thresh and r['min_fits'] == min_fits]
    agg = agg_fwhm_for_condition(cond_rows)
    n_valid_imgs = sum(1 for r in cond_rows if not np.isnan(r['fwhm_median']))
    in_range = bool(ESF_LO <= agg <= ESF_HI) if not np.isnan(agg) else False
    entry = {
        'r2_thresh': r2_thresh,
        'min_fits':  min_fits,
        'aggregate_fwhm_median': agg,
        'n_valid_images': n_valid_imgs,
        'in_range': in_range,
    }
    summary['conditions'].append(entry)
    if not in_range:
        deviant.append({'r2_thresh': r2_thresh, 'min_fits': min_fits, 'agg_fwhm': agg})

summary['deviant_conditions'] = deviant

if len(deviant) == 0:
    verdict = 'ROBUST'
else:
    verdict = 'SENSITIVE'
summary['verdict'] = verdict
summary['esf_accept_range'] = [ESF_LO, ESF_HI]

with open(OUT_JSON, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'[exp7] Saved JSON: {OUT_JSON}')

# ── Console verdict summary ────────────────────────────────────────────────────
print('\n' + '=' * 65)
print('EXPERIMENT 7 ESF SENSITIVITY VERDICT')
print('=' * 65)
print(f'  Accept range: f_ESF ∈ [{ESF_LO}, {ESF_HI}]')
print(f'\n  Baseline (R²={R2_BASELINE}, min_fits={MIN_FITS_BASE}):')
b = summary['baseline']
print(f"    agg_fwhm={b['aggregate_fwhm_median']:.3f}  in_range={b['in_range']}")
print(f'\n  Grid conditions:')
for entry in summary['conditions']:
    tag = 'OK  ' if entry['in_range'] else 'DEVIANT'
    print(f"    R²≥{entry['r2_thresh']}  min_fits={entry['min_fits']:2d}  "
          f"agg_fwhm={entry['aggregate_fwhm_median']:.3f}  [{tag}]")
if deviant:
    print(f'\n  Deviant conditions ({len(deviant)}):')
    for d in deviant:
        print(f"    R²≥{d['r2_thresh']}  min_fits={d['min_fits']}  agg_fwhm={d['agg_fwhm']:.3f}")
print(f'\n  → VERDICT: {verdict}')
print('=' * 65)
print('[exp7] Done.')
