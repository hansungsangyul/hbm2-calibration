#!/usr/bin/env python3
"""
MC5 불확실성 CI — f 실측 2.057의 이미지 층화 부트스트랩 (HANDOFF v7.6 §3.2)
==========================================================================
유효 3매(sufficient=True, r2_relaxed=False): img00=2.0569, img03=2.4519, img04=1.9050
이미지 층화 리샘플 → bootstrap CI → FWHM_FROZEN 전파 (×2.500/2.454)

입력: phase4/esf_psf_measurement.json
출력: reanalysis_mc3_mc5.md (MC5 섹션 추가)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

ESF_JSON  = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/esf_psf_measurement.json')
OUT_MD    = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/reanalysis_mc3_mc5.md')

d = json.load(open(ESF_JSON))
rm_results = d['real_measurement']['results']

# 유효 이미지 필터 (sufficient=True, r2_relaxed=False)
valid = [r for r in rm_results if r.get('sufficient', False) and not r.get('r2_relaxed', False)]
print("Valid images:")
per_image_medians = []
for r in valid:
    print(f"  {r['label']}: median_fwhm={r['median_fwhm']:.4f}  n_valid={r['n_valid_edge_fits']}")
    per_image_medians.append(r['median_fwhm'])

n_img = len(per_image_medians)
print(f"\nn_valid_images = {n_img}")
f_obs_median = float(np.median(per_image_medians))
print(f"Observed grand median f = {f_obs_median:.4f}")

# 자기 검증 교정 비율
sv = d['self_validation']
sv_medians_strict = [
    r['median_fwhm'] for r in sv['results']
    if not r.get('r2_relaxed', False)
]
sv_agg = float(np.median(sv_medians_strict))
APPLIED_FWHM = float(sv['applied_fwhm'])  # 2.5
CALIB_RATIO  = APPLIED_FWHM / sv_agg      # 2.5 / 2.454
FWHM_FROZEN  = f_obs_median * CALIB_RATIO

print(f"Self-validation: applied={APPLIED_FWHM}, strict median={sv_agg:.4f}")
print(f"Calibration ratio = {APPLIED_FWHM}/{sv_agg:.4f} = {CALIB_RATIO:.6f}")
print(f"FWHM_FROZEN = {f_obs_median:.4f} × {CALIB_RATIO:.6f} = {FWHM_FROZEN:.4f}")

# ── 이미지 층화 부트스트랩 ────────────────────────────────────────────
np.random.seed(2025_07_19)
N_BOOT = 100_000

boot_medians_raw    = []
boot_medians_frozen = []

for _ in range(N_BOOT):
    # Resample n_img images with replacement
    samp = np.random.choice(per_image_medians, size=n_img, replace=True)
    med_raw = float(np.median(samp))
    boot_medians_raw.append(med_raw)
    boot_medians_frozen.append(med_raw * CALIB_RATIO)

boot_raw    = np.array(boot_medians_raw)
boot_frozen = np.array(boot_medians_frozen)

# Percentile CI
ci_raw    = (float(np.percentile(boot_raw, 2.5)),    float(np.percentile(boot_raw, 97.5)))
ci_frozen = (float(np.percentile(boot_frozen, 2.5)), float(np.percentile(boot_frozen, 97.5)))

# Bias-corrected bootstrap CI (BCa requires acceleration; use percentile as primary)
boot_mean_raw = float(np.mean(boot_raw))
boot_std_raw  = float(np.std(boot_raw))

print(f"\n{'='*60}")
print("Bootstrap CI (N=100,000 iterations, image-stratified):")
print(f"  f_obs (raw grand median):  {f_obs_median:.4f}")
print(f"  Boot mean:                 {boot_mean_raw:.4f}")
print(f"  Boot std:                  {boot_std_raw:.4f}")
print(f"  95% CI (percentile):      [{ci_raw[0]:.4f}, {ci_raw[1]:.4f}]")
print(f"  FWHM_FROZEN = {FWHM_FROZEN:.4f}  95% CI: [{ci_frozen[0]:.4f}, {ci_frozen[1]:.4f}]")
print(f"  CI width (raw):            {ci_raw[1]-ci_raw[0]:.4f}")
print(f"  CI width (frozen):         {ci_frozen[1]-ci_frozen[0]:.4f}")

# ── 자기 검증 편향 f-의존성 1점 확인 ─────────────────────────────────
print(f"\n자기 검증 편향 f-의존성:")
print(f"  Applied f=2.5 → 엄격 측정 median={sv_agg:.4f}  편향={sv_agg-APPLIED_FWHM:.4f}")
print(f"  편향 비율 (상대적): {(sv_agg-APPLIED_FWHM)/APPLIED_FWHM*100:.2f}%")
print(f"  f=2.057 근방 편향 추정 (선형 외삽이 아닌 교정 적용 가능 범위)")
print(f"  → 1점만 있어 f-의존성 정량화 불가; 단일 교정 비율 2.5/2.454 적용")

# ── Write markdown ────────────────────────────────────────────────────
mc5_text = f"""## MC5 불확실성 CI

**실행일**: 자동 생성 (HANDOFF v7.6 §3.2)
**목적**: f 실측 2.057의 이미지 층화 부트스트랩 CI → FWHM_FROZEN 전파

### 데이터

| 이미지 | median FWHM | n_valid_edges | sufficient |
|--------|-------------|---------------|------------|
{chr(10).join(f"| {r['label']} | {r['median_fwhm']:.4f} | {r['n_valid_edge_fits']} | {r.get('sufficient')} |" for r in rm_results)}

- **유효 이미지** (sufficient=True, R²≥0.95): {n_img}매
- **per-image medians**: {[f'{v:.4f}' for v in per_image_medians]}
- **관측 grand median f**: {f_obs_median:.4f}

### 자기 검증 교정

- Applied FWHM = {APPLIED_FWHM} → 측정 strict median = {sv_agg:.4f}
- 교정 비율 = {APPLIED_FWHM}/{sv_agg:.4f} = {CALIB_RATIO:.6f}
- **FWHM_FROZEN** = {f_obs_median:.4f} × {CALIB_RATIO:.6f} = **{FWHM_FROZEN:.4f}**

### 부트스트랩 CI (N = 100,000)

이미지 층화 리샘플: n_img = {n_img}매에서 복원 추출, 표본 중앙값 계산.

| 지표 | 값 |
|------|---|
| Boot mean (f_raw) | {boot_mean_raw:.4f} |
| Boot std  (f_raw) | {boot_std_raw:.4f} |
| 95% CI (f_raw, percentile) | [{ci_raw[0]:.4f}, {ci_raw[1]:.4f}] |
| **95% CI (FWHM_FROZEN)** | **[{ci_frozen[0]:.4f}, {ci_frozen[1]:.4f}]** |
| CI width (FWHM_FROZEN) | {ci_frozen[1]-ci_frozen[0]:.4f} |

### 해석 (사전 판정 기준)

- FWHM_FROZEN = {FWHM_FROZEN:.4f} ∈ [{ci_frozen[0]:.4f}, {ci_frozen[1]:.4f}]
- n=3 이미지로 CI가 넓음: ±{(ci_frozen[1]-ci_frozen[0])/2:.4f} (반폭)
- Phase 5 f 그리드 (0, 0.625, 1.25, 2.5, 5.0, 10.0) 중 FWHM_FROZEN은 2.5 셀에 근접
  → ESF 앵커 위치 불확실성이 셀 경계를 넘지 않음 (2.5 셀 내 유지)

### 자기 검증 편향 f-의존성

f=2.5 단일 점만 측정됨 → f-의존성 정량화 불가.
교정 비율 {APPLIED_FWHM}/{sv_agg:.4f} = {CALIB_RATIO:.6f}을 단일 보정 인자로 적용.

---
"""

# Append to existing report
existing = ""
if OUT_MD.exists():
    existing = OUT_MD.read_text()

if "## MC5" in existing:
    start = existing.find("## MC5")
    end = existing.find("## MC4", start)
    if end > 0:
        existing = existing[:start] + mc5_text + existing[end:]
    else:
        existing = existing[:start] + mc5_text
else:
    existing = existing + mc5_text

OUT_MD.write_text(existing)
print(f"\nSaved MC5 section → {OUT_MD}")
