#!/usr/bin/env python3
"""
P4.2r 테스트셋 3종 사전 생성 (HANDOFF v6.6 §2.1)
==================================================
온-더-플라이 열화 금지 → ε 시드 고정으로 디스크 저장.

TEST_ID : θ_manual, pre-norm 미적용, 500장  (D_manual 주 평가셋)
TEST_OOD: OOD(×1.5), pre-norm 미적용, 500장 (OOD 평가셋)
TEST_CAL: D_calib 파이프라인, pre-norm 적용, 500장 (D_calib 주 평가셋)

소스: test_data/HBM_clean (500장, indices 1501-2000)
라벨: test_data/HBM_labels (기존 JSON 파일 그대로 참조)

산출물:
  phase4/p42r_testsets/{test_id,test_ood,test_cal}/
    image_{N}.npy   : float32 [H_crop, W_crop] (border+crop 적용 후)
    meta.json       : 생성 파라미터·시드·경로

주의: HANDOFF 명세 1000장 vs 실제 test_data 500장 불일치.
     test_data 전체(500장)를 사용, p42r_testsets.md에 기재.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
import cv2
import torch

DIFFPIPE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe')
AUGMENT_DIR  = Path('/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation')
OUT_BASE     = DIFFPIPE_DIR / 'phase4' / 'p42r_testsets'

sys.path.insert(0, str(DIFFPIPE_DIR / 'src'))
sys.path.insert(0, str(AUGMENT_DIR))

from pipeline import DegradationPipeline, softplus_inverse, sigmoid_box_inverse, _N_LO, _N_HI

# ── Pre-norm constants (from d_calib_config.json) ──────────────────────────
PRENORM_A = 0.501834
PRENORM_B = 0.173320

# ── θ_manual ───────────────────────────────────────────────────────────────
THETA_MANUAL = dict(
    N=300.0, sigma_g=0.0408, p=0.001,
    fwhm=2.5, alpha=0.35,
    in_low=0.05, in_high=0.90,
    out_low=0.10, out_high=0.90,
    sigma_post=0.0, ring_strength=0.12, ring_n=8.0,
)

# ── TEST_OOD: 전작 R1-3 (×1.5 강도) ────────────────────────────────────────
THETA_OOD = {**THETA_MANUAL,
    'fwhm': 3.75,             # 2.5 × 1.5
    'sigma_g': 0.0612,        # 0.0408 × 1.5
    'ring_strength': 0.18,    # 0.12 × 1.5 (0.12→0.18)
}

# ── D_calib ─────────────────────────────────────────────────────────────────
THETA_CALIB = {**THETA_MANUAL,
    'fwhm': 2.0956,
    'sigma_post': 0.0,
}

TESTSET_SPECS = {
    'test_id': {
        'theta':   THETA_MANUAL,
        'prenorm': False,
        'desc':    'θ_manual, pre-norm 미적용 (D_manual 주 평가셋, 스펙 v1.0 §4.2)',
        'eps_seed_base': 500000,
    },
    'test_ood': {
        'theta':   THETA_OOD,
        'prenorm': False,
        'desc':    '전작 R1-3 OOD ×1.5 (PSF 2.5→3.75, σ_g×1.5, ring 0.12→0.18), pre-norm 미적용',
        'eps_seed_base': 510000,
    },
    'test_cal': {
        'theta':   THETA_CALIB,
        'prenorm': True,
        'desc':    'D_calib 파이프라인 (pre-norm 적용, fwhm=2.0956) (D_calib 주 평가셋)',
        'eps_seed_base': 520000,
    },
}


def make_pipe(theta: dict) -> DegradationPipeline:
    pipe = DegradationPipeline()
    with torch.no_grad():
        N_c = float(np.clip(theta['N'], _N_LO + 1e-3, _N_HI - 1e-3))
        pipe.N_raw.fill_(sigmoid_box_inverse(N_c))
        pipe.sigma_g_raw.fill_(softplus_inverse(max(theta['sigma_g'], 1e-6)))
        pipe.fwhm_raw.fill_(softplus_inverse(theta['fwhm']))
        pipe.alpha_raw.fill_(softplus_inverse(theta['alpha']))
        pipe.ring_strength_raw.fill_(softplus_inverse(theta['ring_strength']))
        pipe.ring_n_raw.fill_(softplus_inverse(theta['ring_n']))
        pipe.in_low.fill_(theta['in_low'])
        pipe.in_high.fill_(theta['in_high'])
        pipe.out_low.fill_(theta['out_low'])
        pipe.out_high.fill_(theta['out_high'])
        pipe.p.fill_(theta['p'])
        pipe.sigma_post.fill_(theta['sigma_post'])
    for param in pipe.parameters():
        param.requires_grad_(False)
    return pipe


def generate_testset(name: str, spec: dict) -> dict:
    theta      = spec['theta']
    use_prenorm = spec['prenorm']
    seed_base  = spec['eps_seed_base']

    out_dir = OUT_BASE / name
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir = AUGMENT_DIR / 'test_data' / 'HBM_clean'
    lbl_dir = AUGMENT_DIR / 'test_data' / 'HBM_labels'

    img_files = sorted(img_dir.glob('*.png'))
    assert len(img_files) == 500, f"Expected 500 test images, got {len(img_files)}"

    pipe = make_pipe(theta)

    generated = []
    t0 = time.time()
    for i, img_path in enumerate(img_files):
        num = img_path.stem.replace('image_', '')
        out_npy = out_dir / f'image_{num}.npy'
        if out_npy.exists():
            generated.append(num)
            continue

        img_raw = cv2.imread(str(img_path))
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        H, W = img_raw.shape

        if use_prenorm:
            img_in = PRENORM_A * img_raw + PRENORM_B
        else:
            img_in = img_raw

        # Fixed ε seed per image
        torch.manual_seed(seed_base + i)
        eps = {
            'eps1':     torch.randn(H, W),
            'eps2':     torch.randn(H, W),
            'eps3':     torch.rand(H, W),
            'eps_post': torch.zeros(H, W),   # sigma_post=0 → no-op
        }

        x = torch.from_numpy(img_in)
        with torch.no_grad():
            y = pipe(x, eps)

        img_deg = y.numpy()
        # Border/crop (same as training data preprocessing)
        img_deg = cv2.copyMakeBorder(img_deg, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)
        img_deg = img_deg[:, 4:516]

        np.save(str(out_npy), img_deg.astype(np.float32))
        generated.append(num)

        if (i + 1) % 100 == 0:
            print(f"  [{name}] {i+1}/500  ({time.time()-t0:.0f}s)")

    # Save metadata
    meta = {
        'name':         name,
        'desc':         spec['desc'],
        'theta':        theta,
        'prenorm':      use_prenorm,
        'prenorm_a':    PRENORM_A if use_prenorm else None,
        'prenorm_b':    PRENORM_B if use_prenorm else None,
        'eps_seed_base': seed_base,
        'n_images':     len(generated),
        'source_dir':   str(img_dir),
        'label_dir':    str(lbl_dir),
        'out_dir':      str(out_dir),
        'handoff_ref':  'HANDOFF_v6.6 §2.1',
        'note_1000':    'HANDOFF 명세 1000장 vs 실제 test_data 500장. test_data 전체(500) 사용.',
    }
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"  [{name}] 완료: {len(generated)}장, {elapsed:.1f}s")
    return meta


if __name__ == '__main__':
    print("P4.2r 테스트셋 3종 생성 (HANDOFF v6.6 §2.1)")
    print(f"출력 위치: {OUT_BASE}")
    print()

    metas = {}
    for name, spec in TESTSET_SPECS.items():
        print(f"[{name}] 생성 중: {spec['desc']}")
        metas[name] = generate_testset(name, spec)
        print()

    # 통합 매니페스트
    manifest = {
        'handoff_ref': 'HANDOFF_v6.6 §2.1',
        'testsets': metas,
        'note': '각 이미지는 test_data/HBM_labels의 동명 label JSON과 대응',
    }
    with open(OUT_BASE / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print("=== 생성 완료 ===")
    for name, m in metas.items():
        print(f"  {name}: {m['n_images']}장  prenorm={m['prenorm']}  fwhm={m['theta']['fwhm']}")
    print(f"manifest: {OUT_BASE}/manifest.json")
