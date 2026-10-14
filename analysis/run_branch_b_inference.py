#!/usr/bin/env python3
"""
HANDOFF v7.19 분기 B — checkpoint 재추론, dataset-level per-image counts 생성
=============================================================================
출력: f1_recompute_results/counts/per_image_counts.csv
      f1_recompute_results/MANIFEST.md

전처리: 모든 조건 동일 — p42r_testsets의 .npy 직접 로드, 추가 prenorm 없음.
        calib+infer_pre만 PRENORM_A * img + PRENORM_B 적용.
재현 대조: 각 조건 첫 run에서 batch 평균 F1도 계산 → results.json 비교.
           소수 3자리 오차 > 0.001이면 즉시 중단.

실행:
    cd /mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation
    python /mnt/gstore/home/xiangyue/HBM_paper2/run_branch_b_inference.py
"""
import csv, json, sys, os, time
from pathlib import Path

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

# ── paths ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, '/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation')
from dataset import json_to_mask
from model import UNet
import config as cfg_mod

PHASE4      = Path('/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4')
TESTSETS_DIR = PHASE4 / 'p42r_testsets'
OUT_DIR     = Path('/mnt/gstore/home/xiangyue/HBM_paper2/f1_recompute_results/counts')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV     = OUT_DIR / 'per_image_counts.csv'
MANIFEST    = OUT_DIR.parent / 'MANIFEST.md'

PRENORM_A = 0.501834
PRENORM_B = 0.173320

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
N_WORKERS  = 2
N_CLASSES  = 4

print(f'device={DEVICE}')

# ── 조건 정의 ─────────────────────────────────────────────────────────────────
# (config_name, run_root, testsets, apply_prenorm, apply_zscore, skip_repro)
# skip_repro=True: results.json이 다른 조건의 것이라 비교 불가 (calib+infer_pre는
#   D_calib checkpoint를 prenorm 추가하여 재추론 — 원 results.json은 prenorm 없음)
CONFIGS = [
    ('D_calib',         PHASE4/'p42r_v2_runs'/'calib',    ['test_id','test_ood','test_cal'], False, False, False),
    ('D_manual',        PHASE4/'p42r_v2_runs'/'manual',   ['test_id','test_ood','test_cal'], False, False, False),
    ('AB-1',            PHASE4/'p43_runs'/'ab1',           ['test_id','test_ood','test_cal'], False, False, False),
    ('AB-2',            PHASE4/'p43_runs'/'ab2',           ['test_id','test_ood','test_cal'], False, False, False),
    ('AB-3',            PHASE4/'p43_runs'/'ab3',           ['test_id','test_ood','test_cal'], False, False, False),
    ('D_v5e',           PHASE4/'v77'/'exp3b_runs',         ['test_id','test_ood','test_cal'], False, False, False),
    ('D_sigma_p',       PHASE4/'v77'/'exp3_runs',          ['test_id','test_ood','test_cal'], False, False, False),
    ('z-score',         PHASE4/'v77'/'exp4_runs',          ['test_id','test_ood','test_cal'], False, True,  False),
    ('calib+infer_pre', PHASE4/'p42r_v2_runs'/'calib',    ['test_id','test_ood'],            True,  False, True),
]

# ── 재현 대조: 조건별 첫 run의 results.json 기준값 ────────────────────────────
def get_ref_f1by_class(run_root: Path, ts: str):
    """results.json에서 f1_by_class 읽기 (첫 run 기준)"""
    rj = sorted(run_root.glob('run*'))[0] / 'results.json'
    if not rj.exists():
        return None
    d = json.load(open(rj))
    return d.get(ts, {}).get('f1_by_class')


# ── Dataset ───────────────────────────────────────────────────────────────────
class FixedTestsetWithIdx(Dataset):
    """이미지 인덱스(image_id)도 함께 반환"""
    def __init__(self, testset_name: str, apply_prenorm: bool = False, apply_zscore: bool = False):
        ts_dir = TESTSETS_DIR / testset_name
        meta   = json.load(open(ts_dir / 'meta.json'))
        lbl_dir = Path(meta['label_dir'])
        lbl_files = sorted(lbl_dir.glob('*.json'))
        self.items = []
        for lf in lbl_files:
            num = lf.stem.replace('label_', '')
            np_path = ts_dir / f'image_{num}.npy'
            if np_path.exists():
                self.items.append((num, np_path, lf))
        self.apply_prenorm = apply_prenorm
        self.apply_zscore  = apply_zscore

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_id, np_path, lf = self.items[idx]
        img_deg = np.load(str(np_path))
        if self.apply_zscore:
            img_deg = (img_deg - img_deg.mean()) / (img_deg.std() + 1e-8)
        img_3ch = np.stack([img_deg] * 3, axis=0).astype(np.float32)
        if self.apply_prenorm:
            img_3ch = PRENORM_A * img_3ch + PRENORM_B
        H_orig = img_deg.shape[0] - 6
        mask   = json_to_mask(str(lf), H_orig, 520)
        mask   = cv2.copyMakeBorder(mask, 3, 3, 0, 0, cv2.BORDER_CONSTANT, value=0)[:, 4:516]
        return (torch.from_numpy(img_3ch),
                torch.from_numpy(mask.astype(np.int64)),
                img_id)


# ── 미리 testset 로드 ─────────────────────────────────────────────────────────
print('Preloading testsets...')
# (ts_name, apply_prenorm) 조합별로 Dataset 하나씩
_all_ts_keys = set()
for _, _, testsets, prenorm, zscore, _skip in CONFIGS:
    for ts in testsets:
        _all_ts_keys.add((ts, prenorm, zscore))

testset_cache = {}
for ts_name, prenorm, zscore in sorted(_all_ts_keys):
    ds = FixedTestsetWithIdx(ts_name, prenorm, zscore)
    testset_cache[(ts_name, prenorm, zscore)] = ds
    print(f'  {ts_name} prenorm={prenorm} zscore={zscore}: {len(ds)} images')


# ── 메트릭 계산 ───────────────────────────────────────────────────────────────
def batch_f1_per_class(preds_logits, targets):
    """batch 평균 F1 (재현 검증용) — 원 캠페인과 동일 구현"""
    pf = preds_logits.argmax(dim=1).flatten()
    tf = targets.flatten()
    f1s = []
    for c in range(N_CLASSES):
        tp = ((pf==c)&(tf==c)).sum().float()
        fp = ((pf==c)&(tf!=c)).sum().float()
        fn = ((pf!=c)&(tf==c)).sum().float()
        prec = tp/(tp+fp+1e-8); rec = tp/(tp+fn+1e-8)
        f1s.append((2*prec*rec/(prec+rec+1e-8)).item())
    return f1s


def per_image_counts(preds_logits, targets, img_ids):
    """이미지별 클래스별 TP/FP/FN 반환: list of (img_id, c, tp, fp, fn)"""
    pred_b = preds_logits.argmax(dim=1).cpu().numpy()  # (B,H,W)
    tgt_b  = targets.cpu().numpy()
    rows = []
    for i, img_id in enumerate(img_ids):
        p = pred_b[i]; g = tgt_b[i]
        for c in range(N_CLASSES):
            tp = int(((p==c)&(g==c)).sum())
            fp = int(((p==c)&(g!=c)).sum())
            fn = int(((p!=c)&(g==c)).sum())
            rows.append((img_id, c, tp, fp, fn))
    return rows


# ── 이미 완료된 (config, ts, run_id) 스킵 ───────────────────────────────────
done_keys = set()
if OUT_CSV.exists():
    with open(OUT_CSV, newline='') as f:
        for row in csv.DictReader(f):
            done_keys.add((row['config'], row['test_set'], row['run_id']))
    print(f'기존 완료 (config,ts,run): {len(done_keys)}개 — 스킵')

# ── 출력 CSV 열기 (append 모드) ───────────────────────────────────────────────
csv_file = open(OUT_CSV, 'a', newline='')
csv_writer = csv.writer(csv_file)
if OUT_CSV.stat().st_size == 0:
    csv_writer.writerow(['config','test_set','run_id','image_id','class_id','tp','fp','fn'])

# ── MANIFEST 초기화 ──────────────────────────────────────────────────────────
manifest_lines = [
    '# MANIFEST — Branch B Inference\n',
    f'생성: 2026-08-20\n\n',
    '| config | run_id | checkpoint | test_set | n_images | rows |\n',
    '|--------|--------|------------|----------|----------|------|\n',
]

# ── 재현 검증 결과 ─────────────────────────────────────────────────────────────
repro_results = []
REPRO_TOL = 0.001

# ── 메인 루프 ─────────────────────────────────────────────────────────────────
total_rows = 0
t0 = time.time()

model = UNet(cfg_mod.IN_CHANNELS, cfg_mod.NUM_CLASSES).to(DEVICE)
model.eval()

for cfg_name, run_root, testsets, apply_prenorm, apply_zscore, skip_repro in CONFIGS:
    run_dirs = sorted(run_root.glob('run*'))
    if not run_dirs:
        print(f'  WARN: {run_root} — run 없음'); continue

    # 재현 대조용 기준값 (첫 run)
    is_first_run = True

    for run_dir in run_dirs:
        pth = run_dir / 'final.pth'
        if not pth.exists():
            print(f'  WARN: {pth} 없음'); continue

        run_id = f'{run_root.name}_{run_dir.name}'

        # 이 run의 모든 testset이 완료됐는지 확인
        all_done = all((cfg_name, ts.upper().replace('TEST_','TEST_'), run_id) in done_keys
                       or (cfg_name, ts, run_id) in done_keys
                       for ts in testsets)
        if all_done:
            print(f'  SKIP {cfg_name}/{run_id} (already done)')
            is_first_run = False
            continue

        print(f'\n[{cfg_name}] {run_id}  prenorm={apply_prenorm}  zscore={apply_zscore}')
        state = torch.load(str(pth), map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval()

        for ts_name in testsets:
            ts_up = ts_name.upper()
            key = (cfg_name, ts_up, run_id)
            if key in done_keys:
                print(f'  SKIP {ts_name}'); continue

            ds = testset_cache[(ts_name, apply_prenorm, apply_zscore)]
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)

            # per-image counts 수집 + 재현 검증용 batch F1
            run_counts = []
            batch_f1_list = []  # for repro check

            with torch.no_grad():
                for imgs, msks, img_ids in loader:
                    imgs, msks = imgs.to(DEVICE), msks.to(DEVICE)
                    outs = model(imgs)

                    # per-image counts
                    run_counts.extend(per_image_counts(outs, msks, list(img_ids)))

                    # 재현 검증 (첫 run만)
                    if is_first_run:
                        batch_f1_list.append(batch_f1_per_class(outs, msks))

            # CSV 기록
            n_rows = 0
            for img_id, c, tp, fp, fn in run_counts:
                csv_writer.writerow([cfg_name, ts_up, run_id, img_id, c, tp, fp, fn])
                n_rows += 1
            csv_file.flush()
            total_rows += n_rows
            done_keys.add(key)
            manifest_lines.append(
                f'| {cfg_name} | {run_id} | {pth} | {ts_up} | {len(ds)} | {n_rows} |\n'
            )
            print(f'  {ts_name}: {n_rows} rows written')

            # ── 재현 검증 ────────────────────────────────────────────────────
            if is_first_run and batch_f1_list and not skip_repro:
                batch_avg = [float(np.mean([b[c] for b in batch_f1_list])) for c in range(N_CLASSES)]
                ref = get_ref_f1by_class(run_root, ts_name)
                if ref is None:
                    print(f'  REPRO: results.json 없음 — 건너뜀')
                else:
                    deltas = [abs(batch_avg[c] - ref[c]) for c in range(N_CLASSES)]
                    max_d = max(deltas)
                    status = 'PASS' if max_d <= REPRO_TOL else 'FAIL'
                    print(f'  REPRO [{status}] {cfg_name}/{ts_name}: '
                          f'batch_avg={[f"{v:.4f}" for v in batch_avg]}  '
                          f'ref={[f"{v:.4f}" for v in ref]}  max_Δ={max_d:.4f}')
                    repro_results.append((cfg_name, ts_name, run_id, status, max_d, batch_avg, ref))
                    if status == 'FAIL':
                        print('  !!! REPRO FAIL — 즉시 중단 !!!')
                        csv_file.close()
                        sys.exit(1)

        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        is_first_run = False

csv_file.close()

elapsed = time.time() - t0
print(f'\n완료: 총 {total_rows} rows, {elapsed/60:.1f}분')

# ── MANIFEST 저장 ─────────────────────────────────────────────────────────────
manifest_lines.append(f'\n총 {total_rows} rows, 소요 {elapsed/60:.1f}분\n')
manifest_lines.append('\n## 재현 검증 결과\n\n')
manifest_lines.append('| config | test_set | run_id | 결과 | max_Δ |\n')
manifest_lines.append('|--------|----------|--------|------|-------|\n')
for cfg_n, ts_n, rid, status, max_d, ba, ref in repro_results:
    manifest_lines.append(f'| {cfg_n} | {ts_n} | {rid} | {status} | {max_d:.5f} |\n')

with open(MANIFEST, 'w') as f:
    f.writelines(manifest_lines)

print(f'MANIFEST: {MANIFEST}')
print(f'counts CSV: {OUT_CSV}')
print(f'\nwc -l {OUT_CSV}:')
with open(OUT_CSV) as f:
    print(f'  {sum(1 for _ in f)} lines (including header)')
