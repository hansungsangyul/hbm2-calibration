#!/bin/bash
# P4.2r launch script — HANDOFF v6.9 §3 조건 (b)
# 40 runs: calib 0-19 → manual 0-19 (직렬)
# commit hash: (커밋 후 기록)

PYTHON=/home/xiangyue/.conda/envs/unet/bin/python
SCRIPT=/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/run_p42r.py
WORK_DIR=/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$WORK_DIR"

echo "[P4.2r] Starting at $(date)"

for IDX in $(seq 0 19); do
    echo "[P4.2r] calib run $IDX"
    $PYTHON $SCRIPT --condition calib  --run_idx $IDX --epochs 41 --batch_size 32
done

for IDX in $(seq 0 19); do
    echo "[P4.2r] manual run $IDX"
    $PYTHON $SCRIPT --condition manual --run_idx $IDX --epochs 41 --batch_size 32
done

echo "[P4.2r] All 40 runs done at $(date)"
