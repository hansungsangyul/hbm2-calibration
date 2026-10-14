#!/bin/bash
# P4.2 TOST launcher — D_calib 20 runs + D_manual 20 runs
# 순서: calib 0~19 → manual 0~19 (순차 실행, Phase 5와 병렬)
# HANDOFF v6.4 §2

set -u
PYTHON=/home/xiangyue/.conda/envs/unet/bin/python
SCRIPT=/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/run_p42_tost.py
LOG_DIR=/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/p42_runs
AUGMENT_DIR=/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation

mkdir -p "$LOG_DIR"

echo "[$(date)] P4.2 TOST launch start" | tee -a "$LOG_DIR/launch.log"

for CONDITION in calib manual; do
    for RUN_IDX in $(seq 0 19); do
        LOGFILE="$LOG_DIR/${CONDITION}_run${RUN_IDX}.log"
        echo "[$(date)] Starting $CONDITION run $RUN_IDX" | tee -a "$LOG_DIR/launch.log"
        cd "$AUGMENT_DIR" && \
        PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            "$PYTHON" "$SCRIPT" \
            --condition "$CONDITION" \
            --run_idx "$RUN_IDX" \
            --epochs 41 \
            --batch_size 4 \
            2>&1 | tee "$LOGFILE"
        STATUS=$?
        echo "[$(date)] $CONDITION run $RUN_IDX done (exit=$STATUS)" | tee -a "$LOG_DIR/launch.log"
    done
done

echo "[$(date)] All P4.2 runs done. Running TOST analysis ..." | tee -a "$LOG_DIR/launch.log"
cd "$AUGMENT_DIR" && \
    "$PYTHON" "$SCRIPT" --analyze_only 2>&1 | tee "$LOG_DIR/tost_analysis.log"
echo "[$(date)] P4.2 complete" | tee -a "$LOG_DIR/launch.log"
