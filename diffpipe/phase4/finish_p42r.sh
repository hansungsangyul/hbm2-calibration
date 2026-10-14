#!/bin/bash
# run0, run1 재실행 + 완료 후 텔레그램 알림

PYTHON=/home/xiangyue/.conda/envs/unet/bin/python
SCRIPT=/mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/run_p42r.py
WORK_DIR=/mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation
NOTIFY=/mnt/gstore/home/xiangyue/notify_telegram.sh

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$WORK_DIR"

# launch_p42r.sh 완료 대기
echo "[finish] launch_p42r.sh 완료 대기..."
while pgrep -f "launch_p42r.sh" > /dev/null 2>&1; do
    sleep 60
done

echo "[finish] launch 완료. calib/run0, run1 재실행..."

$PYTHON $SCRIPT --condition calib --run_idx 0 --epochs 41 --batch_size 32
$PYTHON $SCRIPT --condition calib --run_idx 1 --epochs 41 --batch_size 32

echo "[finish] 전체 완료. 텔레그램 알림..."
bash "$NOTIFY" "P4.2r 완료: calib×20 + manual×20 = 40 runs 전부 완료. 분석 착수 가능."
