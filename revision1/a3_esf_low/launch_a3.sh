#!/bin/bash
PY=/home/xiangyue/.conda/envs/unet/bin/python; S=/mnt/gstore/home/xiangyue/HBM_paper2/revision1/a3_esf_low/run_a3_ablation.py; L=/mnt/gstore/home/xiangyue/HBM_paper2/revision1/a3_esf_low/logs; mkdir -p $L
while ! grep -q "A2 ALL DONE" /mnt/gstore/home/xiangyue/HBM_paper2/revision1/a2_ident/launch_a2.log; do sleep 120; done
cd /mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation
for i in $(seq 0 19); do
  [ -f /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4/p43_runs/ab4/run$i/results.json ] && continue
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $S --condition ab4 --run_idx $i > $L/ab4_run$i.log 2>&1
  tail -1 $L/ab4_run$i.log >> $L/summary.log
done
echo "A3 ALL DONE $(date -Is)" >> $L/summary.log
