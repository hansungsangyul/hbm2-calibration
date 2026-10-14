#!/bin/bash
# sequential on GPU0; each run ~23 min
PY=/mnt/gstore/home/xiangyue/HBM_paper/envs/hbm_cudnn/bin/python
D=/mnt/gstore/home/xiangyue/HBM_paper2/revision1/a1_weights
cd /mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation
run(){ s=$1; seed=$2; [ -f $D/runs/${s}_s${seed}/result.json ] && { echo "skip $s $seed"; return; }
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $D/run_a1_weights.py --scheme $s --seed $seed --out $D/runs/${s}_s${seed} 2>&1 | tee -a $D/launch_a1.log; }
run W0 970000
for s in W1 W2 W3 W4; do for seed in 970000 971000 972000; do run $s $seed; done; done
echo "A1 ALL DONE $(date -Is)" | tee -a $D/launch_a1.log
