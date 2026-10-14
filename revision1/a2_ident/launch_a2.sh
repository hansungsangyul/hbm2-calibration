#!/bin/bash
PY=/mnt/gstore/home/xiangyue/HBM_paper/envs/hbm_cudnn/bin/python; D=/mnt/gstore/home/xiangyue/HBM_paper2/revision1/a2_ident
cd /mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation
while ! grep -q "A1 ALL DONE" /mnt/gstore/home/xiangyue/HBM_paper2/revision1/a1_weights/launch_a1.log; do sleep 60; done
PYTHONUNBUFFERED=1 $PY $D/run_a2a_hessian.py 2>&1 | tee -a $D/launch_a2.log
for N in 50 100 300 600 1000; do for seed in 970000 971000 972000; do
  [ -f $D/profile/N${N}_s${seed}/result.json ] && continue
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $D/run_a2b_profile.py --N $N --seed $seed --out $D/profile/N${N}_s${seed} 2>&1 | tee -a $D/launch_a2.log
done; done
echo "A2 ALL DONE $(date -Is)" | tee -a $D/launch_a2.log
