#!/bin/bash
PY=/mnt/gstore/home/xiangyue/HBM_paper/envs/hbm_cudnn/bin/python; D=/mnt/gstore/home/xiangyue/HBM_paper2/revision1
cd /mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation
for fi in 1.0 1.5 2.0 3.0 3.5 4.0 5.0; do
  [ -f $D/a2_ident/multistart/fi${fi}/result.json ] && continue
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $D/a1_weights/run_a1_weights.py --scheme W0 --seed 970000 --fwhm_init $fi --out $D/a2_ident/multistart/fi${fi} 2>&1 | tee -a $D/a2_ident/launch_a2c.log
done
echo "A2c ALL DONE $(date -Is)" | tee -a $D/a2_ident/launch_a2c.log
