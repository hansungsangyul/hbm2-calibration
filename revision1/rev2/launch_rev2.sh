#!/bin/bash
PY=/home/xiangyue/.conda/envs/unet/bin/python; PYC=/mnt/gstore/home/xiangyue/HBM_paper/envs/hbm_cudnn/bin/python; D=/mnt/gstore/home/xiangyue/HBM_paper2/revision1/rev2
cd $D; mkdir -p a7 a8 a9 a10 a11; export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( cd /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe && $PYC $D/a8_esf.py > $D/a8/a8.log 2>&1; echo "A8 DONE $(date -Is)" >> $D/rev2.log ) &
$PY a7_fpr.py > a7/a7.log 2>&1 && $PY a7_summary.py > a7/a7_summary.txt 2>&1; echo "A7 DONE $(date -Is)" >> rev2.log
$PY a9_tiles.py > a9/a9.log 2>&1; echo "A9 inference DONE $(date -Is)" >> rev2.log
( $PY a9_analysis.py > a9/a9_analysis.log 2>&1; echo "A9 analysis DONE $(date -Is)" >> $D/rev2.log ) &
cd /mnt/gstore/home/xiangyue/HBM_paper2/diffpipe/phase4 && $PYC $D/a10_stress.py > $D/a10/a10.log 2>&1; cd $D; echo "A10 DONE $(date -Is)" >> rev2.log
cd /mnt/gstore/home/xiangyue/HBM_paper/hbm_augmentation
for W in "W1:1,2.841,22.79,233.3" "W2:1,0.01574,15.74,37.02"; do n=${W%%:*}; l=${W#*:}; for N in 50 100 300 600 1000; do
  [ -f $D/a11/prof_${n}_N${N}/result.json ] || $PYC $D/a11_profile.py --N $N --seed 970000 --lam $l --out $D/a11/prof_${n}_N${N} >> $D/a11/a11.log 2>&1; done; done
echo "A11a DONE $(date -Is)" >> $D/rev2.log
for Ni in 100 800; do for sg in 0.02 0.08; do for fi in 1.5 3.0; do
  [ -f $D/a11/joint_N${Ni}_sg${sg}_f${fi}/result.json ] || $PYC $D/a11_profile.py --N $Ni --N_init $Ni --sg_init $sg --f_init $fi --free_N --seed 970000 --out $D/a11/joint_N${Ni}_sg${sg}_f${fi} >> $D/a11/a11.log 2>&1; done; done; done
echo "A11b DONE $(date -Is)" >> $D/rev2.log
$PYC $D/a11c_kkt_hessian.py > $D/a11/a11c.log 2>&1; echo "A11c DONE $(date -Is)" >> $D/rev2.log
wait; echo "REV2 ALL DONE $(date -Is)" >> $D/rev2.log
