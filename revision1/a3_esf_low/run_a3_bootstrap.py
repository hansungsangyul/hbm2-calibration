import numpy as np, pandas as pd, json
from pathlib import Path
rng = np.random.default_rng(20260926)
base = pd.read_csv('/mnt/gstore/home/xiangyue/HBM_paper2/f1_recompute_results/counts/per_image_counts.csv')
ab4 = pd.read_csv('/mnt/gstore/home/xiangyue/HBM_paper2/revision1/a3_esf_low/counts/per_image_counts.csv')
df = pd.concat([base[base.config.isin(['D_calib','D_manual','AB-1'])], ab4]); B=2000
def f1s(tp,fp,fn): return 2*tp/np.maximum(2*tp+fp+fn,1)
T={}
for (cfg,ts),g in df.groupby(['config','test_set']):
    runs=sorted(g.run_id.unique()); imgs=sorted(g.image_id.unique()); ri={r:i for i,r in enumerate(runs)}; ii={m:i for i,m in enumerate(imgs)}
    A=np.zeros((len(runs),len(imgs),4,3)); A[g.run_id.map(ri).values,g.image_id.map(ii).values,g.class_id.values,:]=g[['tp','fp','fn']].values
    S=A.sum(1); f=f1s(S[...,0],S[...,1],S[...,2]); T[(cfg,ts)]=dict(A=A,m4=f.mean(1),md=f[:,1:].mean(1))
from scipy import stats
rows=[]
for ts in ['TEST_ID','TEST_OOD','TEST_CAL']:
    for cfg in ['AB-1','D_calib','AB-4','D_manual']:
        d=T[(cfg,ts)]; rows.append(dict(test_set=ts,config=cfg,f=({'AB-1':2.5,'D_calib':2.096,'AB-4':1.941,'D_manual':2.5}[cfg]),n=len(d['m4']),macro4=d['m4'].mean(),sd4=d['m4'].std(ddof=1),macrod=d['md'].mean(),sdd=d['md'].std(ddof=1)))
    for a,b in [('AB-4','D_calib'),('AB-4','AB-1')]:
        x,y=T[(a,ts)],T[(b,ts)]; t=stats.ttest_ind(x['m4'],y['m4'],equal_var=False); td=stats.ttest_ind(x['md'],y['md'],equal_var=False)
        boots=[]
        for _ in range(B):
            rs=rng.integers(0,20,20); im=rng.integers(0,500,500)
            def m(d):
                Sb=d['A'][rs][:,im].sum(1); fb=f1s(Sb[...,0],Sb[...,1],Sb[...,2]); return fb.mean(1).mean(), fb[:,1:].mean(1).mean()
            mx,my=m(x),m(y); boots.append((mx[0]-my[0],mx[1]-my[1]))
        boots=np.array(boots)
        print(f"{ts} {a}-{b}: ?4={x['m4'].mean()-y['m4'].mean():+.4f} CI[{np.percentile(boots[:,0],2.5):+.3f},{np.percentile(boots[:,0],97.5):+.3f}] Welch t={t.statistic:.2f} p={t.pvalue:.3g} | ?defect={x['md'].mean()-y['md'].mean():+.4f} CI[{np.percentile(boots[:,1],2.5):+.3f},{np.percentile(boots[:,1],97.5):+.3f}] p={td.pvalue:.3g}")
pd.DataFrame(rows).to_csv('a3_summary.csv',index=False); print(pd.DataFrame(rows).round(4).to_string(index=False))
