# Fig. 2 (r2): one content-matched pair from the paired evaluation: clean image 0251 degraded by model A (each configuration's own pipeline)
# and by the reference pipeline (R), with predictions of D_calib and D_manual (run0 each) and the tile-label grid of R. Image choice pre-specified: first held-out image (0251).
from common import *; import json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; from matplotlib.patches import Rectangle
from dataset import json_to_mask
IMG='image_0251'; AIN=Path('/mnt/gstore/home/xiangyue/p3stats/a8_paired'); LBL=Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_labels')
lab=json.load(open(LBL/f'label_{IMG.split("_")[1]}.json'))
R_img=load_real(Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2')/f'{IMG}.png')
cfgs=['D_calib','D_manual']; preds={}; imgs={}; RUNS=json.load(open('/mnt/gstore/home/xiangyue/p3stats/a8_paired/fig2_runs.json'))
for c in cfgs:
    rd,mode=CONFIGS[c]; ck=[k for k in ckpts(rd) if k.parent.name==RUNS[c]][0]; m=load_model(ck)
    a=np.load(AIN/c/f'{IMG}.npy').astype(np.float32); imgs[('A',c)]=a
    preds[('A',c)]=predict(m,[a],bs=1)[0].numpy()
    preds[('R',c)]=predict(m,[(prep(R_img,'zscore') if mode=='zscore' else R_img).astype(np.float32)],bs=1)[0].numpy()  # raw input convention
    del m; torch.cuda.empty_cache()
VAL=slice(3,317)
COL={1:(0.0,0.45,0.70),2:(0.90,0.62,0.0),3:(0.80,0.47,0.65)}  # crack blue, open orange, hole purple (Okabe-Ito)
NAMES={0:'background',1:'crack',2:'open',3:'hole'}
def overlay(ax,img,pred,title):
    ax.imshow(img[VAL],cmap='gray',vmin=0,vmax=1); ov=np.zeros((314,512,4))
    for k,cc in COL.items(): sel=(pred[VAL]==k); ov[sel]=(*cc,0.55)
    ax.imshow(ov); ax.set_title(title,fontsize=7.5); ax.set_xticks([]); ax.set_yticks([])
def grid(ax):
    for it in lab:
        r=it['row']-1; c=it['col']-1; y0=r*62; y1=(r+1)*62 if r<4 else 314; x0=c*26-4; x1=((c+1)*26 if c<19 else 520)-4
        x0=max(x0,0); x1=min(x1,512)
        if it['label']=='normal': ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec='white',lw=0.8))
        else: ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec='black',lw=0.3,ls=':'))
fig,axes=plt.subplots(2,3,figsize=(7.0,3.6))
for i,side in enumerate(['A','R']):
    base=imgs[('A','D_calib')] if side=='A' else R_img
    axes[i,0].imshow(base[VAL],cmap='gray',vmin=0,vmax=1); grid(axes[i,0]); axes[i,0].set_xticks([]); axes[i,0].set_yticks([])
    axes[i,0].set_title(('A-degraded (own pipeline)' if side=='A' else 'R-degraded (reference pipeline)'),fontsize=7)
    for j,c in enumerate(cfgs):
        p=preds[(side,c)]; overlay(axes[i,j+1],imgs[('A',c)] if side=='A' else R_img,p,('$D_\\mathrm{calib}$' if c=='D_calib' else '$D_\\mathrm{manual}$')+f' prediction ({RUNS[c]})' if side=='A' else ''); grid(axes[i,j+1])
        comp={k:int((p[VAL]==k).sum()) for k in range(4)}; tot=sum(comp.values())
        axes[i,j+1].set_xlabel(f"background {100*comp[0]/tot:.1f}%  crack {100*comp[1]/tot:.1f}%\nopen {100*comp[2]/tot:.1f}%  hole {100*comp[3]/tot:.1f}%",fontsize=6.2,labelpad=2)
n_normal=sum(1 for it in lab if it['label']=='normal')
fig.suptitle(f'{IMG}: {n_normal} normal tiles (white outline), {100-n_normal} defect tiles (dotted)',fontsize=7.5,y=0.995)
plt.tight_layout(h_pad=1.4); out=Path('/mnt/gstore/home/xiangyue/p3stats/a8_paired'); plt.savefig(out/'fig2_pair.pdf'); plt.savefig(out/'fig2_pair.png',dpi=150)
json.dump({'image':IMG,'n_normal':n_normal,'composition':{f'{s}_{c}':{NAMES[k]:int((preds[(s,c)][VAL]==k).sum()) for k in range(4)} for s in ('A','R') for c in cfgs}},open(out/'fig2_pair.json','w'),indent=1)
print('FIG_DONE', n_normal)
