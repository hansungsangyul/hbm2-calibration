"""
지시 11 Step 2 — fwhm 앵커 검증: perturbed init 실험
fwhm 초기값만 1.5 / 3.5로 변경, 나머지 v3 설정 동일
seeds: 930000, 931000
"""
import csv, json, sys, time, numpy as np, cv2, torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT/'src'))
from pipeline import DegradationPipeline, softplus_inverse
from losses import CalibrationLoss

TEMPLATE_DIR = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_clean')
REAL_DIR     = Path('/mnt/gstore/home/xiangyue/HBM_paper/download/HBM_dirty2')
PHASE4_DIR   = Path(__file__).parent
N_STEPS, BATCH_SIZE = 2000, 8
PRENORM_A, PRENORM_B = 0.501834, 0.173320
N_LO, N_HI = 50.0, 1000.0
SIGMA_G_MAX = 0.042; SIGMA_G_MIN = 1e-4
IN_LOW_MIN, IN_LOW_MAX = 0.0, 0.3; IN_HIGH_MIN, IN_HIGH_MAX = 0.7, 1.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def n_sigmoid_inv(N):
    r = np.clip((N-N_LO)/(N_HI-N_LO), 1e-6, 1-1e-6)
    return float(np.log(r/(1-r)))
def n_from_raw(raw): return N_LO + (N_HI-N_LO)*torch.sigmoid(raw)

THETA_MANUAL = dict(N=300.0, sigma_g=0.0408, fwhm=2.5, p=0.001, alpha=0.35,
    in_low=0.05, in_high=0.90, out_low=0.10, out_high=0.90, ring_strength=0.12, ring_n=8.0)

print(f"Device: {DEVICE}")
clean_paths = sorted(TEMPLATE_DIR.glob('*.png'))[:50]
tmpl_np = [PRENORM_A*cv2.imread(str(p),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0+PRENORM_B for p in clean_paths]
H,W = tmpl_np[0].shape
tmpl_t = [torch.from_numpy(t).to(DEVICE) for t in tmpl_np]
real_np = [cv2.imread(str(p),cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
           for p in sorted(REAL_DIR.glob('*.png'))[:50]]
loss_fn = CalibrationLoss(real_np).to(DEVICE)
print("Loss built.")

PERTURBED_RUNS = [
    {'fwhm_init': 1.5, 'seed': 930000, 'label': 'fwhm_init_1.5'},
    {'fwhm_init': 3.5, 'seed': 931000, 'label': 'fwhm_init_3.5'},
]

results = []
for cfg in PERTURBED_RUNS:
    fwhm0, seed, label = cfg['fwhm_init'], cfg['seed'], cfg['label']
    print(f"\n[{label}] seed={seed}")
    pipe = DegradationPipeline().to(DEVICE)
    theta_init = THETA_MANUAL.copy()
    theta_init['fwhm'] = fwhm0
    with torch.no_grad():
        pipe.N_raw.fill_(n_sigmoid_inv(theta_init['N']))
        pipe.sigma_g_raw.fill_(softplus_inverse(theta_init['sigma_g']))
        pipe.fwhm_raw.fill_(softplus_inverse(fwhm0))
        pipe.alpha_raw.fill_(softplus_inverse(theta_init['alpha']))
        pipe.ring_strength_raw.fill_(softplus_inverse(theta_init['ring_strength']))
        pipe.ring_n_raw.fill_(softplus_inverse(theta_init['ring_n']))
        pipe.in_low.fill_(theta_init['in_low']); pipe.in_high.fill_(theta_init['in_high'])
        pipe.out_low.fill_(0.10); pipe.out_high.fill_(0.90); pipe.p.fill_(0.001)

    opt = torch.optim.Adam([
        {'params':[pipe.N_raw],'lr':1.0},
        {'params':[pipe.sigma_g_raw,pipe.fwhm_raw,pipe.alpha_raw,pipe.ring_strength_raw,pipe.ring_n_raw],'lr':0.01},
        {'params':[pipe.in_low,pipe.in_high,pipe.out_low,pipe.out_high],'lr':0.01},
    ])

    torch.manual_seed(seed)
    fwhm_trace = []
    for step in range(N_STEPS):
        opt.zero_grad()
        x = tmpl_t[step % len(tmpl_t)]
        loss = torch.tensor(0.0, device=DEVICE)
        for _ in range(BATCH_SIZE):
            eps = {'eps1':torch.randn(H,W,device=DEVICE),'eps2':torch.randn(H,W,device=DEVICE),'eps3':torch.rand(H,W,device=DEVICE)}
            L,_ = loss_fn(pipe(x,eps)); loss = loss + L
        (loss/BATCH_SIZE).backward(); opt.step()
        with torch.no_grad():
            sg = float(F.softplus(pipe.sigma_g_raw).item())
            sg_c = float(np.clip(sg, SIGMA_G_MIN, SIGMA_G_MAX))
            if abs(sg-sg_c)>1e-9: pipe.sigma_g_raw.fill_(softplus_inverse(sg_c))
            pipe.in_low.clamp_(IN_LOW_MIN,IN_LOW_MAX); pipe.in_high.clamp_(IN_HIGH_MIN,IN_HIGH_MAX)
            pipe.N_raw.clamp_(-6.,6.)
        fwhm_now = float(F.softplus(pipe.fwhm_raw).item())
        fwhm_trace.append(fwhm_now)
        if (step+1)%400==0:
            N_now = float(n_from_raw(pipe.N_raw).item())
            print(f"  step {step+1:4d}  fwhm={fwhm_now:.4f}  N={N_now:.1f}  L={loss.item()/BATCH_SIZE:.5f}")

    final_fwhm = float(F.softplus(pipe.fwhm_raw).item())
    N_final    = float(n_from_raw(pipe.N_raw).item())
    anchored   = max(abs(v-fwhm0) for v in fwhm_trace) < 0.5  # stayed within 0.5 of init
    converged  = 2.4 <= final_fwhm <= 2.6  # converged to ~2.5 range
    print(f"  final fwhm={final_fwhm:.4f}  N={N_final:.1f}  converged={converged}  anchored={anchored}")
    res = {'label':label,'fwhm_init':fwhm0,'seed':seed,'final_fwhm':final_fwhm,
           'final_N':N_final,'fwhm_min':min(fwhm_trace),'fwhm_max':max(fwhm_trace),
           'converged_to_25':bool(converged),'anchored_to_init':bool(anchored),
           'fwhm_at_500':fwhm_trace[499],'fwhm_at_1000':fwhm_trace[999],'fwhm_at_2000':fwhm_trace[-1]}
    results.append(res)
    # Save trace
    out_dir = PHASE4_DIR / f'rep_{label}'
    out_dir.mkdir(exist_ok=True)
    with open(out_dir/'fwhm_trace.csv','w',newline='') as f:
        w = csv.writer(f); w.writerow(['step','fwhm']); w.writerows(enumerate(fwhm_trace))

# Verdict
both_converge = all(r['converged_to_25'] for r in results)
verdict = 'a' if both_converge else 'b'
print(f"\n=== 판별 결과 ===")
for r in results:
    print(f"  {r['label']}: final={r['final_fwhm']:.4f}  converged={r['converged_to_25']}")
print(f"  최종 판별: ({verdict}) {'탐색 후 수렴 — 확정' if verdict=='a' else '앵커 — 일부 잔류'}")

output = {'step1_trace_analysis': {'verdict_prelim':'a','excursion_steps_pct':10.5,'max_deviation':0.062},
          'step2_perturbed_init': results,
          'final_verdict': verdict,
          'interpretation': '(a) fwhm converges to ~2.5 regardless of init' if verdict=='a' else '(b) fwhm anchored to init in some runs',
          'paper_action': 'fwhm 2.503 식별 확정: §4.1 기재 + D_calib 적용' if verdict=='a' else 'fwhm §3.3 앵커로 기재, D_calib fwhm=manual'}
json.dump(output, open(PHASE4_DIR/'fwhm_anchor_test.json','w'), indent=2)
print("Saved: fwhm_anchor_test.json")
