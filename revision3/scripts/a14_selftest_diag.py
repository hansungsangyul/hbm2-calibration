#!/usr/bin/env python3
"""A14 self-test and generation diagnostics (no training, no test-set prediction).
(1) degrade(mode='none') == DegradationPipeline.forward(x) and degrade(mode='pre') == forward(a x + b), bit-exact.
(2) Algebraic identities at stage 2 output for c1/c2 against closed forms.
(3) Diagnostics on 50 training images (image_0001..): final-image mean/SD, stage-6 pre-clamp fractions below 0 and
    above 1, noiseless-image SD, effective noise SD (noisy minus noiseless final image) and their ratio.
Output: <script dir>/a14/selftest_diag.json
"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np, cv2, torch
HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('a14t', str(HERE/'a14_train.py'))
# import helpers without running training: exec only the definitions we need
src = open(HERE/'a14_train.py').read()
cut = src.index('# ── Args')
ns = {}
exec(compile(src[:cut], 'a14_train_head', 'exec'), ns)
exec(compile(src[src.index('def degrade'):src.index('# ── Training Dataset')], 'a14_degrade', 'exec'), ns)
A, B = ns['PRENORM_A'], ns['PRENORM_B']
degrade = ns['degrade']
fwhm = 2.5
ns['fwhm'] = fwhm
make_pipe = None
# rebuild make_pipe with fwhm in scope
exec(compile(src[src.index('def make_pipe'):src.index('def degrade')], 'a14_mkpipe', 'exec'), ns)
pipe = ns['make_pipe']()
AUG = ns['AUGMENT_DIR']
imgs = sorted((AUG/'train_data/HBM_clean').glob('image_*.png'))[:50]
assert len(imgs) == 50, f'Expected 50 diagnostic images, got {len(imgs)}'
res = {'selftest': {}, 'diag': {}}
maxdiff = {'none': 0.0, 'pre': 0.0, 'c1_stage2': 0.0, 'c2_stage2': 0.0}
stats = {m: {'mean': [], 'sd': [], 'clip_lo': [], 'clip_hi': [], 'sd_clean': [], 'sd_noise': []} for m in ['none','pre','post','prepost']}
def stage6_preclamp(pipe, y):
    r = (y - pipe.in_low) / (pipe.in_high - pipe.in_low + 1e-8)
    return r * (pipe.out_high - pipe.out_low) + pipe.out_low
def run_to_stage5(pipe, x, eps, mode, noisy=True):
    if mode in ('pre','prepost'): x = A*x + B
    if noisy:
        y = pipe._stage2_gaussian(pipe._stage1_poisson(x, eps['eps1']), eps['eps2'])
    else:
        y = x.clone()
    if mode == 'post': y = A*y + B
    elif mode == 'prepost': y = (y - B)/A
    if noisy: y = pipe._stage3_salt_pepper(y, eps['eps3'])
    y = pipe._stage4_psf_blur(y); y = pipe._stage5_vignetting(y)
    return y
with torch.no_grad():
    for k, p in enumerate(imgs):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        H, W = img.shape
        x = torch.from_numpy(img)
        torch.manual_seed(k)
        eps = {'eps1': torch.randn(H,W), 'eps2': torch.randn(H,W), 'eps3': torch.rand(H,W)}
        ref_none = pipe(x, eps); ref_pre = pipe(A*x+B, eps)
        maxdiff['none'] = max(maxdiff['none'], float((degrade(pipe,x,eps,'none')-ref_none).abs().max()))
        maxdiff['pre'] = max(maxdiff['pre'], float((degrade(pipe,x,eps,'pre')-ref_pre).abs().max()))
        N = float(300.0); sg = 0.0408
        n0 = torch.sqrt(torch.clamp(x,min=1e-6)/N)*eps['eps1'] + sg*eps['eps2']
        n1 = (torch.sqrt(torch.clamp(A*x+B,min=1e-6)/N)*eps['eps1'] + sg*eps['eps2'])/A
        s2_c1 = A*pipe._stage2_gaussian(pipe._stage1_poisson(x,eps['eps1']),eps['eps2'])+B
        s2_c2 = (pipe._stage2_gaussian(pipe._stage1_poisson(A*x+B,eps['eps1']),eps['eps2'])-B)/A
        maxdiff['c1_stage2'] = max(maxdiff['c1_stage2'], float((s2_c1-(A*(x+n0)+B)).abs().max()))
        maxdiff['c2_stage2'] = max(maxdiff['c2_stage2'], float((s2_c2-(x+n1)).abs().max()))
        for m in stats:
            y5 = run_to_stage5(pipe, x, eps, m, True); y5c = run_to_stage5(pipe, x, eps, m, False)
            pc = stage6_preclamp(pipe, y5)
            fin = degrade(pipe, x, eps, m)
            finc = pipe._stage7_ring(pipe._stage6_contrast(y5c))
            stats[m]['mean'].append(float(fin.mean())); stats[m]['sd'].append(float(fin.std()))
            stats[m]['clip_lo'].append(float((pc<0).float().mean())); stats[m]['clip_hi'].append(float((pc>1).float().mean()))
            stats[m]['sd_clean'].append(float(finc.std())); stats[m]['sd_noise'].append(float((fin-finc).std()))
res['selftest'] = maxdiff
res['selftest_pass'] = bool(maxdiff['none']==0.0 and maxdiff['pre']==0.0 and maxdiff['c1_stage2']<1e-5 and maxdiff['c2_stage2']<1e-5)
name = {'none':'D_manual','pre':'AB-1','post':'C1','prepost':'C2'}
for m, d in stats.items():
    res['diag'][name[m]] = {k: float(np.median(v)) for k, v in d.items()}
    res['diag'][name[m]]['snr_eff'] = float(np.median(np.array(d['sd_clean'])/np.array(d['sd_noise'])))
out = HERE / 'a14' / 'selftest_diag.json'
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(out, 'w'), indent=2)
print(json.dumps(res, indent=2))
