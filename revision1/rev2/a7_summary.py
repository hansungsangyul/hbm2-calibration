import pandas as pd, numpy as np
d = pd.read_csv('a7/a7_per_image.csv')
g = d.groupby(['config', 'set']).agg(n_obs=('pixel_fpr', 'size'), median_fpr=('pixel_fpr', 'median'), p5_fpr=('pixel_fpr', lambda x: np.percentile(x, 5)), min_fpr=('pixel_fpr', 'min'), mean_bg=('bg_px', 'mean'), frac_imgs_fpr_lt_0p9=('pixel_fpr', lambda x: (x < 0.9).mean())).reset_index()
g.to_csv('a7/a7_summary.csv', index=False); print(g.round(4).to_string(index=False))
