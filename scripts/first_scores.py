import numpy as np
from satdrift.drift.detectors import KSPCA, MMD, Mahalanobis

A = 'artifacts'
ref = np.load(f'{A}/emb_ref_clean.npz')
clean = np.load(f'{A}/emb_test_clean.npz')

ks = KSPCA().fit(ref['emb'])
mmd = MMD().fit(ref['emb'])
mah = Mahalanobis().fit(ref['emb'], ref['pred'])

print(f"{'jeu':18s} {'acc':>6s} {'KS':>8s} {'MMD':>9s} {'Mahal':>8s}")
for name in ['clean'] + [f'{c}_{s}' for c in ['blur', 'noise', 'haze', 'brightness'] for s in (1, 3, 5)]:
    path = f'{A}/emb_test_clean.npz' if name == 'clean' else f'{A}/drift/emb_test_{name}.npz'
    d = np.load(path)
    acc = (d['y'] == d['pred']).mean()
    print(f"{name:18s} {acc:6.3f} {ks.score(d['emb']):8.2f} {mmd.score(d['emb']):9.5f} {mah.score(d['emb']):8.2f}")