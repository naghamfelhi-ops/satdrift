"""Analyse par sous-lots : score de dérive vs baisse de précision."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from satdrift.drift.detectors import KSPCA, MMD, Mahalanobis

A = 'artifacts'
BS = 200                                   # taille de lot réaliste
CORR = ['blur', 'noise', 'haze', 'brightness']
rng = np.random.default_rng(0)

ref = np.load(f'{A}/emb_ref_clean.npz')
DET = {
    'KS': KSPCA().fit(ref['emb']).score,
    'MMD': MMD().fit(ref['emb']).score,
    'Mahal': Mahalanobis().fit(ref['emb'], ref['pred']).score,
}

clean = dict(np.load(f'{A}/emb_test_clean.npz'))
n = len(clean['y'])
perm = rng.permutation(n)
calib_pool, eval_pool = perm[:n // 2], perm[n // 2:]
eval_batches = [eval_pool[i:i + BS] for i in range(0, len(eval_pool) - BS + 1, BS)]


def scores(emb):
    return {k: f(emb) for k, f in DET.items()}


# --- seuils : 95e percentile sur 100 lots propres du pool de calibration ---
calib = pd.DataFrame([scores(clean['emb'][rng.choice(calib_pool, BS, replace=False)])
                      for _ in range(100)])
thr = calib.quantile(0.95)
print('Seuils (95e percentile, lots propres) :')
print(thr.round(4).to_string(), '\n')

# --- évaluation ---
names = ['clean'] + [f'{c}_{s}' for c in CORR for s in range(1, 6)]
rows = []
for name in names:
    d = clean if name == 'clean' else dict(np.load(f'{A}/drift/emb_test_{name}.npz'))
    corr, lvl = ('clean', 0) if name == 'clean' else (name.split('_')[0], int(name.split('_')[1]))
    for b, idx in enumerate(eval_batches):
        acc = (d['y'][idx] == d['pred'][idx]).mean()
        acc0 = (clean['y'][idx] == clean['pred'][idx]).mean()
        rows.append(dict(corruption=corr, level=lvl, batch=b, acc=acc,
                         drop=acc0 - acc, **scores(d['emb'][idx])))
df = pd.DataFrame(rows)
df.to_csv(f'{A}/batch_scores.csv', index=False)

# --- corrélation score vs baisse de précision ---
print('Spearman (score vs baisse de précision), tous lots confondus :')
for k in DET:
    print(f'  {k:6s} rho = {spearmanr(df[k], df["drop"])[0]:.3f}')
print('\nPar type de dérive :')
for c in CORR:
    sub = df[df.corruption == c]
    print(f'  {c:11s}', *[f'{k}={spearmanr(sub[k], sub["drop"])[0]:.2f}' for k in DET])

# --- taux d'alerte (la ligne clean/0 donne le taux de fausses alertes) ---
for k in DET:
    df[f'{k}_alarm'] = df[k] > thr[k]
tab = df.groupby(['corruption', 'level'])[['drop'] + [f'{k}_alarm' for k in DET]].mean().round(2)
print("\nBaisse de précision moyenne et taux d'alerte par (dérive, niveau) :")
print(tab.to_string())