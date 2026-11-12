"""Détecteurs de dérive sur embeddings, sans labels.

Chaque détecteur est ajusté sur une référence, puis `score(lot)` retourne un
nombre : plus il est grand, plus le lot s'éloigne de la référence.
"""
import numpy as np
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA


class KSPCA:
    """Test de Kolmogorov-Smirnov par composante PCA.

    score = -log10(p-value minimale corrigée par Bonferroni) : grand = dérive forte.
    """
    def __init__(self, n_components=10):
        self.n_components = n_components

    def fit(self, ref):
        ref = np.asarray(ref, dtype=np.float64)
        self.pca = PCA(self.n_components, random_state=0).fit(ref)
        self.ref_proj = self.pca.transform(ref)
        return self

    def score(self, x):
        proj = self.pca.transform(np.asarray(x, dtype=np.float64))
        pvals = [float(ks_2samp(self.ref_proj[:, k], proj[:, k]).pvalue)
                 for k in range(self.n_components)]
        p = min(1.0, min(pvals) * self.n_components)      # Bonferroni
        return float(-np.log10(max(p, 1e-300)))


class MMD:
    """MMD² (noyau RBF, largeur = médiane des distances) entre référence et lot."""
    def __init__(self, n_ref=1000, seed=0):
        self.n_ref, self.seed = n_ref, seed

    def fit(self, ref):
        ref = np.asarray(ref, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        self.ref = ref[rng.choice(len(ref), min(self.n_ref, len(ref)), replace=False)]
        d = self._sqdist(self.ref[:500], self.ref[:500])
        self.gamma = 1.0 / np.median(d[d > 0])
        return self

    @staticmethod
    def _sqdist(a, b):
        return (a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2 * a @ b.T

    def _k(self, a, b):
        return np.exp(-self.gamma * self._sqdist(a, b))

    def score(self, x):
        x = np.asarray(x, dtype=np.float64)
        n, m = len(self.ref), len(x)
        kxx, kyy, kxy = self._k(self.ref, self.ref), self._k(x, x), self._k(self.ref, x)
        # estimateur non biaisé : on retire les diagonales
        mmd2 = ((kxx.sum() - np.trace(kxx)) / (n * (n - 1))
                + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                - 2 * kxy.mean())
        return float(mmd2)


class Mahalanobis:
    """Distance de Mahalanobis moyenne au centroïde de la classe la plus proche.

    Moyennes par classe, covariance partagée. Les pseudo-labels de la référence
    sont ceux prédits par le modèle : aucune vérité terrain n'est utilisée.
    """
    def fit(self, ref, ref_pred):
        ref = np.asarray(ref, dtype=np.float64)
        self.classes = np.unique(ref_pred)
        self.means = np.stack([ref[ref_pred == c].mean(0) for c in self.classes])
        centered = ref - self.means[np.searchsorted(self.classes, ref_pred)]
        cov = centered.T @ centered / len(ref) + 1e-3 * np.eye(ref.shape[1])
        self.prec = np.linalg.inv(cov)
        return self

    def score(self, x):
        x = np.asarray(x, dtype=np.float64)
        d = np.stack([np.einsum('ij,jk,ik->i', x - m, self.prec, x - m)
                      for m in self.means], axis=1)
        return float(np.sqrt(d.min(1)).mean())


class MaxSoftmax:
    """Baseline : 1 - confiance moyenne du modèle. Nécessite les logits, pas les embeddings."""
    @staticmethod
    def score(probs):
        return float(1.0 - probs.max(1).mean())