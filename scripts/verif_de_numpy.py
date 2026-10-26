#!/usr/bin/env python3
"""verif_de_numpy.py -- contrôle indépendant de la qualité attendue du DE.

Implémentation NumPy minimale, écrite indépendamment du code C++/CUDA, du DE/rand/1/bin
générationnel (F = 0,5, CR = 0,3), sur Rastrigin décalée, D = 50, N = 100, domaine [-5, 5].
Elle sert à vérifier si l'écart entre nos résultats et ceux publiés pour cudaDEi
(Qin et al., GECCO 2012 : erreur 2,936 sur Rastrigin D = 50, NP = 100) peut venir de la
gestion des bornes ou du budget d'évaluations.

  python3 scripts/verif_de_numpy.py            (environ 2 min)

Résultats obtenus (graines 1 à 3) :
  bornes ramenées au bord (clamp)   : 169,1  168,1  165,4
  bornes réinitialisées au hasard   : 172,7  151,2  162,9
  aucune gestion des bornes         : 168,9  174,1  147,3
  budget x2 / x5 / x10 (graine 1)   : 143,7  103,1  79,6
Notre campagne (cuda_v2, 10 runs) : 165,5 en moyenne. Aucune de ces variantes n'approche 2,9.
"""
import numpy as np


def rastrigin(Z):
    return (Z ** 2 - 10 * np.cos(2 * np.pi * Z) + 10).sum(1)


def de(D=50, N=100, F=0.5, CR=0.3, bound="clamp", seed=1, lo=-5.0, hi=5.0, mult=1):
    rng = np.random.default_rng(seed)
    o = rng.uniform(0.8 * lo, 0.8 * hi, D)
    X = rng.uniform(lo, hi, (N, D)); f = rastrigin(X - o); fe = N
    idx = np.arange(N)
    while fe < mult * 10000 * D:
        r = np.array([rng.choice(np.delete(idx, i), 3, replace=False) for i in range(N)])
        V = X[r[:, 0]] + F * (X[r[:, 1]] - X[r[:, 2]])
        mask = rng.random((N, D)) < CR; mask[idx, rng.integers(0, D, N)] = True
        U = np.where(mask, V, X)
        if bound == "clamp":
            U = np.clip(U, lo, hi)
        elif bound == "random":
            bad = (U < lo) | (U > hi); U[bad] = rng.uniform(lo, hi, bad.sum())
        fu = rastrigin(U - o); fe += N
        b = fu <= f; X[b] = U[b]; f[b] = fu[b]
    return f.min()


if __name__ == "__main__":
    for b in ("clamp", "random", "none"):
        print(b, [round(de(bound=b, seed=s), 1) for s in (1, 2, 3)])
    # budget multiplié (graine 1) : chaque appel repart de zéro, compter environ 5 min au total
    for m in (2, 5, 10):
        print(f"budget x{m}", round(de(seed=1, mult=m), 1))
