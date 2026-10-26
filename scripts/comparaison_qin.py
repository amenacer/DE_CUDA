#!/usr/bin/env python3
"""comparaison_qin.py -- tests complémentaires et comparaison avec cudaDEi.

1. Les trois implémentations du même DE/rand/1/bin (sde, cuda_v1, cuda_v2) donnent-elles
   la même qualité ? Kruskal-Wallis entre ces trois seules (jDE est un autre algorithme,
   il est traité à part par analyse.py).
2. Comparaison avec les résultats publiés de cudaDEi :
   A. K. Qin, F. Raimondo, F. Forbes, Y. S. Ong, "An improved CUDA-based implementation
   of differential evolution on GPU", GECCO 2012, tableaux 1 à 3.
   Même protocole que le nôtre : DE/rand/1/bin, F = 0,5, CR = 0,3, 10^4 x D évaluations,
   D = 10/50/100, NP = 50/100/500(/1000), mêmes 4 fonctions CEC 2005 décalées.
   Différences : 25 runs (nous 10), GPU GTX 560 + CUDA 4.0 (nous Tesla T4 + CUDA 13),
   précision flottante, gestion des bornes et périmètre du chronométrage non précisés
   dans l'article.

  python3 scripts/comparaison_qin.py --results results/campagne.csv --outdir results/analyse
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

# cudaDEi, GECCO 2012, tableaux 1-3 : (erreur moyenne, écart-type, temps moyen en s, taux de succès)
# Fonctions de l'article : F1 sphere, F2 rosenbrock, F3 griewank, F4 rastrigin. 25 runs.
QIN = {
    (10, "sphere"):     {50: (0.000, 0.000, 1.159, 1.00), 100: (0.000, 0.000, 0.633, 1.00), 500: (0.000, 0.000, 0.228, 1.00)},
    (10, "rosenbrock"): {50: (0.388, 1.235, 1.159, 0.44), 100: (0.044, 0.068, 0.632, 0.32), 500: (1.306, 1.388, 0.229, 0.00)},
    (10, "griewank"):   {50: (0.000, 0.000, 1.261, 1.00), 100: (0.000, 0.000, 0.648, 1.00), 500: (0.000, 0.000, 0.222, 1.00)},
    (10, "rastrigin"):  {50: (0.000, 0.000, 1.158, 1.00), 100: (0.000, 0.000, 0.635, 1.00), 500: (0.000, 0.000, 0.207, 1.00)},
    (50, "sphere"):     {50: (0.000, 0.000, 6.881, 1.00), 100: (0.000, 0.000, 4.937, 1.00), 500: (0.000, 0.000, 3.453, 1.00)},
    (50, "rosenbrock"): {50: (213.544, 625.959, 6.940, 0.00), 100: (12.244, 7.100, 5.055, 0.04), 500: (46.968, 19.274, 3.549, 0.00)},
    (50, "griewank"):   {50: (0.001, 0.002, 8.142, 0.92), 100: (0.000, 0.000, 5.902, 1.00), 500: (0.000, 0.000, 4.349, 1.00)},
    (50, "rastrigin"):  {50: (18.286, 3.97, 7.006, 0.00), 100: (2.936, 1.472, 4.914, 0.04), 500: (28.345, 14.084, 3.851, 0.04)},
    (100, "sphere"):     {50: (0.000, 0.000, 18.624, 1.00), 100: (0.000, 0.000, 16.396, 1.00), 500: (0.000, 0.000, 12.289, 1.00)},
    (100, "rosenbrock"): {50: (15.720, 24.466, 19.003, 0.00), 100: (42.293, 32.790, 16.919, 0.00), 500: (113.292, 22.349, 12.679, 0.00)},
    (100, "griewank"):   {50: (0.002, 0.006, 22.376, 0.84), 100: (0.005, 0.027, 20.929, 0.96), 500: (0.000, 0.002, 15.784, 0.96)},
    (100, "rastrigin"):  {50: (63.166, 8.984, 18.862, 0.00), 100: (48.329, 9.172, 17.417, 0.00), 500: (40.891, 8.539, 13.236, 0.00)},
}
FUNCS = ["sphere", "rosenbrock", "griewank", "rastrigin"]
SUCCES = 1e-8  # seuil de succès de l'article


def md(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    return s + "".join("| " + " | ".join(map(str, r)) + " |\n" for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/campagne.csv")
    ap.add_argument("--outdir", default="results/analyse")
    a = ap.parse_args()
    df = pd.read_csv(a.results)
    out = []

    # 1. équivalence des trois implémentations du même algorithme
    rows, n_sig = [], 0
    for f in FUNCS:
        for D in (10, 50, 100):
            for N in (50, 100, 500):
                g = [df[(df.algo == al) & (df.func == f) & (df.D == D) & (df.N == N)].best_error.to_numpy()
                     for al in ("sde", "cuda_v1", "cuda_v2")]
                if np.ptp(np.concatenate(g)) == 0:
                    h, p = float("nan"), 1.0
                else:
                    r = stats.kruskal(*g)
                    h, p = r.statistic, r.pvalue
                n_sig += p < 0.05
                rows.append([f, D, N] + [f"{x.mean():.3e}" for x in g] + ["-" if np.isnan(h) else f"{h:.2f}", f"{p:.3g}",
                             "différence" if p < 0.05 else "="])
    out.append("### 1. Les trois implémentations du DE/rand/1/bin (sde, cuda_v1, cuda_v2)\n")
    out.append(f"Kruskal-Wallis à 3 groupes, 10 runs, seuil 5 % (p brut) : différence détectée dans "
               f"{n_sig} configurations sur 36.\n\n")
    out.append(md(["fonction", "D", "N", "sde", "cuda_v1", "cuda_v2", "H", "p", "verdict"], rows))

    # 2. comparaison avec cudaDEi
    rows = []
    for D in (10, 50, 100):
        for f in FUNCS:
            for N in (50, 100, 500):
                m_q, s_q, t_q, sr_q = QIN[(D, f)][N]
                v2 = df[(df.algo == "cuda_v2") & (df.func == f) & (df.D == D) & (df.N == N)]
                e, t = v2.best_error.to_numpy(), v2.time_s.to_numpy()
                rows.append([D, f, N, f"{m_q:.3g} ({s_q:.3g})", f"{e.mean():.3g} ({e.std(ddof=1):.3g})",
                             f"{sr_q:.2f}", f"{(e < SUCCES).mean():.2f}",
                             f"{t_q:.3f}", f"{t.mean():.4f}", f"{t_q / t.mean():.0f}"])
    out.append("\n### 2. Comparaison avec cudaDEi (Qin et al., GECCO 2012)\n\n")
    out.append("Erreur finale moyenne (écart-type), taux de succès (erreur < 1e-8) et temps moyen (s). "
               "cudaDEi : 25 runs, GTX 560, CUDA 4.0. Notre v2 : 10 runs, Tesla T4, CUDA 13. "
               "Le rapport des temps compare deux GPU de générations différentes : il n'isole pas l'effet de "
               "l'implémentation.\n\n")
    out.append(md(["D", "fonction", "N", "cudaDEi erreur", "v2 erreur", "cudaDEi SR", "v2 SR",
                   "cudaDEi t (s)", "v2 t (s)", "t cudaDEi / t v2"], rows))
    os.makedirs(os.path.join(a.outdir, "tables"), exist_ok=True)
    p = os.path.join(a.outdir, "tables", "comparaison_qin.md")
    open(p, "w", encoding="utf-8").write("".join(out))
    print("".join(out))


if __name__ == "__main__":
    main()
