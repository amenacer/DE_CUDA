"""Genere les lignes des tableaux d'erreur finale de l'article (tab:err50, tab:err, tab:err500).

Entrees : results/campagne.csv et results/analyse/tables/stats_pairwise.csv
Sortie  : lignes LaTeX sur la sortie standard (moyenne +- ecart-type d'echantillon,
          exposant + / - si l'algorithme est significativement meilleur / pire que sde,
          Mann-Whitney avec correction de Holm dans la configuration).

Usage : python scripts/table_erreurs_n100.py [N]   (N = 50, 100 ou 500 ; 100 par defaut)
"""
import math
import sys
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
FONCTIONS = [("sphere", "Sphere"), ("rosenbrock", "Rosenbrock"),
             ("griewank", "Griewank"), ("rastrigin", "Rastrigin")]
ALGOS = ["sde", "cuda_v1", "cuda_v2", "gpu_jde"]


def fmt(x):
    if x == 0:
        return "0"
    a = abs(x)
    if 0.01 <= a < 1000:
        d = max(0, 2 - int(math.floor(math.log10(a))))
        return f"{x:.{d}f}"
    e = int(math.floor(math.log10(a)))
    return f"{x / 10**e:.1f}{{\\times}}10^{{{e}}}"


def main():
    camp = pd.read_csv(RACINE / "results" / "campagne.csv")
    stats = pd.read_csv(RACINE / "results" / "analyse" / "tables" / "stats_pairwise.csv")
    camp = camp[camp["N"] == N]
    for k, (f, nom) in enumerate(FONCTIONS):
        for i, d in enumerate([10, 50, 100]):
            cellules = [nom if i == 0 else "", str(d)]
            for algo in ALGOS:
                v = camp[(camp["func"] == f) & (camp["D"] == d) & (camp["algo"] == algo)]["best_error"]
                assert len(v) == 10, (f, d, algo, len(v))
                marque = ""
                if algo != "sde":
                    r = stats[(stats["function"] == f) & (stats["dim"] == d)
                              & (stats["pop"] == N) & (stats["algo"] == algo)]
                    assert len(r) == 1, (f, d, algo)
                    marque = {"+": "\\textsuperscript{$+$}", "-": "\\textsuperscript{$-$}",
                              "=": ""}[r["decision"].iloc[0]]
                cellules.append(f"${fmt(v.mean())}\\pm{fmt(v.std(ddof=1))}${marque}")
            print(" & ".join(cellules) + r"\\")
        if k < len(FONCTIONS) - 1:
            print(r"\midrule")


if __name__ == "__main__":
    main()
