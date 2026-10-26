"""Experience de controle : DE sequentiel generationnel (sde --generationnel).

Hypothese testee : les versions GPU sont parfois moins precises que le DE sequentiel parce
qu'elles remplacent la population en fin de generation (generationnel), alors que le
sequentiel remplace un individu des que son essai gagne (asynchrone).

Entrees : results/campagne.csv et results/sde_generationnel.csv (360 runs, memes graines)
Sortie  : results/analyse/tables/controle_generationnel.csv + resume sur la sortie standard

Mann-Whitney bilateral, p brutes, alpha = 0.05 ; deux echantillons identiques donnent p = 1.
Lancer les runs : voir l'en-tete de src/seq/sde.cpp (option --generationnel).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

RACINE = Path(__file__).resolve().parent.parent
ALPHA = 0.05


def p_mw(a, b):
    if np.all(np.concatenate([a, b]) == a[0]):
        return 1.0
    return mannwhitneyu(a, b, alternative="two-sided").pvalue


def p_kw(*ech):
    tout = np.concatenate(ech)
    if np.all(tout == tout[0]):
        return 1.0
    return kruskal(*ech).pvalue


def main():
    camp = pd.read_csv(RACINE / "results" / "campagne.csv")
    gen = pd.read_csv(RACINE / "results" / "sde_generationnel.csv")
    df = pd.concat([camp, gen])
    lignes = []
    for (f, d, n), g in df.groupby(["func", "D", "N"]):
        e = {a: g[g["algo"] == a]["best_error"].values
             for a in ["sde", "sde_gen", "cuda_v1", "cuda_v2"]}
        assert all(len(v) == 10 for v in e.values()), (f, d, n)
        lignes.append({
            "func": f, "D": d, "N": n,
            "mean_sde": e["sde"].mean(), "mean_sde_gen": e["sde_gen"].mean(),
            "mean_cuda_v1": e["cuda_v1"].mean(), "mean_cuda_v2": e["cuda_v2"].mean(),
            "p_gen_vs_sde": p_mw(e["sde_gen"], e["sde"]),
            "p_gen_vs_v1": p_mw(e["sde_gen"], e["cuda_v1"]),
            "p_gen_vs_v2": p_mw(e["sde_gen"], e["cuda_v2"]),
            "p_kw_sde_v1_v2": p_kw(e["sde"], e["cuda_v1"], e["cuda_v2"]),
        })
    r = pd.DataFrame(lignes)
    sortie = RACINE / "results" / "analyse" / "tables" / "controle_generationnel.csv"
    r.to_csv(sortie, index=False)

    diff_gen = set(map(tuple, r[r["p_gen_vs_sde"] < ALPHA][["func", "D", "N"]].values))
    diff_kw = set(map(tuple, r[r["p_kw_sde_v1_v2"] < ALPHA][["func", "D", "N"]].values))
    print(f"sde_gen differe de sde            : {len(diff_gen)} configurations sur 36")
    print(f"Kruskal-Wallis sde/v1/v2 significatif : {len(diff_kw)} configurations")
    print(f"memes configurations               : {diff_gen == diff_kw}")
    print(f"sde_gen indiscernable de v1        : {(r['p_gen_vs_v1'] >= ALPHA).sum()} / 36")
    print(f"sde_gen indiscernable de v2        : {(r['p_gen_vs_v2'] >= ALPHA).sum()} / 36")
    print(f"tableau : {sortie.relative_to(RACINE)}")


if __name__ == "__main__":
    main()
