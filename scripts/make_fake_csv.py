#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fake_csv.py -- génère de FAUX résultats pour développer/tester analyse.py
avant que la vraie campagne (run_all.py) ne soit terminée.

!!! LES VALEURS SONT INVENTÉES. Ne jamais les utiliser dans l'article. !!!

Sorties (dans --outdir, par défaut ./fake_data) :
  results_FAKE.csv      1 ligne par run terminé
      algo,function,dim,pop,run,seed,best_fitness,error,time_s
  convergence_FAKE.csv  courbe de convergence de chaque run (échantillonnée)
      algo,function,dim,pop,run,evals,best_fitness

Le protocole reproduit celui du sujet :
  - Dim in {10, 50, 100}, Pop in {50, 100, 500}, 10 runs
  - budget d'évaluations FEs = 10^4 x Dim
  - 4 fonctions shiftées, chacune avec son biais f_bias
    (Sphere -450, Rastrigin -330, Rosenbrock 390, Griewank -180)

Les "algorithmes" fictifs :
  DE_seq      version séquentielle (référence)
  DE_gpu_v1   CUDA v1 : plusieurs kernels + copies hôte<->GPU à chaque génération
  DE_gpu_v2   CUDA v2 : kernels fusionnés, pas de copies -> même qualité, plus rapide
  DE_gpu_jDE  variante améliorée : meilleure qualité (pour voir les tests détecter une différence)

Usage :
  python make_fake_csv.py --outdir fake_data --seed 42
"""
import argparse
import os

import numpy as np
import pandas as pd

BIAS = {"sphere": -450.0, "rastrigin": -330.0, "rosenbrock": 390.0, "griewank": -180.0}

# log10 de l'erreur finale "typique" a Dim=10, Pop=50 (valeurs inventées)
LOG_ERR_BASE = {"sphere": -6.0, "rastrigin": 0.8, "rosenbrock": 1.5, "griewank": -2.0}
# log10 de l'erreur de la population initiale a Dim=10 (ordre de grandeur réaliste)
LOG_ERR_START = {"sphere": 4.5, "rastrigin": 2.5, "rosenbrock": 9.0, "griewank": 2.5}
# de combien l'erreur (en log10) augmente quand Dim est multipliée par 10
DIM_SLOPE = {"sphere": 3.0, "rastrigin": 1.0, "rosenbrock": 1.0, "griewank": 1.5}
# effet de la taille de population (a budget FEs fixe, une grande Pop = moins de générations)
POP_SLOPE = {"sphere": 0.5, "rastrigin": -0.3, "rosenbrock": 0.5, "griewank": 0.3}

# effet de l'algorithme sur log10(erreur) : seq, v1, v2 = même algorithme donc même qualité
ALGO_EFFECT = {"DE_seq": 0.0, "DE_gpu_v1": 0.0, "DE_gpu_v2": 0.0, "DE_gpu_jDE": -0.6}

DIMS = [10, 50, 100]
POPS = [50, 100, 500]


def model_time(algo, dim, pop, rng):
    """Temps d'exécution fictif (secondes) inspiré du comportement réel d'un GPU.

    - le séquentiel est proportionnel au travail total (FEs x Dim)
    - le GPU paie un coût fixe par lancement de kernel : pour un petit problème
      (Dim=10, Pop=50) il peut donc être PLUS LENT que le CPU.
    """
    fes = 1e4 * dim
    gens = fes / pop
    work = fes * dim * 2e-8  # temps séquentiel (s)
    if algo == "DE_seq":
        t = work
    else:
        parallel_units = min(pop * dim, 4000)
        compute = work / parallel_units * 5.0  # un coeur GPU est ~5x plus lent qu'un coeur CPU
        if algo == "DE_gpu_v1":
            overhead = 6 * 8e-6 + 2e-5  # 6 lancements + copies hôte<->GPU
        else:  # v2 et jDE : kernels fusionnés
            overhead = 2 * 8e-6
        t = compute + gens * overhead + 0.05  # 0.05 s = cudaMalloc + init cuRAND
    return t * rng.lognormal(0.0, 0.04)


def convergence_curve(le, ls, pop, fes, bias, rng, n_points=60):
    """Courbe fictive : erreur (log10) qui descend de ls (initiale) a le (finale)."""
    evals = np.unique(np.round(np.geomspace(pop, fes, n_points)).astype(int))
    x = np.log(evals / pop) / np.log(fes / pop)  # 0 -> 1
    log_err = le + (ls - le) * (1.0 - x) ** 1.5
    log_err = log_err + rng.normal(0, 0.05, size=len(evals)) * (1.0 - x)
    log_err[-1] = le  # le dernier point est exactement l'erreur finale
    log_err = np.minimum.accumulate(log_err)  # le meilleur trouvé ne remonte jamais
    return evals, bias + 10.0 ** log_err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="fake_data")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    res_rows, conv_parts = [], []
    for func, bias in BIAS.items():
        for dim in DIMS:
            for pop in POPS:
                fes = int(1e4 * dim)
                for algo, eff in ALGO_EFFECT.items():
                    for run in range(args.runs):
                        seed = 1000 + run  # même graine pour tous les algos -> appariement possible
                        le = (
                            LOG_ERR_BASE[func]
                            + DIM_SLOPE[func] * np.log10(dim / 10)
                            + POP_SLOPE[func] * np.log10(pop / 50)
                            + eff
                            + rng.normal(0, 0.35)
                        )
                        le = max(le, -12.0)
                        err = 10.0 ** le
                        res_rows.append(
                            dict(
                                algo=algo, function=func, dim=dim, pop=pop, run=run, seed=seed,
                                best_fitness=bias + err, error=err,
                                time_s=model_time(algo, dim, pop, rng),
                            )
                        )
                        ls = max(LOG_ERR_START[func] + np.log10(dim / 10) + rng.normal(0, 0.1), le + 0.5)
                        evals, fit = convergence_curve(le, ls, pop, fes, bias, rng)
                        conv_parts.append(
                            pd.DataFrame(
                                dict(algo=algo, function=func, dim=dim, pop=pop, run=run,
                                     evals=evals, best_fitness=fit)
                            )
                        )

    res = pd.DataFrame(res_rows)
    conv = pd.concat(conv_parts, ignore_index=True)
    p1 = os.path.join(args.outdir, "results_FAKE.csv")
    p2 = os.path.join(args.outdir, "convergence_FAKE.csv")
    res.to_csv(p1, index=False)
    conv.to_csv(p2, index=False)
    print("ATTENTION : données FICTIVES, uniquement pour tester les scripts d'analyse.")
    print(f"  {p1}  ({len(res)} lignes)")
    print(f"  {p2}  ({len(conv)} lignes)")


if __name__ == "__main__":
    main()
