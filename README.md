# DE_CUDA — Évolution différentielle massivement parallèle sur GPU

Projet du module **Calcul massivement parallèle**, M2 IMDS, Université de Haute-Alsace.
Amina Menacer · Lounis Mesrour · Lisa Adrar

## Le sujet en bref

Le code de départ est le **PSO CUDA du cours** (`src/pso_original/`, non modifié).
Nous l'avons transformé en **évolution différentielle** (DE/rand/1/bin, F = 0,5, CR = 0,3) exécutée sur GPU,
puis comparée à un DE séquentiel en C++, à budget égal (10⁴·D évaluations), sur quatre fonctions de test
décalées : Sphere, Rosenbrock, Griewank, Rastrigin (D = 10, 50, 100 ; N = 50, 100, 500 ; 10 runs chacune).

Le projet avance en quatre étapes :

1. **Comprendre le PSO du cours** : l'exécuter, relever ses défauts, mesurer où part son temps.
2. **Écrire un DE séquentiel** (C++, un cœur) qui sert de référence.
3. **Écrire le DE sur GPU**, en deux versions :
   - **version 1** : un thread par individu, trois kernels par génération ;
   - **version 2** : un bloc par individu et un thread par gène, deux kernels par génération,
     aucune copie CPU ↔ GPU dans la boucle.
4. **Comparer** : campagne de 1440 runs, tests statistiques, variantes (dont jDE),
   comparaison avec l'article de référence (Qin et al., GECCO 2012).

## Résultats en bref

| | Résultat |
|---|---|
| Vitesse | la version 2 va jusqu'à **58 fois** plus vite que le DE séquentiel (D = 100, N = 500) ; la version 1 au plus 9,5 fois |
| Qualité | v1 et v2 donnent la même erreur finale que le séquentiel, sauf dans quelques cas avec N = 500 ; une expérience de contrôle montre que l'écart vient du remplacement par génération, pas du GPU |
| jDE | meilleur que le séquentiel dans 17 configurations sur 36, pour 10 à 33 % de temps en plus que la v2 |

Le détail, les tableaux et les figures sont dans l'article (`paper/main.tex`).

## Les notebooks, dans l'ordre

Chaque notebook correspond à une étape. Il récupère le dépôt (public, aucun compte nécessaire), compile
le code et affiche les résultats de l'étape. Pour l'ouvrir : cliquer sur le lien, puis
*Exécution → Modifier le type d'exécution → GPU T4*, puis *Exécution → Tout exécuter*.

| # | Notebook | Ce qu'il montre | GPU |
|---|---|---|---|
| 1 | [Etape1_PSO_diagnostic](https://colab.research.google.com/github/amenacer/DE_CUDA/blob/main/notebooks/Etape1_PSO_diagnostic.ipynb) | le PSO du cours : exécution et défauts relevés | oui |
| 2 | [Etape1_3_chronometrage](https://colab.research.google.com/github/amenacer/DE_CUDA/blob/main/notebooks/Etape1_3_chronometrage.ipynb) | où part le temps du PSO : seul un tiers est du calcul GPU | oui |
| 3 | [Etape2_interface_commune](https://colab.research.google.com/github/amenacer/DE_CUDA/blob/main/notebooks/Etape2_interface_commune.ipynb) | les fonctions de test communes au CPU et au GPU, et leurs tests | oui |
| 4 | [Etape2_resultats_sde](https://colab.research.google.com/github/amenacer/DE_CUDA/blob/main/notebooks/Etape2_resultats_sde.ipynb) | le DE séquentiel : validation, convergence, qualité | non |
| 5 | [Etape3_DE_CUDA_v1](https://colab.research.google.com/github/amenacer/DE_CUDA/blob/main/notebooks/Etape3_DE_CUDA_v1.ipynb) | la version 1 : contrôles (erreurs CUDA, reproductibilité, temps) | oui |
| 6 | [Etape3_6_validation_speedup](https://colab.research.google.com/github/amenacer/DE_CUDA/blob/main/notebooks/Etape3_6_validation_speedup.ipynb) | la version 1 contre le séquentiel : même qualité, premier speedup | oui |
| 7 | [Etape4_DE_CUDA_v2](https://colab.research.google.com/github/amenacer/DE_CUDA/blob/main/notebooks/Etape4_DE_CUDA_v2.ipynb) | la version 2 : occupation, qualité, temps, choix de la taille de bloc | oui |
| 8 | [Etape7_variantes_DE_GPU](https://colab.research.google.com/github/amenacer/DE_CUDA/blob/main/notebooks/Etape7_variantes_DE_GPU.ipynb) | variantes : choix de CR, current-to-best/1, jDE | oui |

Les sorties obtenues lors de nos exécutions (GPU Tesla T4) sont enregistrées dans `results/`
(`etape2_sde_resultats.md`, `etape3_v1_controles.md`, `etape3_validation_speedup.md`,
`etape4_v2_resultats.md`, `etape7_variantes_gpu.md`).

## La campagne et l'analyse (scripts)

La campagne complète et l'analyse statistique de l'article se lancent par deux scripts
(dans Colab, après avoir récupéré le dépôt ; environ 15 minutes pour la campagne) :

```bash
python3 scripts/run_all.py --jde --out results/campagne.csv     # 1440 runs : sde, cuda_v1, cuda_v2, gpu_jde
python3 scripts/analyse.py --results results/campagne.csv --reference sde \
        --outdir results/analyse --holm-scope global             # tests, tableaux, figures
```

Nos 1440 runs sont déjà dans `results/campagne.csv` : l'analyse peut être refaite sans relancer la campagne.
Le rapport d'analyse est `results/analyse/RAPPORT.md`.

Autres scripts :

| Script | Rôle |
|---|---|
| `scripts/table_erreurs_n100.py [N]` | tableaux d'erreur finale de l'article (N = 50, 100, 500) |
| `scripts/figures_article.py` | figures de l'article |
| `scripts/controle_generationnel.py` | expérience de contrôle : DE séquentiel qui remplace comme le GPU |
| `scripts/comparaison_qin.py` | comparaison avec Qin et al. (2012) |
| `scripts/verif_de_numpy.py`, `scripts/verif_sphere_n500_numpy.py` | DE indépendants en NumPy, pour vérifier nos valeurs |
| `scripts/verify_benchmarks_numpy.py`, `scripts/check_sde_sphere10.py` | vérifications des fonctions de test et du DE séquentiel |

## Organisation du dépôt

```
src/
  pso_original/   PSO du cours, non modifié ; verif.cpp montre ses défauts
  pso_timing/     le même PSO, chronométré (seul kernel.cu change)
  common/         fonctions de test communes au CPU et au GPU (benchmarks.h), format CSV, tests
  seq/            DE séquentiel (sde.cpp)
  cuda/           DE GPU : de_cuda_v1.cu, de_cuda_v2.cu, de_cuda_variantes.cu (jDE, current-to-best/1)
notebooks/        les 8 notebooks ci-dessus
scripts/          campagne, analyse, vérifications
results/          résultats bruts (CSV), sorties des notebooks, analyse (results/analyse/)
figures/          figures et diagrammes UML de l'article
docs/             correspondance PSO → DE, interface commune, diagramme de séquence du PSO,
                  réponses aux questions sur le PSO
paper/            article (LaTeX, format IEEE)
```

Tous les programmes s'utilisent de la même façon et écrivent une ligne par run dans le même format CSV :

```
programme <fonction> <D> <N> <graine> [fichier.csv]
algo,func,D,N,seed,best_error,time_s,FE
```

## Matériel

GPU NVIDIA Tesla T4 (Google Colab), CUDA 13.0, g++ 13.3, compilation `nvcc -O3 -arch=sm_75`.
Le DE séquentiel utilise un seul cœur du CPU. Détails : `results/setup.txt`.
