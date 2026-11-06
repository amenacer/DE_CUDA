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
| Vitesse | la version 2 va jusqu'à **58 fois** plus vite que le DE séquentiel en moyenne sur les 4 fonctions (D = 100, N = 500), 74 fois sur Rastrigin ; la version 1 au plus 9,5 fois |
| Qualité | v1 et v2 donnent la même erreur finale que le séquentiel, sauf dans quelques cas avec N = 500 ; une expérience de contrôle montre que l'écart vient du remplacement par génération, pas du GPU |
| jDE | meilleur que le séquentiel dans 17 configurations sur 36, pour 10 à 33 % de temps en plus que la v2 |

Le détail, les tableaux et les figures sont dans l'article (`paper/`).

## Les notebooks, dans l'ordre

Chaque notebook correspond à une étape : il compile le code de l'étape, l'exécute et affiche les résultats.

**Les 8 notebooks ont déjà été exécutés** (Google Colab, GPU Tesla T4, 27 octobre 2026) : les résultats, tableaux
et figures s'affichent directement à l'ouverture du fichier (Colab, Jupyter, VS Code), sans rien relancer.

**Pour les relancer sur Google Colab :**

1. ouvrir <https://colab.research.google.com>, *Fichier → Importer un notebook*, choisir un fichier de `notebooks/` ;
2. *Exécution → Modifier le type d'exécution → GPU T4* ;
3. (facultatif) dans le panneau *Fichiers* à gauche, importer `DE_CUDA.zip` (l'archive de ce dossier) :
   le notebook utilisera alors exactement le code de ce dossier ; sinon il récupère le code sur notre dépôt
   GitHub public, sans compte ni mot de passe ;
4. *Exécution → Tout exécuter*.

| # | Notebook | Ce qu'il montre | GPU |
|---|---|---|---|
| 1 | `01_PSO_diagnostic` | le PSO du cours : exécution et défauts relevés | oui |
| 2 | `02_PSO_chronometrage` | où part le temps du PSO : seul un tiers est du calcul GPU | oui |
| 3 | `03_interface_commune` | les fonctions de test communes au CPU et au GPU, et leurs tests | oui |
| 4 | `04_DE_sequentiel` | le DE séquentiel : validation, convergence, qualité | non |
| 5 | `05_DE_CUDA_v1` | la version 1 : contrôles (erreurs CUDA, reproductibilité, temps) | oui |
| 6 | `06_v1_validation_speedup` | la version 1 contre le séquentiel : même qualité, premier speedup | oui |
| 7 | `07_DE_CUDA_v2` | la version 2 : occupation, qualité, temps, choix de la taille de bloc | oui |
| 8 | `08_variantes_DE_GPU` | variantes : choix de CR, current-to-best/1, jDE | oui |

Les sorties obtenues lors de nos exécutions (GPU Tesla T4) sont enregistrées dans `results/`
(`etape2_sde_resultats.md`, `etape3_v1_controles.md`, `etape3_validation_speedup.md`,
`etape4_v2_resultats.md`, `etape7_variantes_gpu.md`).

## La campagne et l'analyse (scripts)

La campagne complète et l'analyse statistique de l'article se lancent par deux scripts
(depuis le dossier `DE_CUDA`, sur une machine avec GPU NVIDIA, par exemple Colab ; environ 15 minutes pour la campagne) :

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

## Contenu du dossier

```
DE_CUDA/
  README.md       ce fichier
  notebooks/      les 8 notebooks ci-dessus
  src/
    pso_original/   PSO du cours, non modifié ; verif.cpp montre ses défauts
    pso_timing/     le même PSO, chronométré (seul kernel.cu change)
    common/         fonctions de test communes au CPU et au GPU (benchmarks.h), format CSV, tests
    seq/            DE séquentiel (sde.cpp)
    cuda/           DE GPU : de_cuda_v1.cu, de_cuda_v2.cu, de_cuda_variantes.cu (jDE, current-to-best/1)
  scripts/        campagne, analyse, vérifications
  results/        résultats bruts (CSV), sorties des notebooks, analyse (results/analyse/)
  figures/        figures et diagrammes UML de l'article
  docs/           correspondance PSO → DE, interface commune, diagramme de séquence du PSO,
                  réponses aux questions sur le PSO, journal du projet étape par étape (JOURNAL.md)
  paper/          article : article_DE_CUDA.pdf et sa source LaTeX main.tex (format IEEE ; figures dans figures/)
  presentation/   diapositives de la soutenance
```

Tous les programmes s'utilisent de la même façon et écrivent une ligne par run dans le même format CSV :

```
programme <fonction> <D> <N> <graine> [fichier.csv]
algo,func,D,N,seed,best_error,time_s,FE
```

## Matériel

GPU NVIDIA Tesla T4 (Google Colab), CUDA 13.0, g++ 13.3, compilation `nvcc -O3 -arch=sm_75`.
Le DE séquentiel utilise un seul cœur du CPU. Détails : `results/setup.txt`.
