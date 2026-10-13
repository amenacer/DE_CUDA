# Étapes 6.1 à 6.6 — Analyse de la campagne complète (issues #15, #16, #21)

Données : `results/campagne.csv` (étape 5, 1440 runs sur Tesla T4 : 4 algorithmes × 4 fonctions × D = 10/50/100 × N = 50/100/500 × 10 graines, budget 10⁴·D évaluations, F = 0,5, CR = 0,3 sauf jDE).

Algorithmes : `sde` (DE séquentiel C++, un cœur CPU), `cuda_v1` (un thread par individu), `cuda_v2` (un bloc par individu, un thread par gène), `gpu_jde` (jDE sur l'organisation de la v2).

Reproduire :

```
python3 scripts/analyse.py --results results/campagne.csv --reference sde --outdir results/analyse --all-configs
python3 scripts/comparaison_qin.py --results results/campagne.csv --outdir results/analyse
python3 scripts/verif_de_numpy.py
```

| Fichier | Contenu |
|---|---|
| `tables/quality_D10/50/100.{md,tex,csv}` | erreur moyenne ± écart-type (ddof = 1), meilleur en gras, test contre `sde` |
| `tables/time_speedup.{md,tex,csv}` | temps moyen (toutes fonctions confondues) et speedup par (D, N) |
| `tables/speedup_par_fonction.{md,csv}` | temps et speedup pour chaque (fonction, D, N) |
| `tables/stats_pairwise.csv`, `stats_kruskal.csv`, `stats_summary.md` | Mann-Whitney contre `sde` avec correction de Holm ; Kruskal-Wallis à 4 algorithmes |
| `tables/comparaison_qin.md` | Kruskal-Wallis entre les 3 implémentations du même DE ; comparaison avec cudaDEi |
| `figures/speedup`, `time_vs_pop`, `time_vs_dim`, `boxplot_D*_P*` | figures PNG (200 dpi) et PDF |

Le script `scripts/analyse.py` est celui écrit par Lisa pour l'issue #15 (testé sur un faux CSV) ; il lit maintenant directement le format commun du projet. Les tableaux et figures produits auparavant à partir du faux CSV ont été retirés du dépôt.

## 1. Temps et speedup

Speedup = temps moyen de `sde` / temps moyen de l'algorithme (10 runs par configuration).

| D | N | v1 | v2 | jDE |
|---|---|---|---|---|
| 10 | 50 | ×0,8 | ×1,9 | ×1,6 |
| 10 | 500 | ×7,9 | ×12,7 | ×10,8 |
| 50 | 100 | ×1,6 | ×14,7 | ×11,1 |
| 100 | 100 | ×1,6 | ×24,2 | ×18,3 |
| 100 | 500 | ×7,2 | ×57,5 | ×50,8 |

- La v2 est la plus rapide dans les 36 configurations. Le speedup croît avec D et avec N : de ×1,5 (Rosenbrock, D = 10, N = 50) à **×74,5 (Rastrigin, D = 100, N = 500)**.
- La v1 n'est vraiment utile qu'avec N = 500 (×6,0 à ×9,5) ; avec N = 50 elle va de ×0,7 (Griewank, plus lente que le CPU) à ×1,5 (Rastrigin). Avec 50 ou 100 individus elle n'occupe qu'un seul SM des 40 du T4.
- jDE coûte 10 à 33 % de temps en plus que la v2 (tirage et stockage de F et CR par individu) et reste ×1,3 à ×68 plus rapide que le séquentiel.
- Ces chiffres confirment les mesures des étapes 3 et 4 (Rastrigin D = 100, N = 100 : v2 0,126 s ici, 0,125 s à l'étape 4).

## 2. Qualité des solutions

Test de Mann-Whitney (Wilcoxon rank-sum) bilatéral de chaque algorithme contre `sde`, correction de Holm par configuration, seuil 5 %. Les runs de deux algorithmes ne sont pas appariés (même graine, mais générateurs aléatoires différents) : le test de Wilcoxon signé ne s'applique pas.

| Algorithme | meilleur que `sde` | pas de différence | pire que `sde` |
|---|---|---|---|
| cuda_v1 | 0 | 29 | 7 |
| cuda_v2 | 1 | 27 | 8 |
| gpu_jde | 19 | 15 | 2 |

**Les trois implémentations du même DE** (`sde`, `cuda_v1`, `cuda_v2`) : Kruskal-Wallis à 3 groupes, différence détectée dans 9 configurations sur 36 (`comparaison_qin.md`, partie 1). Ces différences sont concentrées sur **N = 500** (6 des 9) : Sphere et Griewank, où le séquentiel atteint une erreur environ deux fois plus faible (par exemple Sphere D = 50 : 1,1e-4 contre 2,3e-4). Explication probable : le séquentiel remplace un individu dès que l'essai est meilleur (variante asynchrone) ; les nouveaux individus servent aussitôt aux mutations suivantes. Les versions GPU remplacent toute la population en fin de génération. Avec N = 500, le budget ne laisse que 200 générations (10⁴·D / N) et l'information se propage plus vite en asynchrone. Avec N = 50 ou 100 (24 configurations), aucune différence n'est détectée dans 21 cas ; les 3 autres sont Rosenbrock D = 50 N = 50 (séquentiel légèrement meilleur, 37,9 contre 39,5), Rastrigin D = 50 N = 50 (GPU légèrement meilleur, 139 contre 151) et Sphere D = 100 N = 100 (erreurs de l'ordre de 1e-29, sans portée pratique).

**jDE** est significativement meilleur que `sde` dans 19 configurations sur 36 et pire dans 2 (Sphere et Griewank D = 100, N = 50, à des erreurs ≤ 0,04). Il résout Rastrigin D = 50 et D = 100 avec N = 100 (erreur 0), là où le DE de base reste vers 165 et 535.

**Kruskal-Wallis à 4 algorithmes** : différence dans 24 configurations sur 36 ; dans les 12 autres, tous les algorithmes atteignent 0 ou sont indiscernables.

## 3. Comparaison avec cudaDEi (Qin et al., GECCO 2012)

Même protocole que notre projet (DE/rand/1/bin, F = 0,5, CR = 0,3, 10⁴·D évaluations, mêmes fonctions et tailles), 25 runs, GPU GTX 560, CUDA 4.0. Tableau complet : `tables/comparaison_qin.md`.

- **Temps** : notre v2 est 59 à 365 fois plus rapide que les temps publiés de cudaDEi (Rastrigin D = 100, NP = 100 : 17,4 s contre 0,126 s). **Ce rapport ne mesure pas la qualité de l'implémentation** : les GPU ont sept ans d'écart (GTX 560, 336 cœurs ; T4, 2560 cœurs), l'article ne précise ni la précision flottante ni ce que couvre le chronométrage. On peut seulement dire que nos temps sont du même ordre de grandeur que ceux attendus pour un GPU récent, pas plus.
- **Qualité** : identique sur Sphere, Griewank et Rastrigin en D = 10 (N = 50/100) et sur Sphere et Griewank en D = 50/100 (N = 50/100). **Mais cudaDEi publie de bien meilleures erreurs sur Rastrigin en D = 50/100** (2,9 contre 165 en D = 50, NP = 100) **et pour N = 500** (Sphere à 0 contre 1e-5 à 1e-1).
- **Vérification indépendante** (`scripts/verif_de_numpy.py`) : un DE/rand/1/bin réécrit en NumPy, indépendamment de notre code, donne 147 à 174 sur Rastrigin D = 50, NP = 100, quelle que soit la gestion des bornes ; même avec 10 fois plus d'évaluations il reste à 80. Nos résultats sont donc ceux attendus de ce DE avec ces paramètres. Nous n'avons pas pu identifier ce qui, dans cudaDEi, produit 2,9 (l'article ne décrit ni la gestion des bornes, ni l'initialisation, ni le domaine de recherche exact) : **l'écart est signalé, pas expliqué**.

## Limites

- 10 runs par configuration (25 dans l'article de référence) ; un test non significatif ne prouve pas l'égalité.
- Un seul GPU (T4 sur Colab) ; le séquentiel tourne sur un cœur du Xeon de Colab, sans vectorisation particulière.
- Les erreurs nulles sont tracées à 1e-12 sur les boîtes à moustaches (échelle logarithmique).
- Les courbes de convergence de la campagne n'existent pas (le CSV ne contient que l'erreur finale) ; celles du séquentiel sont dans `figures/convergence_sde_D30.png` (étape 2) et celles des variantes dans le notebook de l'étape 7.
