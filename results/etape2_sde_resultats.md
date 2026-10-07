# Étape 2 — résultats du DE séquentiel (issues #6, #7, #8)

Sortie du notebook `notebooks/Etape2_resultats_sde.ipynb` (lecture seule, aucun push), exécuté le 7 octobre 2026 sur Colab, CPU Intel Xeon @ 2.20GHz. Code : `src/seq/sde.cpp` (DE/rand/1/bin, F = 0,5, CR = 0,3, budget 10⁴ × D évaluations). Figure : `figures/convergence_sde_D30.png`.

## Vérifications

- **Fonctions de `benchmarks.h` contre NumPy (#6)** : 20 points (4 fonctions × 5), écart relatif maximal 3,9e-16 (tolérance 1e-12). `=== TOUS LES TESTS PASSENT ===`.
- **Validation (#8)** : 10 runs Sphere D = 10, N = 50 → erreur 0 à chaque run, FE = 100 000, `sante=OK` (données : `results/sde_sphere10_check.csv`).
- **Reproductibilité** : Rosenbrock D = 10, N = 50, graine 7 lancée deux fois → 8,610298119e-01 les deux fois ; graine 8 → 2,677083333e+00.

## Un run par fonction (D = 10, N = 50, graine 1)

| fonction | best_error | time_s |
|---|---|---|
| sphere | 0 | 0,026 |
| rosenbrock | 3,93e-02 | 0,026 |
| griewank | 0 | 0,042 |
| rastrigin | 0 | 0,037 |

## Qualité et temps (N = 100, 5 graines)

| fonction | D | erreur moyenne | écart-type | médiane | temps moyen (s) | FE |
|---|---|---|---|---|---|---|
| sphere | 10 | 0 | 0 | 0 | 0,0256 | 100 000 |
| sphere | 50 | 0 | 0 | 0 | 0,528 | 500 000 |
| sphere | 100 | 0 | 0 | 0 | 2,246 | 1 000 000 |
| rosenbrock | 10 | 1,493 | 1,010 | 0,875 | 0,0266 | 100 000 |
| rosenbrock | 50 | 40,88 | 0,447 | 40,99 | 0,550 | 500 000 |
| rosenbrock | 100 | 89,74 | 0,170 | 89,80 | 2,713 | 1 000 000 |
| griewank | 10 | 0 | 0 | 0 | 0,0464 | 100 000 |
| griewank | 50 | 0 | 0 | 0 | 1,097 | 500 000 |
| griewank | 100 | 0 | 0 | 0 | 3,709 | 1 000 000 |
| rastrigin | 10 | 0 | 0 | 0 | 0,0397 | 100 000 |
| rastrigin | 50 | 169,3 | 7,45 | 170,7 | 1,308 | 500 000 |
| rastrigin | 100 | 536,4 | 11,0 | 537,2 | 5,403 | 1 000 000 |

Écart-type : `pandas.std` (échantillon, n − 1).

## Lecture des résultats

- **Budget respecté** : FE = 10⁴ × D exactement dans tous les runs, tous `sante=OK`.
- **Convergence (figure, D = 30, N = 50)** : Sphere et Griewank convergent exponentiellement (droite en échelle log) et atteignent 0 en double précision vers 74 000 et 62 000 FE (le palier à 1e-20 est l'affichage de 0). Rosenbrock trouve vite la vallée puis progresse très lentement ; Rastrigin stagne dans un minimum local. La meilleure erreur ne remonte jamais (sélection `<=`).
- **Effet de la dimension** : Rastrigin passe de 0 (D = 10) à 169 (D = 50) puis 536 (D = 100). Sur Rosenbrock, l'écart-type très faible en D = 50 et 100 indique une progression lente et régulière plutôt qu'un piège aléatoire.
- **Temps en D²** : × 20 environ de D = 10 à 50, × 4 environ de 50 à 100 (D fois plus d'évaluations, chacune D fois plus coûteuse). Griewank et Rastrigin sont plus lents (`cos`, `sqrt`). C'est ce coût que le DE GPU doit réduire.
- Cohérence avec le GPU v1 (`results/etape4_v2_resultats.md`, Rosenbrock D = 50, N = 50) : 39,2 ± 1,3, du même ordre que 40,9 ± 0,45 ici (N = 100, donc pas directement comparable).
