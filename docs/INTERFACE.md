# Interface commune DE_CUDA (issue #5)

Contrat partagé entre la version séquentielle (`src/seq`) et la version GPU (`src/cuda`).
**Ne pas modifier sans prévenir les deux autres membres du groupe.**

## Fichiers

| Fichier | Contenu |
|---|---|
| `src/common/benchmarks.h` | 4 fonctions de test, domaines, décalage `o`, budget |
| `src/common/results_csv.h` | écriture d'une ligne de résultat dans un CSV |
| `src/common/test_benchmarks.cpp` | tests CPU (f(o) = 0, domaines, valeurs de référence, CSV) |
| `src/common/test_gpu.cu` | test GPU : même résultat que le CPU |

## Fonctions (en `double`, renvoient l'erreur f(x) − f(x*), 0 à l'optimum)

| id | nom | domaine | formule (z = x − o) |
|---|---|---|---|
| 0 | sphere | [−100, 100] | Σ z_i² |
| 1 | rosenbrock | [−100, 100] | Σ_{i<D−1} 100 (z_i² − z_{i+1})² + (z_i − 1)², avec z = x − o + 1 |
| 2 | griewank | [−600, 600] | Σ z_i²/4000 − Π cos(z_i/√(i+1)) + 1 |
| 3 | rastrigin | [−5, 5] | Σ z_i² − 10 cos(2π z_i) + 10 |

- Appel unique : `evaluate(f, x, o, D)`, marqué `__host__ __device__` (utilisable sur CPU et GPU).
- Biais CEC 2005 **non ajouté** : on travaille directement sur l'erreur (précision, bug 9).
- Décalage : `make_shift(f, D)` tire `o` dans 80 % du domaine avec un générateur déterministe
  (splitmix64, graine 1000·f + D). CPU et GPU utilisent donc exactement le même `o`.
- Budget : `max_fe(D) = 10^4 × D` évaluations, **en comptant les N évaluations de l'initialisation**.

## Ligne de commande (commune aux deux versions)

```
./programme <fonction> <D> <N> <seed> [fichier.csv]
./sde sphere 10 50 1 results/sde.csv
```

## Format CSV (une ligne par run)

```
algo,func,D,N,seed,best_error,time_s,FE
sde,sphere,10,50,1,1.234000e-09,0.500000,100000
```

| colonne | sens |
|---|---|
| algo | `sde`, `cuda_v1`, `cuda_v2`, … |
| func | nom de la fonction |
| D, N, seed | dimension, population, graine du run |
| best_error | meilleure erreur trouvée (notation scientifique) |
| time_s | temps de calcul en secondes (sans l'initialisation du contexte CUDA) |
| FE | nombre d'évaluations réellement faites (≤ 10^4 × D) |
