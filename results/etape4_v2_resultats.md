# Étape 4 — DE CUDA v2 : occupation, qualité, temps (issues #12 et #13)

GPU : Tesla T4 (Google Colab), CPU : Intel(R) Xeon(R) CPU @ 2.00GHz, CUDA 12 (nvcc -O3 -arch=sm_75)

Sortie du notebook `notebooks/Etape4_DE_CUDA_v2.ipynb`. Figure : `figures/temps_v1_v2_N.png`.

```
de_cuda_v1.cu : compilation SANS aucun warning
de_cuda_v2.cu : compilation SANS aucun warning
GPU : Tesla T4, 40 SM, 1024 threads max par SM, 49152 octets de memoire partagee par bloc
Kernel k_MCER, D = 10 (choix automatique : 32 threads si N < 160, 32 threads sinon)
Kernel k_MCER, D = 50 (choix automatique : 64 threads si N < 160, 32 threads sinon)
Kernel k_MCER, D = 100 (choix automatique : 64 threads si N < 160, 32 threads sinon)
| D | threads/bloc | shmem (octets) | blocs/SM | occupation (%) | threads utiles (%) |
|---|---|---|---|---|---|
| 10 | 32 | 592 | 16 | 50.0 | 31.2 |
| 10 | 64 | 1104 | 16 | 100.0 | 15.6 |
| 10 | 128 | 2128 | 8 | 100.0 | 7.8 |
| 10 | 256 | 4176 | 4 | 100.0 | 3.9 |
| 10 | 512 | 8272 | 2 | 100.0 | 2.0 |
| 50 | 32 | 912 | 16 | 50.0 | 100.0 |
| 50 | 64 | 1424 | 16 | 100.0 | 78.1 |
| 50 | 128 | 2448 | 8 | 100.0 | 39.1 |
| 50 | 256 | 4496 | 4 | 100.0 | 19.5 |
| 50 | 512 | 8592 | 2 | 100.0 | 9.8 |
| 100 | 32 | 1312 | 16 | 50.0 | 100.0 |
| 100 | 64 | 1824 | 16 | 100.0 | 100.0 |
| 100 | 128 | 2848 | 8 | 100.0 | 78.1 |
| 100 | 256 | 4896 | 4 | 100.0 | 39.1 |
| 100 | 512 | 8992 | 2 | 100.0 | 19.5 |
| fonction | D | threads/bloc | best_error | FE | temps (s) | sante |
|---|---|---|---|---|---|---|
| sphere | 10 | 32 | 0.000e+00 | 100000 | 0.0088 | OK |
| sphere | 50 | 64 | 0.000e+00 | 500000 | 0.0487 | OK |
| sphere | 100 | 64 | 4.930e-32 | 1000000 | 0.1062 | OK |
| rosenbrock | 10 | 32 | 6.259e-01 | 100000 | 0.0092 | OK |
| rosenbrock | 50 | 64 | 4.114e+01 | 500000 | 0.0462 | OK |
| rosenbrock | 100 | 64 | 8.978e+01 | 1000000 | 0.1088 | OK |
| griewank | 10 | 32 | 0.000e+00 | 100000 | 0.0097 | OK |
| griewank | 50 | 64 | 0.000e+00 | 500000 | 0.0583 | OK |
| griewank | 100 | 64 | 0.000e+00 | 1000000 | 0.1570 | OK |
| rastrigin | 10 | 32 | 0.000e+00 | 100000 | 0.0088 | OK |
| rastrigin | 50 | 64 | 1.536e+02 | 500000 | 0.0505 | OK |
| rastrigin | 100 | 64 | 5.356e+02 | 1000000 | 0.1247 | OK |
Determinisme v2 (rastrigin D=50 N=100 seed 3) : 1.631731634000000e+02 / 1.631731634000000e+02 -> IDENTIQUES
Qualite v1 contre v2 (D = 10, N = 50, 10 runs, moyenne ± ecart-type de l'erreur)
| fonction | v1 | v2 | p (Mann-Whitney) | conclusion (p > 0,05) |
|---|---|---|---|---|
| sphere | 0.000e+00 ± 0.00e+00 | 0.000e+00 ± 0.00e+00 | 1.000 | meme qualite |
| rosenbrock | 7.206e-01 ± 1.05e+00 | 4.633e-01 ± 5.09e-01 | 0.623 | meme qualite |
| griewank | 0.000e+00 ± 0.00e+00 | 0.000e+00 ± 0.00e+00 | 1.000 | meme qualite |
| rastrigin | 0.000e+00 ± 0.00e+00 | 0.000e+00 ± 0.00e+00 | 1.000 | meme qualite |
Temps moyen (s) de la v2 selon le nombre de threads par bloc (3 runs)
| fonction | D | N | T=32 | T=64 | T=128 | T=256 | meilleur T | auto (s) |
|---|---|---|---|---|---|---|---|---|
| rastrigin | 50 | 50 | 0.1039 | 0.0913 | 0.1003 | 0.1262 | 64 | 0.0912 |
| rastrigin | 50 | 100 | 0.0535 | 0.0504 | 0.0587 | 0.0778 | 64 | 0.0505 |
| rastrigin | 50 | 500 | 0.0205 | 0.0222 | 0.0383 | 0.0818 | 32 | 0.0202 |
| rastrigin | 100 | 50 | 0.2741 | 0.2194 | 0.2240 | 0.2708 | 64 | 0.2198 |
| rastrigin | 100 | 100 | 0.1421 | 0.1246 | 0.1376 | 0.1716 | 64 | 0.1253 |
| rastrigin | 100 | 500 | 0.0599 | 0.0642 | 0.1026 | 0.1704 | 32 | 0.0599 |
| sphere | 50 | 50 | 0.0875 | 0.0809 | 0.0904 | 0.1180 | 64 | 0.0813 |
| sphere | 50 | 100 | 0.0449 | 0.0437 | 0.0517 | 0.0735 | 64 | 0.0437 |
| sphere | 50 | 500 | 0.0141 | 0.0179 | 0.0349 | 0.0811 | 32 | 0.0141 |
| sphere | 100 | 50 | 0.2160 | 0.1801 | 0.1859 | 0.2392 | 64 | 0.1801 |
| sphere | 100 | 100 | 0.1102 | 0.0961 | 0.1111 | 0.1504 | 64 | 0.0962 |
| sphere | 100 | 500 | 0.0336 | 0.0407 | 0.0899 | 0.1658 | 32 | 0.0336 |
Temps moyen (s), Rastrigin, 3 runs
| D | N | v1 (s) | v2 (s) | gain v1/v2 |
|---|---|---|---|---|
| 10 | 50 | 0.0433 | 0.0171 | 2.53 |
| 10 | 100 | 0.0259 | 0.0088 | 2.95 |
| 10 | 500 | 0.0054 | 0.0031 | 1.72 |
| 50 | 50 | 0.7573 | 0.0910 | 8.32 |
| 50 | 100 | 0.4796 | 0.0504 | 9.51 |
| 50 | 500 | 0.1014 | 0.0201 | 5.05 |
| 100 | 50 | 2.7749 | 0.2194 | 12.65 |
| 100 | 100 | 1.8271 | 0.1249 | 14.63 |
| 100 | 500 | 0.3864 | 0.0597 | 6.47 |
```

## Complément : qualité v1 contre v2 en D = 50

Exécuté après le notebook, sur la même session Colab (cellule « 4 bis » du notebook). En D = 10, Sphere, Griewank et Rastrigin sont résolues exactement par les deux versions, donc le test précédent ne renseigne que sur Rosenbrock.

```
Qualite v1 contre v2 (D = 50, N = 50, 10 runs)
| fonction | v1 moyenne ± ET | v1 mediane | v2 moyenne ± ET | v2 mediane | p (Mann-Whitney) | conclusion (5 %) |
|---|---|---|---|---|---|---|
| rosenbrock | 3.920e+01 ± 1.26e+00 | 3.973e+01 | 3.951e+01 ± 1.29e+00 | 3.997e+01 | 0.345 | pas de difference |
| rastrigin | 1.456e+02 ± 6.56e+00 | 1.459e+02 | 1.388e+02 ± 7.42e+00 | 1.412e+02 | 0.104 | pas de difference |
```

Données brutes : `results/qualite_v1_vs_v2_D50.csv`.

Note sur `results/taille_bloc_v2.csv` : lors de ce run, la cellule du balayage écrivait le CSV sans colonne `threads` ; la colonne a été reconstituée à partir de l'ordre de la boucle (fonction, D, N, T, graine), qui est fixe. La cellule du notebook écrit désormais cette colonne elle-même. Les temps de la colonne « auto » ne sont que dans le tableau ci-dessus (moyennes de 3 runs).

## Lecture des résultats

- **Justesse.** Chaque run de la v2 vérifie lui-même (hors chronomètre) que l'argmin calculé sur GPU est égal au minimum calculé sur CPU, et que la valeur de f obtenue par réduction parallèle est égale à `evaluate()` sur CPU (écart relatif < 1e-9). Tous les runs finissent avec `sante=OK`.
- **Qualité.** Aucune différence significative détectée entre v1 et v2 : Rosenbrock D = 10 (p = 0,623), Rosenbrock D = 50 (p = 0,345), Rastrigin D = 50 (p = 0,104) ; en D = 10, Sphere, Griewank et Rastrigin sont résolues exactement par les deux versions. Avec 10 runs, « pas de différence détectée » ne prouve pas l'équivalence ; la campagne complète (10 runs × toutes les configurations) le confirmera ou non.
- **Temps.** La v2 est de × 1,7 à × 14,6 plus rapide que la v1. Le gain grandit avec D (× 12,7 à × 14,6 en D = 100) ; à D fixé, il est le plus fort pour N = 100. Attention, la v1 n'a pas été optimisée : avec 128 threads par bloc, N = 50 ou 100 individus tiennent dans **un seul bloc**, donc sur un seul des 40 SM du T4 ; chaque thread boucle sur D gènes avec des accès mémoire espacés de D cases (non coalescés). La v2 répartit les gènes entre les threads d'un bloc et lance un bloc par individu.
- **Taille de bloc.** L'occupation théorique ne suffit pas à choisir : 128 ou 256 threads donnent 100 % d'occupation, mais sont toujours plus lents que le meilleur choix entre 32 et 64. La règle automatique (32 si D ≤ 32 ou N ≥ 4 × SM, sinon 64) reproduit le meilleur T dans les 12 cas mesurés, mais elle a été ajustée sur ces mêmes cas : le seuil 160 est seulement situé entre N = 100 et N = 500, et la branche D ≤ 32 n'a pas été balayée.
- **Ordre de grandeur face au séquentiel** (Rastrigin, N = 100 ; temps du séquentiel dans `results/speedup_v1.csv`, mesurés dans la même session Colab que la v2) : D = 10 : 0,0354 s contre 0,0088 s (≈ × 4) ; D = 50 : 1,117 s contre 0,0504 s (≈ × 22) ; D = 100 : 4,126 s contre 0,1249 s (≈ × 33). Ce sont des ordres de grandeur : le séquentiel est mono-cœur, compilé en `-O2`, et ses temps varient de 10 à 17 % d'un run à l'autre sur Colab. Le speedup définitif viendra de la campagne de l'étape 5.
- Les « ± » sont des écarts-types de population (`statistics.pstdev`).
