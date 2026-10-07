# Étape 3.6-3.7 — validation du GPU contre le séquentiel et speedup (issue #11)

GPU : Tesla T4 (Google Colab), CPU : Intel(R) Xeon(R) CPU @ 2.00GHz, CUDA 12 (nvcc -O3 -arch=sm_75)

Sortie du notebook `notebooks/Etape3_6_validation_speedup.ipynb` (séquentiel : `src/seq/sde.cpp` de Lounis, branche `lounis/2.7-2.9-csv-convergence`). Figure : `figures/speedup_v1.png`.

```
de_cuda_v1.cu : compilation SANS aucun warning
CPU : Intel(R) Xeon(R) CPU @ 2.00GHz
Validation D = 10, N = 50, 10 runs (erreur finale f(x) - f(x*))
| fonction | algo | moyenne | ecart-type | mediane | min | max | p (Mann-Whitney) | conclusion |
|---|---|---|---|---|---|---|---|---|
| sphere | sde | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |  |  |
| sphere | cuda_v1 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000 | p > 0,05 : pas de difference |
| rastrigin | sde | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |  |  |
| rastrigin | cuda_v1 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000 | p > 0,05 : pas de difference |
Validation complementaire (N = 50, 10 runs)
| fonction | D | sde moyenne ± ET | sde mediane | GPU v1 moyenne ± ET | GPU v1 mediane | p (Mann-Whitney) | conclusion (5 %) |
|---|---|---|---|---|---|---|---|
| rosenbrock | 10 | 2.033e+00 ± 1.62e+00 | 2.036e+00 | 7.206e-01 ± 1.05e+00 | 1.040e-01 | 0.089 | pas de difference |
| rastrigin | 50 | 1.514e+02 ± 6.55e+00 | 1.516e+02 | 1.456e+02 ± 6.56e+00 | 1.459e+02 | 0.140 | pas de difference |
Speedup du DE GPU v1 (N = 100, temps moyen sur 5 runs, en secondes)
| fonction | D | sequentiel (s) | GPU v1 (s) | speedup |
|---|---|---|---|---|
| sphere | 10 | 0.0213 ± 0.0001 | 0.0169 ± 0.0017 | 1.26 |
| sphere | 50 | 0.4500 ± 0.0087 | 0.2769 ± 0.0480 | 1.63 |
| sphere | 100 | 1.7608 ± 0.0310 | 0.8912 ± 0.0484 | 1.98 |
| rosenbrock | 10 | 0.0218 ± 0.0008 | 0.0180 ± 0.0001 | 1.21 |
| rosenbrock | 50 | 0.4664 ± 0.0050 | 0.3269 ± 0.0222 | 1.43 |
| rosenbrock | 100 | 2.0467 ± 0.2650 | 1.1786 ± 0.0472 | 1.74 |
| griewank | 10 | 0.0389 ± 0.0014 | 0.0360 ± 0.0001 | 1.08 |
| griewank | 50 | 0.7468 ± 0.0041 | 0.7618 ± 0.0226 | 0.98 |
| griewank | 100 | 3.2384 ± 0.3248 | 2.9149 ± 0.0680 | 1.11 |
| rastrigin | 10 | 0.0354 ± 0.0034 | 0.0261 ± 0.0002 | 1.36 |
| rastrigin | 50 | 1.1170 ± 0.1737 | 0.4930 ± 0.0253 | 2.27 |
| rastrigin | 100 | 4.1260 ± 0.4075 | 1.8526 ± 0.0467 | 2.23 |
```

## Lecture des résultats

- **Validation.** En D = 10, Sphere et Rastrigin sont résolues exactement (erreur 0) par les deux programmes : le test ne peut alors rien distinguer (p = 1 par convention). Le complément sur deux cas où l'erreur n'est pas nulle ne montre pas de différence significative au seuil de 5 % : Rastrigin D = 50 (p = 0,140) et Rosenbrock D = 10 (p = 0,089). Ce dernier cas est limite : les médianes valent 2,04 (séquentiel) et 0,10 (GPU), avec seulement 10 runs. Les deux programmes ne sont pourtant pas strictement le même algorithme : le séquentiel remplace x_i dès que l'essai est meilleur, le GPU remplace en fin de génération.
- **Speedup de la v1 : modeste (× 0,98 à × 2,3).** Sur Griewank D = 50, la v1 est même un peu plus lente que le CPU (× 0,98). Avec N = 100 et 128 threads par bloc, la v1 ne lance qu'**un seul bloc**, donc n'utilise qu'un des 40 SM du T4 ; chacun des 100 threads boucle sur les D gènes, en `double`, avec des accès mémoire espacés de D cases (non coalescés). Griewank est le cas le plus défavorable (× 1 environ). Hypothèse, non vérifiée par profilage : chaque gène y demande un `cos`, une `sqrt` et une division en double précision, opérations lentes sur un T4.
- Ces chiffres justifient la v2 (étape 4) : un bloc par individu et un thread par gène.
- Les « ± » sont des écarts-types de population (`statistics.pstdev`).
