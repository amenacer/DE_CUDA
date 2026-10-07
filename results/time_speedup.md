### Temps d'exécution et speedup

| Dim | Pop | DE_seq (s) | DE_gpu_v1 (s) [speedup] | DE_gpu_v2 (s) [speedup] | DE_gpu_jDE (s) [speedup] |
|---|---|---|---|---|---|
| 10 | 50 | 0.020 | 0.189 [x0.1] | 0.083 [x0.2] | 0.082 [x0.2] |
| 10 | 100 | 0.020 | 0.118 [x0.2] | 0.066 [x0.3] | 0.066 [x0.3] |
| 10 | 500 | 0.020 | 0.064 [x0.3] | 0.053 [x0.4] | 0.053 [x0.4] |
| 50 | 50 | 0.498 | 0.734 [x0.7] | 0.212 [x2.3] | 0.213 [x2.3] |
| 50 | 100 | 0.500 | 0.387 [x1.3] | 0.129 [x3.9] | 0.129 [x3.9] |
| 50 | 500 | 0.502 | 0.118 [x4.2] | 0.066 [x7.6] | 0.067 [x7.5] |
| 100 | 50 | 2.016 | 1.415 [x1.4] | 0.374 [x5.4] | 0.372 [x5.4] |
| 100 | 100 | 1.993 | 0.741 [x2.7] | 0.212 [x9.4] | 0.213 [x9.4] |
| 100 | 500 | 2.012 | 0.189 [x10.6] | 0.084 [x23.9] | 0.085 [x23.7] |

Temps moyens (s) sur toutes les fonctions et tous les runs. Speedup = T_DE_seq / T_algo (un speedup < 1 signifie que l'algorithme est plus lent que la référence).
