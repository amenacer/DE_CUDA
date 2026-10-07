### Qualité des solutions, Dim = 10

| Fonction | Pop | DE_seq | DE_gpu_v1 | DE_gpu_v2 | DE_gpu_jDE |
|---|---|---|---|---|---|
| Shifted Sphere | 50 | 2.04e-06 ± 1.48e-06 | 9.46e-07 ± 1.06e-06 (=) | 1.39e-06 ± 1.15e-06 (=) | **3.91e-07 ± 6.59e-07 (+)** |
| Shifted Sphere | 100 | 1.61e-06 ± 1.34e-06 | 2.30e-06 ± 2.20e-06 (=) | 1.86e-06 ± 1.87e-06 (=) | **8.64e-07 ± 5.22e-07 (=)** |
| Shifted Sphere | 500 | 3.07e-06 ± 1.47e-06 | 8.98e-06 ± 8.83e-06 (=) | 5.49e-06 ± 3.17e-06 (=) | **1.26e-06 ± 1.88e-06 (+)** |
| Shifted Rastrigin | 50 | 1.29e+01 ± 7.88e+00 | 6.64e+00 ± 2.53e+00 (=) | 5.83e+00 ± 4.21e+00 (+) | **2.33e+00 ± 2.41e+00 (+)** |
| Shifted Rastrigin | 100 | 7.99e+00 ± 4.92e+00 | 9.22e+00 ± 8.04e+00 (=) | 4.11e+00 ± 1.99e+00 (=) | **1.31e+00 ± 7.49e-01 (+)** |
| Shifted Rastrigin | 500 | 4.02e+00 ± 2.29e+00 | 4.56e+00 ± 5.00e+00 (=) | 3.77e+00 ± 2.40e+00 (=) | **1.40e+00 ± 9.22e-01 (+)** |
| Shifted Rosenbrock | 50 | 3.80e+01 ± 4.35e+01 | 4.90e+01 ± 4.00e+01 (=) | 2.86e+01 ± 2.33e+01 (=) | **1.20e+01 ± 9.02e+00 (=)** |
| Shifted Rosenbrock | 100 | 4.95e+01 ± 4.32e+01 | 7.96e+01 ± 8.07e+01 (=) | 4.28e+01 ± 4.02e+01 (=) | **2.44e+01 ± 2.54e+01 (=)** |
| Shifted Rosenbrock | 500 | 1.10e+02 ± 1.04e+02 | 3.92e+02 ± 5.94e+02 (=) | 1.47e+02 ± 1.04e+02 (=) | **2.58e+01 ± 9.13e+00 (+)** |
| Shifted Griewank | 50 | 1.52e-02 ± 1.70e-02 | 1.83e-02 ± 2.09e-02 (=) | 1.12e-02 ± 6.22e-03 (=) | **2.07e-03 ± 1.35e-03 (+)** |
| Shifted Griewank | 100 | 1.76e-02 ± 1.02e-02 | 1.19e-02 ± 7.87e-03 (=) | 8.18e-03 ± 4.34e-03 (=) | **3.60e-03 ± 2.45e-03 (+)** |
| Shifted Griewank | 500 | 2.43e-02 ± 1.25e-02 | 2.48e-02 ± 2.18e-02 (=) | 2.83e-02 ± 1.73e-02 (=) | **9.10e-03 ± 5.04e-03 (+)** |

Erreur = f(x) - f*, moyenne ± écart-type (ddof=1) sur les runs. Gras = meilleure moyenne. Symbole (test contre DE_seq) : + significativement meilleur, - significativement pire, = pas de différence significative.
