### Qualité des solutions, Dim = 100

| Fonction | Pop | DE_seq | DE_gpu_v1 | DE_gpu_v2 | DE_gpu_jDE |
|---|---|---|---|---|---|
| Shifted Sphere | 50 | 1.13e-03 ± 7.55e-04 | 1.19e-03 ± 7.19e-04 (=) | 2.00e-03 ± 2.33e-03 (=) | **2.95e-04 ± 2.35e-04 (+)** |
| Shifted Sphere | 100 | 1.82e-03 ± 1.89e-03 | 2.70e-03 ± 2.03e-03 (=) | 3.62e-03 ± 4.90e-03 (=) | **8.45e-04 ± 1.05e-03 (=)** |
| Shifted Sphere | 500 | 5.58e-03 ± 6.69e-03 | 2.57e-03 ± 7.03e-04 (=) | 5.09e-03 ± 5.02e-03 (=) | **1.44e-03 ± 1.63e-03 (+)** |
| Shifted Rastrigin | 50 | 7.16e+01 ± 5.51e+01 | 1.51e+02 ± 6.26e+01 (-) | 5.49e+01 ± 2.62e+01 (=) | **2.16e+01 ± 1.84e+01 (+)** |
| Shifted Rastrigin | 100 | 4.48e+01 ± 3.39e+01 | 6.91e+01 ± 8.24e+01 (=) | 8.61e+01 ± 1.08e+02 (=) | **9.84e+00 ± 7.39e+00 (+)** |
| Shifted Rastrigin | 500 | 3.54e+01 ± 2.65e+01 | 4.10e+01 ± 3.71e+01 (=) | 6.58e+01 ± 5.46e+01 (=) | **1.18e+01 ± 9.36e+00 (+)** |
| Shifted Rosenbrock | 50 | 3.68e+02 ± 1.89e+02 | 2.42e+02 ± 1.46e+02 (=) | 2.75e+02 ± 2.70e+02 (=) | **8.81e+01 ± 4.49e+01 (+)** |
| Shifted Rosenbrock | 100 | 6.54e+02 ± 5.29e+02 | 4.11e+02 ± 2.86e+02 (=) | 7.72e+02 ± 1.22e+03 (=) | **1.97e+02 ± 1.84e+02 (+)** |
| Shifted Rosenbrock | 500 | 7.62e+02 ± 4.65e+02 | 1.30e+03 ± 7.53e+02 (=) | 1.35e+03 ± 1.21e+03 (=) | **2.39e+02 ± 1.70e+02 (+)** |
| Shifted Griewank | 50 | 9.93e-01 ± 9.52e-01 | 3.44e-01 ± 1.59e-01 (=) | 3.34e-01 ± 1.72e-01 (=) | **8.03e-02 ± 8.12e-02 (+)** |
| Shifted Griewank | 100 | 3.13e-01 ± 2.75e-01 | 3.73e-01 ± 3.76e-01 (=) | 3.59e-01 ± 2.96e-01 (=) | **1.19e-01 ± 1.01e-01 (=)** |
| Shifted Griewank | 500 | 9.92e-01 ± 6.10e-01 | 7.85e-01 ± 5.95e-01 (=) | 9.54e-01 ± 9.63e-01 (=) | **2.48e-01 ± 2.07e-01 (+)** |

Erreur = f(x) - f*, moyenne ± écart-type (ddof=1) sur les runs. Gras = meilleure moyenne. Symbole (test contre DE_seq) : + significativement meilleur, - significativement pire, = pas de différence significative.
