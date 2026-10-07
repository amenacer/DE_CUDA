### Qualité des solutions, Dim = 50

| Fonction | Pop | DE_seq | DE_gpu_v1 | DE_gpu_v2 | DE_gpu_jDE |
|---|---|---|---|---|---|
| Shifted Sphere | 50 | 1.88e-04 ± 1.02e-04 | 2.92e-04 ± 1.59e-04 (=) | 2.65e-04 ± 2.67e-04 (=) | **3.51e-05 ± 2.12e-05 (+)** |
| Shifted Sphere | 100 | 2.42e-04 ± 1.78e-04 | 2.07e-04 ± 1.49e-04 (=) | 1.47e-04 ± 1.15e-04 (=) | **5.10e-05 ± 2.00e-05 (+)** |
| Shifted Sphere | 500 | 7.98e-04 ± 6.49e-04 | 8.74e-04 ± 9.59e-04 (=) | 6.81e-04 ± 5.33e-04 (=) | **1.12e-04 ± 1.02e-04 (+)** |
| Shifted Rastrigin | 50 | 4.18e+01 ± 2.18e+01 | 2.65e+01 ± 1.71e+01 (=) | 4.09e+01 ± 3.22e+01 (=) | **1.78e+01 ± 2.43e+01 (+)** |
| Shifted Rastrigin | 100 | 5.15e+01 ± 5.73e+01 | 3.41e+01 ± 1.59e+01 (=) | 2.32e+01 ± 2.58e+01 (=) | **8.01e+00 ± 5.86e+00 (+)** |
| Shifted Rastrigin | 500 | 1.91e+01 ± 1.70e+01 | 1.92e+01 ± 1.34e+01 (=) | 1.05e+01 ± 6.17e+00 (=) | **4.64e+00 ± 1.92e+00 (+)** |
| Shifted Rosenbrock | 50 | 2.67e+02 ± 3.98e+02 | 2.32e+02 ± 1.22e+02 (=) | 2.65e+02 ± 3.13e+02 (=) | **4.83e+01 ± 2.32e+01 (=)** |
| Shifted Rosenbrock | 100 | 2.42e+02 ± 1.60e+02 | 3.23e+02 ± 1.56e+02 (=) | 3.81e+02 ± 4.04e+02 (=) | **5.96e+01 ± 3.16e+01 (+)** |
| Shifted Rosenbrock | 500 | 7.26e+02 ± 6.56e+02 | 5.78e+02 ± 3.77e+02 (=) | 7.45e+02 ± 3.81e+02 (=) | **1.80e+02 ± 1.09e+02 (+)** |
| Shifted Griewank | 50 | 1.24e-01 ± 8.50e-02 | 1.47e-01 ± 9.62e-02 (=) | 1.27e-01 ± 1.20e-01 (=) | **3.81e-02 ± 2.69e-02 (+)** |
| Shifted Griewank | 100 | 1.59e-01 ± 1.34e-01 | 3.06e-01 ± 4.58e-01 (=) | 1.85e-01 ± 1.91e-01 (=) | **5.24e-02 ± 5.52e-02 (+)** |
| Shifted Griewank | 500 | 1.73e-01 ± 1.04e-01 | 6.05e-01 ± 1.09e+00 (=) | 2.38e-01 ± 1.42e-01 (=) | **5.29e-02 ± 2.25e-02 (+)** |

Erreur = f(x) - f*, moyenne ± écart-type (ddof=1) sur les runs. Gras = meilleure moyenne. Symbole (test contre DE_seq) : + significativement meilleur, - significativement pire, = pas de différence significative.
