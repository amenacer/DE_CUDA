### Qualité des solutions, Dim = 50

| Fonction | Pop | sde | cuda_v1 | cuda_v2 | gpu_jde |
|---|---|---|---|---|---|
| Shifted Sphere | 50 | **0.00e+00 ± 0.00e+00** | **0.00e+00 ± 0.00e+00 (=)** | **0.00e+00 ± 0.00e+00 (=)** | 5.05e-30 ± 1.60e-29 (=) |
| Shifted Sphere | 100 | **0.00e+00 ± 0.00e+00** | **0.00e+00 ± 0.00e+00 (=)** | **0.00e+00 ± 0.00e+00 (=)** | **0.00e+00 ± 0.00e+00 (=)** |
| Shifted Sphere | 500 | 1.09e-04 ± 1.14e-05 | 2.28e-04 ± 2.80e-05 (-) | 2.36e-04 ± 2.82e-05 (-) | **7.30e-11 ± 1.76e-11 (+)** |
| Shifted Rastrigin | 50 | 1.51e+02 ± 6.91e+00 | 1.46e+02 ± 6.92e+00 (=) | 1.39e+02 ± 7.82e+00 (+) | **0.00e+00 ± 0.00e+00 (+)** |
| Shifted Rastrigin | 100 | 1.68e+02 ± 6.01e+00 | 1.65e+02 ± 9.99e+00 (=) | 1.65e+02 ± 1.03e+01 (=) | **0.00e+00 ± 0.00e+00 (+)** |
| Shifted Rastrigin | 500 | 2.24e+02 ± 6.49e+00 | 2.29e+02 ± 7.92e+00 (=) | 2.22e+02 ± 7.61e+00 (=) | **1.79e+01 ± 2.67e+00 (+)** |
| Shifted Rosenbrock | 50 | 3.79e+01 ± 1.58e+00 | 3.92e+01 ± 1.33e+00 (-) | 3.95e+01 ± 1.36e+00 (-) | **1.46e+00 ± 2.41e+00 (+)** |
| Shifted Rosenbrock | 100 | 4.10e+01 ± 3.53e-01 | 4.28e+01 ± 5.35e+00 (=) | 4.11e+01 ± 4.93e-01 (=) | **3.29e+01 ± 2.16e+01 (=)** |
| Shifted Rosenbrock | 500 | 3.58e+02 ± 5.32e+01 | 4.34e+02 ± 5.53e+01 (-) | 4.00e+02 ± 3.36e+01 (-) | **4.52e+01 ± 1.01e+00 (+)** |
| Shifted Griewank | 50 | **0.00e+00 ± 0.00e+00** | **0.00e+00 ± 0.00e+00 (=)** | **0.00e+00 ± 0.00e+00 (=)** | **0.00e+00 ± 0.00e+00 (=)** |
| Shifted Griewank | 100 | **0.00e+00 ± 0.00e+00** | **0.00e+00 ± 0.00e+00 (=)** | **0.00e+00 ± 0.00e+00 (=)** | **0.00e+00 ± 0.00e+00 (=)** |
| Shifted Griewank | 500 | 2.11e-04 ± 2.99e-05 | 4.07e-04 ± 5.89e-05 (-) | 4.26e-04 ± 4.27e-05 (-) | **9.89e-11 ± 2.00e-11 (+)** |

Erreur = f(x) - f*, moyenne ± écart-type (ddof=1) sur les runs. Gras = meilleure moyenne. Symbole (test contre sde) : + significativement meilleur, - significativement pire, = pas de différence significative.
