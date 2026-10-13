### Qualité des solutions, Dim = 100

| Fonction | Pop | sde | cuda_v1 | cuda_v2 | gpu_jde |
|---|---|---|---|---|---|
| Shifted Sphere | 50 | **0.00e+00 ± 0.00e+00** | **0.00e+00 ± 0.00e+00 (=)** | **0.00e+00 ± 0.00e+00 (=)** | 6.75e-28 ± 7.22e-28 (-) |
| Shifted Sphere | 100 | **0.00e+00 ± 0.00e+00** | 1.86e-30 ± 4.35e-30 (=) | 5.59e-29 ± 1.02e-28 (-) | 2.15e-29 ± 6.35e-29 (=) |
| Shifted Sphere | 500 | 7.33e-02 ± 5.45e-03 | 9.79e-02 ± 1.17e-02 (-) | 9.82e-02 ± 6.54e-03 (-) | **4.32e-15 ± 1.33e-15 (+)** |
| Shifted Rastrigin | 50 | 4.93e+02 ± 1.02e+01 | 4.95e+02 ± 1.84e+01 (=) | 4.95e+02 ± 1.14e+01 (=) | **4.97e-01 ± 8.46e-01 (+)** |
| Shifted Rastrigin | 100 | 5.42e+02 ± 1.09e+01 | 5.32e+02 ± 1.45e+01 (=) | 5.33e+02 ± 1.44e+01 (=) | **0.00e+00 ± 0.00e+00 (+)** |
| Shifted Rastrigin | 500 | 6.28e+02 ± 1.42e+01 | 6.23e+02 ± 1.75e+01 (=) | 6.23e+02 ± 1.78e+01 (=) | **3.54e+01 ± 3.22e+00 (+)** |
| Shifted Rosenbrock | 50 | 8.96e+01 ± 1.69e+01 | 8.46e+01 ± 2.28e+00 (=) | 8.51e+01 ± 2.95e+00 (=) | **7.94e+01 ± 6.43e+01 (=)** |
| Shifted Rosenbrock | 100 | **8.96e+01 ± 2.01e-01** | 9.00e+01 ± 6.94e-01 (=) | 8.97e+01 ± 1.95e-01 (=) | 1.12e+02 ± 3.17e+01 (=) |
| Shifted Rosenbrock | 500 | 2.19e+04 ± 3.41e+03 | 2.33e+04 ± 3.41e+03 (=) | 2.50e+04 ± 3.26e+03 (=) | **1.04e+02 ± 2.45e+01 (+)** |
| Shifted Griewank | 50 | **0.00e+00 ± 0.00e+00** | **0.00e+00 ± 0.00e+00 (=)** | **0.00e+00 ± 0.00e+00 (=)** | 4.40e-02 ± 5.55e-02 (-) |
| Shifted Griewank | 100 | **0.00e+00 ± 0.00e+00** | **0.00e+00 ± 0.00e+00 (=)** | **0.00e+00 ± 0.00e+00 (=)** | **0.00e+00 ± 0.00e+00 (=)** |
| Shifted Griewank | 500 | 4.39e-02 ± 3.27e-03 | 5.57e-02 ± 4.99e-03 (-) | 5.71e-02 ± 5.39e-03 (-) | **1.90e-15 ± 9.24e-16 (+)** |

Erreur = f(x) - f*, moyenne ± écart-type (ddof=1) sur les runs. Gras = meilleure moyenne. Symbole (test contre sde) : + significativement meilleur, - significativement pire, = pas de différence significative.
