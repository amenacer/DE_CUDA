# Étape 7 — Variantes du DE sur GPU (issue #17, partie GPU)

GPU : Tesla T4

Sortie du notebook `notebooks/Etape7_variantes_DE_GPU.ipynb` (Colab, 7 octobre). Données brutes : `results/variantes_gpu_D50_N100.csv`.

```
de_cuda_v2.cu : compilation SANS aucun warning
de_cuda_variantes.cu : compilation SANS aucun warning
Controle v2 contre variantes (rand1, F = 0,5, CR = 0,3), D = 50, N = 100
| fonction | graine | v2 | variantes rand1 | verdict |
|---|---|---|---|---|
| sphere | 1 | 0.000000000e+00 | 0.000000000e+00 | identique |
| sphere | 2 | 0.000000000e+00 | 0.000000000e+00 | identique |
| sphere | 3 | 0.000000000e+00 | 0.000000000e+00 | identique |
| rosenbrock | 1 | 4.113724274e+01 | 4.113724274e+01 | identique |
| rosenbrock | 2 | 4.041335155e+01 | 4.041335155e+01 | identique |
| rosenbrock | 3 | 4.139833952e+01 | 4.139833952e+01 | identique |
| griewank | 1 | 0.000000000e+00 | 0.000000000e+00 | identique |
| griewank | 2 | 0.000000000e+00 | 0.000000000e+00 | identique |
| griewank | 3 | 0.000000000e+00 | 0.000000000e+00 | identique |
| rastrigin | 1 | 1.536470311e+02 | 1.536470311e+02 | identique |
| rastrigin | 2 | 1.601138035e+02 | 1.601138035e+02 | identique |
| rastrigin | 3 | 1.631731634e+02 | 1.631731634e+02 | identique |

=> OK : resultats identiques au bit pres
sphere (D = 50, N = 100, 10 runs) -- Kruskal-Wallis : H = 36.97, p = 6.06e-07 => difference significative entre variantes
| rang | variante | moyenne ± ET | mediane | rang moyen | p Holm vs reference |
|---|---|---|---|---|---|
| 1 | rand1 CR=0,1 | 0.000e+00 ± 0.00e+00 | 0.000e+00 | 23.5 | 1 |
| 2 | rand1 CR=0,3 | 0.000e+00 ± 0.00e+00 | 0.000e+00 | 23.5 | reference |
| 3 | jDE | 0.000e+00 ± 0.00e+00 | 0.000e+00 | 23.5 | 1 |
| 4 | rand1 CR=0,5 | 5.167e-30 ± 1.37e-29 | 0.000e+00 | 28.3 | 0.504 |
| 5 | rand1 CR=0,9 | 6.058e-29 ± 9.25e-29 | 0.000e+00 | 32.2 | 0.307 |
| 6 | ctb1 CR=0,3 | 2.386e-28 ± 1.18e-28 | 2.019e-28 | 52.0 | 0.00101 * |

rosenbrock (D = 50, N = 100, 10 runs) -- Kruskal-Wallis : H = 43.49, p = 2.95e-08 => difference significative entre variantes
| rang | variante | moyenne ± ET | mediane | rang moyen | p Holm vs reference |
|---|---|---|---|---|---|
| 1 | rand1 CR=0,9 | 2.772e+01 ± 1.62e+01 | 2.220e+01 | 12.9 | 0.00565 * |
| 2 | jDE | 3.295e+01 ± 2.05e+01 | 2.336e+01 | 17.1 | 0.0257 * |
| 3 | rand1 CR=0,5 | 3.602e+01 ± 3.91e-01 | 3.595e+01 | 22.5 | 0.000913 * |
| 4 | rand1 CR=0,3 | 4.112e+01 ± 4.68e-01 | 4.116e+01 | 32.5 | reference |
| 5 | rand1 CR=0,1 | 4.794e+01 ± 3.41e+00 | 4.659e+01 | 42.5 | 0.000913 * |
| 6 | ctb1 CR=0,3 | 1.726e+03 ± 2.81e+03 | 3.408e+02 | 55.5 | 0.000913 * |

griewank (D = 50, N = 100, 10 runs) -- Kruskal-Wallis : H = 0.00, p = 1 => pas de difference detectee
| rang | variante | moyenne ± ET | mediane | rang moyen | p Holm vs reference |
|---|---|---|---|---|---|
| 1 | rand1 CR=0,1 | 0.000e+00 ± 0.00e+00 | 0.000e+00 | 30.5 | 1 |
| 2 | rand1 CR=0,3 | 0.000e+00 ± 0.00e+00 | 0.000e+00 | 30.5 | reference |
| 3 | rand1 CR=0,5 | 0.000e+00 ± 0.00e+00 | 0.000e+00 | 30.5 | 1 |
| 4 | rand1 CR=0,9 | 0.000e+00 ± 0.00e+00 | 0.000e+00 | 30.5 | 1 |
| 5 | ctb1 CR=0,3 | 0.000e+00 ± 0.00e+00 | 0.000e+00 | 30.5 | 1 |
| 6 | jDE | 0.000e+00 ± 0.00e+00 | 0.000e+00 | 30.5 | 1 |

rastrigin (D = 50, N = 100, 10 runs) -- Kruskal-Wallis : H = 56.20, p = 7.39e-11 => difference significative entre variantes
| rang | variante | moyenne ± ET | mediane | rang moyen | p Holm vs reference |
|---|---|---|---|---|---|
| 1 | jDE | 0.000e+00 ± 0.00e+00 | 0.000e+00 | 5.5 | 0.000319 * |
| 2 | rand1 CR=0,1 | 2.364e-11 ± 3.24e-11 | 7.315e-12 | 15.5 | 0.000731 * |
| 3 | ctb1 CR=0,3 | 1.338e+02 ± 1.27e+01 | 1.368e+02 | 25.8 | 0.000879 * |
| 4 | rand1 CR=0,3 | 1.655e+02 ± 9.73e+00 | 1.618e+02 | 35.5 | reference |
| 5 | rand1 CR=0,9 | 2.273e+02 ± 2.72e+01 | 2.351e+02 | 47.1 | 0.000879 * |
| 6 | rand1 CR=0,5 | 2.552e+02 ± 1.20e+01 | 2.546e+02 | 53.6 | 0.000731 * |

Classement global (rang moyen par fonction, puis moyenne des 4 fonctions ; plus petit = meilleur)
| variante | sphere | rosenbrock | griewank | rastrigin | moyenne |
|---|---|---|---|---|---|
| jDE | 23.5 | 17.1 | 30.5 | 5.5 | 19.15 |
| rand1 CR=0,1 | 23.5 | 42.5 | 30.5 | 15.5 | 28.00 |
| rand1 CR=0,3 | 23.5 | 32.5 | 30.5 | 35.5 | 30.50 |
| rand1 CR=0,9 | 32.2 | 12.9 | 30.5 | 47.1 | 30.68 |
| rand1 CR=0,5 | 28.3 | 22.5 | 30.5 | 53.6 | 33.73 |
| ctb1 CR=0,3 | 52.0 | 55.5 | 30.5 | 25.8 | 40.95 |

* : p Holm < 0,05. Les ecarts-types sont des ecarts-types de population (statistics.pstdev).
jDE : parametres moyens en fin de run (10 runs)
| fonction | F moyen | CR moyen |
|---|---|---|
| sphere | 0.557 | 0.512 |
| rosenbrock | 0.373 | 0.545 |
| griewank | 0.378 | 0.309 |
| rastrigin | 0.365 | 0.320 |

Temps moyen (s) par run, D = 50, N = 100, 10 runs
| variante | sphere | rosenbrock | griewank | rastrigin |
|---|---|---|---|---|
| rand1 CR=0,1 | 0.0493 | 0.0520 | 0.0695 | 0.0577 |
| rand1 CR=0,3 | 0.0494 | 0.0525 | 0.0685 | 0.0583 |
| rand1 CR=0,5 | 0.0495 | 0.0537 | 0.0687 | 0.0587 |
| rand1 CR=0,9 | 0.0490 | 0.0519 | 0.0680 | 0.0576 |
| ctb1 CR=0,3 | 0.0690 | 0.0719 | 0.0871 | 0.0776 |
| jDE | 0.0575 | 0.0619 | 0.0770 | 0.0688 |
```

## Lecture des résultats

- **Contrôle.** Avec rand/1, F = 0,5 et CR = 0,3, le programme des variantes redonne au bit près les erreurs de la v2 : les variantes partent bien du même code.
- **jDE est la meilleure variante en moyenne** (rang moyen 19,2). Elle résout Rastrigin D = 50 à chaque run (erreur 0), alors que le DE de référence (CR = 0,3) reste vers 165.
- **Rastrigin est séparable** : chaque coordonnée peut être optimisée indépendamment. Un CR faible (0,1) change peu de coordonnées à la fois, ce qui convient à ce type de fonction ; c'est pourquoi CR = 0,1 descend à 1e-11. Ce résultat ne se généralise pas aux fonctions non séparables : sur Rosenbrock, c'est l'inverse (CR = 0,9 est le meilleur, CR = 0,1 le moins bon des rand/1).
- **current-to-best/1 (ctb1) est la moins bonne variante** : convergence prématurée (Rosenbrock : médiane 341, et 3 runs au-dessus de 2 000 ; Sphere : arrêt vers 2e-28 au lieu de 0). Elle coûte aussi environ 40 % de temps en plus (recherche du meilleur avant chaque génération).
- **Griewank** est résolue (erreur 0) par toutes les variantes en D = 50 : le test ne peut rien départager.
- **Limites.** Une seule taille (D = 50, N = 100), 10 runs, une valeur de F. Les égalités exactes (0) donnent des rangs ex aequo. La figure de convergence (`figures/convergence_variantes_D50.png`) est produite par le notebook mais n'a pas été versionnée ici ; elle se régénère en relançant la cellule de la figure.
