### Bilan des tests statistiques

- Référence : `sde` ; test : Mann-Whitney U (échantillons indépendants) ; seuil alpha = 0.05
- Correction de Holm, famille = les comparaisons d'une même (fonction, dim, pop)
- `+` : l'algorithme est significativement meilleur que la référence ; `-` significativement pire ; `=` différence non significative

#### Par algorithme et dimension

| Algorithme | Dim | + | = | - |
|---|---|---|---|---|
| cuda_v1 | 10 | 0 | 11 | 1 |
| cuda_v1 | 50 | 0 | 8 | 4 |
| cuda_v1 | 100 | 0 | 10 | 2 |
| cuda_v2 | 10 | 0 | 11 | 1 |
| cuda_v2 | 50 | 1 | 7 | 4 |
| cuda_v2 | 100 | 0 | 9 | 3 |
| gpu_jde | 10 | 6 | 6 | 0 |
| gpu_jde | 50 | 7 | 5 | 0 |
| gpu_jde | 100 | 6 | 4 | 2 |


Kruskal-Wallis (tous algorithmes) : 24 configurations sur 36 avec p < 0.05 (p brut, sans correction).
