### Bilan des tests statistiques

- Référence : `DE_seq` ; test : Mann-Whitney U (échantillons indépendants) ; seuil alpha = 0.05
- Correction de Holm, famille = les comparaisons d'une même (fonction, dim, pop)
- `+` : l'algorithme est significativement meilleur que la référence ; `-` significativement pire ; `=` différence non significative

#### Par algorithme et dimension

| Algorithme | Dim | + | = | - |
|---|---|---|---|---|
| DE_gpu_jDE | 10 | 9 | 3 | 0 |
| DE_gpu_jDE | 50 | 11 | 1 | 0 |
| DE_gpu_jDE | 100 | 10 | 2 | 0 |
| DE_gpu_v1 | 10 | 0 | 12 | 0 |
| DE_gpu_v1 | 50 | 0 | 12 | 0 |
| DE_gpu_v1 | 100 | 0 | 11 | 1 |
| DE_gpu_v2 | 10 | 1 | 11 | 0 |
| DE_gpu_v2 | 50 | 0 | 12 | 0 |
| DE_gpu_v2 | 100 | 0 | 12 | 0 |


Kruskal-Wallis (tous algorithmes) : 33 configurations sur 36 avec p < 0.05 (p brut, sans correction).
