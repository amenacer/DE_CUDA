# DE_CUDA — Évolution différentielle massivement parallèle sur GPU

Projet M2 IMDS (UHA), module Calcul massivement parallèle — L. Idoumghar.
Transformer le PSO CUDA fourni en DE (DE/rand/1/bin) massivement parallèle, le comparer
à un DE séquentiel et à Qin et al. (GECCO 2012) sur Shifted Sphere, Rosenbrock, Griewank, Rastrigin.

Rendu : 27 octobre 2026, 12h (article IEEE, code, slides).

## Arborescence

| Dossier | Contenu |
|---|---|
| `src/pso_original/` | Code PSO du prof, **non modifié** (référence) |
| `src/seq/` | DE séquentiel C++ (étape 2) |
| `src/cuda/` | DE CUDA v1 et v2 (étapes 3-4) |
| `notebooks/` | Notebooks Colab, un par étape |
| `scripts/` | Campagne expérimentale et analyse (Python) |
| `results/` | CSV des runs, `setup.txt` (matériel) |
| `figures/` | Figures pour l'article et les slides |
| `paper/` | Article LaTeX (IEEEtran) |
| `slides/` | Soutenance |
| `docs/` | Diagrammes UML, notes |

## Travailler depuis Colab

1. Créer un token GitHub : Settings → Developer settings → Personal access tokens →
   *Fine-grained* → accès au seul dépôt `DE_CUDA`, permission **Contents : Read and write**.
2. Dans Colab : 🔑 Secrets → `GITHUB_TOKEN` = le token → activer l'accès au notebook.
3. Début de séance : exécuter la cellule 0 du notebook (clone ou pull).
4. Fin de séance : `git add -A && git commit -m "..." && git push`.

Ne jamais écrire le token en clair dans un notebook : il serait publié avec le commit.

## Règles

- Une branche par étape (`etape2-seq`, `etape3-cuda-v1`…) fusionnée dans `main` quand le checkpoint est validé.
- Messages de commit courts et explicites : `seq: croisement binomial avec jrand`.
- Compilation GPU : `nvcc -O2 -arch=sm_75` (GPU T4 de Colab).
