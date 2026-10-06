# Journal du projet DE_CUDA

Ce journal explique, étape par étape, **ce qu'on a fait, pourquoi, et ce qu'on a obtenu**.
On ajoute une section à la fin de chaque tâche terminée.

---

## Règles de travail

1. **Une tâche = une issue dont le titre commence par un verbe à l'infinitif**, pour dire clairement
   ce qu'on veut faire : « Chronométrer le PSO avec cudaEvent », « Implémenter le DE séquentiel »…
2. **Une branche par tâche**, nommée `prenom/numero-verbe-objet`
   (ex. `amina/3.1-mettre-en-place-cuda-check`). On ne travaille jamais directement sur `main`.
3. **Une pull request par branche**, avec `closes #N` dans la description et le résultat obtenu
   (sortie de test, tableau, figure).
4. **On ne fusionne jamais sa propre pull request.** Un autre membre la relit, pose ses questions,
   puis c'est lui qui clique sur *Merge*. Après la fusion, on supprime la branche.
5. Messages de commit courts et explicites : `seq : croisement binomial avec jrand (refs #7)`.

---

## Étape 0 — Organiser le projet

**Objectif.** Travailler à trois sans se marcher dessus, et garder la trace de qui fait quoi.

**Ce qu'on a fait.**
- Dépôt GitHub `DE_CUDA` avec l'arborescence du `README.md` (`src/`, `notebooks/`, `docs/`, `paper/`…).
- 21 issues (une par tâche du plan), réparties en 8 jalons (milestones) avec une date limite chacun,
  et des étiquettes par responsable (Amina, Lounis, Lisa, Équipe).
- Tableau GitHub Projects « DE_CUDA - Projet CUDA M2 » : Todo → In Progress → In Review → Done.
- Travail sur GPU depuis Google Colab (GPU T4), avec un token GitHub rangé dans les *Secrets* de Colab,
  jamais écrit dans un notebook.
- Squelette de l'article au format IEEE dans `paper/main.tex` (aussi sur Overleaf).

---

## Étape 1 — Comprendre le PSO du professeur

### 1.1 — Faire tourner le code original
**Objectif.** Avoir une référence (*baseline*) qui fonctionne avant de modifier quoi que ce soit.

**Ce qu'on a fait.** Copie **non modifiée** du code du prof dans `src/pso_original/`
(`kernel.h`, `kernel.cpp`, `kernel.cu`, `main.cpp`), compilée avec
`nvcc -O2 -arch=sm_75` (architecture du T4) et exécutée dans `notebooks/Etape1_PSO_diagnostic.ipynb`.

**Résultat.** Environ 3 s pour Levy 3D, 512 particules, 30 000 itérations.

### 1.2 — Diagnostiquer le code
**Objectif.** Repérer les défauts avant de réutiliser ce squelette pour le DE.

**Ce qu'on a trouvé (14 défauts, détail dans le notebook de l'étape 1).**
- *Résultats faux* : race condition sur `tempParticle` partagé par tous les threads ;
  produit de Griewank initialisé à 0 ; `getRandom` qui tire dans [-5.12, 6.12) ; pas de décalage `o` ;
  mêmes bornes pour toutes les fonctions ; aucune gestion des bornes.
- *Protocole non respecté* : 10⁴·D **itérations** au lieu de 10⁴·D **évaluations** (512 fois trop) ;
  D et N fixés à la compilation ; `float` + biais qui empêchent de mesurer une erreur < 3·10⁻⁵.
- *Performance* : gBest calculé sur le CPU à chaque itération avec deux copies ; 1 thread sur D actif
  dans `kernelUpdatePBest` ; `ceil` appliqué après une division entière ; aléatoire tiré sur le CPU ;
  `clock()` qui mesure le temps CPU et non le temps réel.

**Preuve.** `src/pso_original/verif.cpp` démontre 4 de ces bugs par l'exécution (2, 3, 7 et 9).
Réponses aux 5 questions de compréhension : branche `lounis/1.2-questions-diagnostic` (issue #1).

### 1.3 — Chronométrer le PSO (issue #2, PR #22, fusionnée)
**Objectif.** Savoir où passe le temps : calcul GPU, transferts ou CPU ?

**Ce qu'on a fait.** Copie du code dans `src/pso_timing/` avec des chronomètres autour des 5 morceaux
d'une itération : `cudaEvent` pour ce qui passe par la file du GPU (les kernels étant asynchrones,
une montre CPU ne mesurerait que leur lancement), `std::chrono` pour la boucle CPU.
Notebook : `notebooks/Etape1_3_chronometrage.ipynb`.

**Résultat (T4, Levy 3D, N = 512, 30 000 itérations).**

| Morceau | ms | % | µs / itération |
|---|---|---|---|
| kernelUpdateParticle (GPU) | 144,3 | 3,8 | 4,8 |
| kernelUpdatePBest (GPU) | 1108,8 | 29,5 | 37,0 |
| copie pBests GPU → CPU (6 Ko) | 339,1 | 9,0 | 11,3 |
| boucle CPU du gBest | 978,3 | 26,0 | 32,6 |
| copie gBest CPU → GPU (12 octets) | 257,2 | 6,8 | 8,6 |
| somme des morceaux | 2827,7 | 75,3 | |
| total de la boucle | 3755,6 | 100 | 125,2 |

**Ce qu'on en retient.**
- Le calcul GPU utile ne représente que **33 %** du temps.
- Copier 12 octets coûte presque autant que 6 Ko : un transfert coûte surtout sa **latence fixe**.
- Les 25 % non attribués sont le coût de la mesure elle-même (4 synchronisations par itération) :
  le total instrumenté (3,76 s) dépasse les ~3 s du programme d'origine.
- **Règles pour notre DE** : aucun `cudaMemcpy` dans la boucle ; meilleur individu trouvé sur le GPU
  (réduction) ; peu de kernels ; tous les threads actifs.

### 1.4 — Dessiner le diagramme de séquence (issue #3, PR #23, en attente)
**Objectif.** Montrer en une image les échanges CPU ↔ GPU à chaque itération (exigé par l'article).

**Ce qu'on a fait.** Diagramme UML de séquence en Mermaid (`docs/uml/sequence_pso.mmd`),
exporté en image (`docs/uml/sequence_pso.png`), expliqué dans `docs/sequence_pso.md`,
avec les temps mesurés en 1.3 sur chaque flèche.

**Ce qu'on en retient.** Dans la boucle, 3 flèches sur 5 (copie vers le CPU, boucle CPU, copie retour)
ne font aucun calcul GPU : c'est exactement ce que le DE doit supprimer.

### 1.5 — Établir la correspondance PSO → DE (issue #4, PR #23 et branche de Lounis, en attente)
**Objectif.** Savoir, pour chaque morceau du PSO, ce qu'il devient dans le DE/rand/1/bin.

**Ce qu'on a fait.** Tableau élément par élément dans `docs/correspondance_PSO_DE.md` :
vitesses supprimées, pBest remplacé par la population, ω/c1/c2 remplacés par F = 0,5 et CR = 0,3,
aléatoire tiré sur le GPU avec cuRAND, gBest calculé seulement à la fin, budget en évaluations.

**Point clé.** Dans le DE, l'individu i lit trois autres individus. On écrit donc les essais
dans un tableau séparé `trial` et on ne sélectionne qu'après, sinon on crée une race condition.
Découpage prévu en kernels : `k_init → k_eval → [k_mutation_croisement → k_eval → k_selection] → argmin`.

⚠️ Deux versions de ce fichier existent (PR #23 d'Amina et branche `lounis/1.5-correspondance-pso-de`) :
à fusionner en une seule lors de la relecture.

---

## Étape 2 — Préparer le DE séquentiel

### 2.0 — Définir l'interface commune (issue #5, PR #24, fusionnée)
**Objectif.** Garantir une comparaison honnête : le DE séquentiel et le DE GPU optimisent
exactement les mêmes fonctions, avec le même décalage et le même budget, et écrivent
leurs résultats dans le même format.

**Ce qu'on a fait.** Notebook `notebooks/Etape2_interface_commune.ipynb`, qui produit :
- `src/common/benchmarks.h` : Sphere, Rosenbrock, Griewank, Rastrigin en `double`, qui renvoient
  directement l'**erreur** (0 à l'optimum) ; une seule copie de chaque fonction, marquée
  `__host__ __device__` pour servir sur CPU et GPU ; domaines CEC 2005 ; décalage `o` déterministe
  (générateur splitmix64, graine 1000·f + D) ; budget `max_fe(D) = 10⁴·D`.
- `src/common/results_csv.h` : une ligne par run, `algo,func,D,N,seed,best_error,time_s,FE`.
- `src/common/test_benchmarks.cpp` et `src/common/test_gpu.cu` : les tests.
- `docs/INTERFACE.md` : le contrat, pour toute l'équipe et pour l'article.

**Validation.**
- f(o) = 0 exactement pour les 4 fonctions en D = 10, 50 et 100.
- `o` reste dans 80 % du domaine.
- Valeurs de référence identiques à un calcul NumPy indépendant (écart relatif < 10⁻¹²).
- Même résultat sur GPU et sur CPU (écart relatif < 10⁻¹²).

### 2.3 à 2.9 — DE séquentiel (issues #6, #7, #8 — Lounis)
Réalisé par Lounis sur ses branches `lounis/2.3-tests-numpy`, `lounis/2.4-2.6-de-sequentiel` et
`lounis/2.7-2.9-csv-convergence` : `src/seq/sde.cpp` (DE/rand/1/bin, F = 0,5, CR = 0,3, mêmes fonctions
et même CSV que l'interface commune). C'est la référence utilisée à l'étape 3.6 pour valider le GPU.

---

## Étape 3 — DE CUDA v1

### 3.1 à 3.5 — Écrire le DE CUDA v1 (issues #9 et #10)
**Objectif.** Un premier DE sur GPU, simple et juste, qui respecte les leçons du chronométrage du PSO.

**Ce qu'on a fait.** `src/cuda/cuda_utils.h` et `src/cuda/de_cuda_v1.cu`, notebook `notebooks/Etape3_DE_CUDA_v1.ipynb`.
- `CUDA_CHECK` sur chaque appel CUDA, `cudaGetLastError` après chaque lancement de kernel.
- Mémoire globale : `pop`, `trial`, `fit`, `fit_trial` ; décalage `o` en **mémoire constante** ;
  un générateur cuRAND (Philox) par individu.
- Une génération = 3 kernels (mutation-croisement, évaluation, sélection), un thread par individu,
  grille (N + T − 1) / T. Essais dans `trial`, sélection ensuite : pas de race condition.
- **Aucun transfert dans la boucle** ; `cudaFree(0)` avant le chrono ; `cudaEvent` autour de l'algorithme.

**Validation (GPU T4, `results/etape3_v1_controles.md`).**
- Compilation sans aucun warning (`-Wall -Wextra`, `-arch=sm_75`).
- Lancement volontairement faux (2048 threads par bloc) : arrêt avec un message clair (fichier, ligne, erreur CUDA).
- Même graine → même résultat, sur les 4 fonctions.
- Écart de temps entre deux runs identiques : 0,06 % (critère : < 10 %).

### 3.6 et 3.7 — Valider le GPU contre le séquentiel et mesurer le speedup (issue #11)
**Ce qu'on a fait.** Notebook `notebooks/Etape3_6_validation_speedup.ipynb`, résultats dans
`results/etape3_validation_speedup.md`, `results/*_v1*.csv` et `figures/speedup_v1.png`.

**Résultats.**
- Sphere et Rastrigin, D = 10, N = 50, 10 runs : les deux programmes trouvent l'optimum exact à chaque run.
- Complément sur des cas non triviaux : Rastrigin D = 50 (p = 0,140) et Rosenbrock D = 10 (p = 0,089, limite) :
  pas de différence significative au seuil de 5 % (Mann-Whitney, 10 runs).
- Speedup de la v1 (N = 100) : de × 0,98 (Griewank D = 50, plus lent que le CPU) à × 2,3 (Rastrigin D = 50).
  Modeste : un seul bloc de 128 threads (1 SM sur 40), chaque thread bouclant sur D gènes avec des accès
  mémoire non coalescés.

**À savoir.** Le séquentiel remplace x_i dès que l'essai est meilleur ; le GPU remplace en fin de
génération (obligatoire en parallèle). Ce ne sont pas exactement le même algorithme, mais aucune
différence de qualité n'a été mesurée.

---

## Étape 4 — DE CUDA v2 optimisé

### 4.1 à 4.5 — Optimiser le DE CUDA (issues #12 et #13)
**Ce qu'on a fait.** `src/cuda/de_cuda_v2.cu`, notebook `notebooks/Etape4_DE_CUDA_v2.ipynb`.
- **Kernel P** : un thread par individu tire r1, r2, r3 et jrand (matrice d'indices).
- **Kernel fusionné MCER** (Mutation, Croisement, Évaluation, Remplacement) : un bloc par individu,
  un thread par gène, essai en **mémoire partagée**, f(u) par **réduction parallèle**, gagnant écrit
  dans un **second tampon** (double tampon, puis échange des pointeurs).
- Meilleur individu par **argmin sur GPU** : 16 octets copiés à la fin.
- **Taille de bloc automatique**, fixée par un balayage mesuré : 32 threads si D ≤ 32 ou N ≥ 4 × (nombre de SM),
  sinon 64. Les blocs de 128 ou 256 threads ont 100 % d'occupation théorique mais sont toujours plus lents
  que le meilleur choix entre 32 et 64. Règle ajustée sur 12 cas mesurés (D = 50/100, N = 50/100/500).

**Résultats (GPU T4, `results/etape4_v2_resultats.md`, `figures/temps_v1_v2_N.png`).**
- Chaque run vérifie argmin GPU = min CPU et f(réduction GPU) = f(CPU) : tous `sante=OK`.
- Pas de différence de qualité détectée avec la v1 (Mann-Whitney, 10 runs) : Rosenbrock D = 10 (p = 0,623),
  Rosenbrock D = 50 (p = 0,345), Rastrigin D = 50 (p = 0,104).
- v2 de **× 1,7 à × 14,6** plus rapide que la v1 (v1 non optimisée : pour N ≤ 128, un seul bloc, donc 1 SM sur 40).
- Ordre de grandeur face au séquentiel (Rastrigin, N = 100, même session Colab) : **≈ × 4 en D = 10, ≈ × 22 en D = 50,
  ≈ × 33 en D = 100**. Le speedup définitif viendra de la campagne de l'étape 5.

---

## État au 6 octobre

| Tâche | Branche | Pull request | État |
|---|---|---|---|
| 1.2 Répondre aux 5 questions | `lounis/1.2-questions-diagnostic` | à ouvrir | à relire |
| 1.3 Chronométrer le PSO | `amina/1.3-chronometrage` | #22 | fusionnée |
| 1.4 Dessiner le diagramme de séquence | `amina/1.4-1.5-docs` | #23 | à relire |
| 1.5 Établir la correspondance PSO → DE | `amina/1.4-1.5-docs` et `lounis/1.5-correspondance-pso-de` | #23 | à relire, versions à réunir |
| 2.0 Définir l'interface commune | `amina/interface-commune` | #24 | fusionnée |
| 2.3-2.9 DE séquentiel (Lounis) | `lounis/2.3-…`, `lounis/2.4-…`, `lounis/2.7-…` | voir GitHub | à relire |
| 3.1-3.5 Écrire le DE CUDA v1 | `amina/3.1-3.5-ecrire-de-cuda-v1` | voir GitHub | à relire |
| 3.6-3.7 Valider le GPU et mesurer le speedup | `amina/3.6-3.7-valider-gpu-speedup` | voir GitHub | à relire |
| 4.1-4.5 Optimiser le DE CUDA (v2) | `amina/4.1-4.5-optimiser-de-cuda-v2` | voir GitHub | à relire |

**Prochaine tâche.** 5.1-5.5 Écrire run_all.py et lancer la campagne complète (issue #14, Lounis) :
4 fonctions × 3 D × 3 N × 10 runs, pour `sde`, `cuda_v1` et `cuda_v2`, avec la même ligne de commande.
