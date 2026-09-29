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

---

## État au 29 septembre

| Tâche | Branche | Pull request | État |
|---|---|---|---|
| 1.2 Répondre aux 5 questions | `lounis/1.2-questions-diagnostic` | à ouvrir | à relire |
| 1.3 Chronométrer le PSO | `amina/1.3-chronometrage` | #22 | fusionnée |
| 1.4 Dessiner le diagramme de séquence | `amina/1.4-1.5-docs` | #23 | à relire |
| 1.5 Établir la correspondance PSO → DE | `amina/1.4-1.5-docs` et `lounis/1.5-correspondance-pso-de` | #23 | à relire, versions à réunir |
| 2.0 Définir l'interface commune | `amina/interface-commune` | #24 | fusionnée |

**Prochaine tâche.** 2.4-2.6 Implémenter le DE/rand/1/bin séquentiel (issue #7), en utilisant
`benchmarks.h` et `results_csv.h`.
