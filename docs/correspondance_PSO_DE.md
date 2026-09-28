# Correspondance PSO -> DE

Ce document fait le lien entre l'implementation PSO originale
(`src/pso_original/`) et l'algorithme Differential Evolution (DE) que
l'equipe developpe pour ce projet. L'objectif est de reutiliser au
maximum l'infrastructure deja validee du PSO (fonctions objectif,
gestion memoire GPU, structure des kernels) en remplacant uniquement
la logique de deplacement des individus.

## 1. Concepts

| PSO | DE | Remarque |
|---|---|---|
| Particule | Individu | Un vecteur de `NUM_OF_DIMENSIONS` flottants dans les deux cas |
| Essaim (`NUM_OF_PARTICLES`) | Population (`POP_SIZE`) | Meme role : ensemble des candidats evalues a chaque iteration |
| Position | Position | Identique : le point candidat dans l'espace de recherche |
| Vitesse (`velocities[]`) | *(aucun equivalent)* | Le DE n'a pas de notion de vitesse ; le deplacement vient de la mutation |
| Meilleure position personnelle (`pBests[]`) | *(aucun equivalent)* | Le DE ne garde pas de memoire par individu, seulement sa position/fitness courante |
| Meilleure position globale (`gBest[]`) | Meilleur individu de la population | Recalcule a la volee en cherchant le minimum de `fitness[]`, pas de tableau dedie |

## 2. Parametres

| PSO (`kernel.h`) | DE (`de_kernel.h`) | Role |
|---|---|---|
| `OMEGA` (inertie, 0.5) | *(aucun)* | Pas d'inertie en DE |
| `c1` (coeff. personnel, 1.5) | *(aucun)* | Remplace par la mutation |
| `c2` (coeff. global, 1.5) | *(aucun)* | Remplace par la mutation |
| `rp`, `rg` (poids aleatoires par appel) | `DE_F` (0.5, fixe) | Facteur d'amplification de la mutation |
| *(aucun)* | `DE_CR` (0.9, fixe) | Probabilite de croisement, propre au DE |
| `MAX_ITER = NUM_OF_DIMENSIONS * 10^4` | `MAX_FES = MAX_FES_MULT * NUM_OF_DIMENSIONS` | Le PSO compte des iterations, le DE compte des evaluations de fonction (convention CEC) |

## 3. Regle de mise a jour

**PSO** (`kernelUpdateParticle`, `src/pso_original/kernel.cu:79-106`) :
```
velocities[i] = OMEGA * velocities[i]
              + c1 * rp * (pBests[i] - positions[i])
              + c2 * rg * (gBest[i % DIM] - positions[i])
positions[i] += velocities[i]
```
Chaque particule se deplace en fonction de sa propre trajectoire et de
l'attraction vers `pBest`/`gBest`.

**DE/rand/1/bin** (`k_de_step`, `de_kernel.cu:109-145`) :
```
mutant[j] = pop[a][j] + F * (pop[b][j] - pop[c][j])   // a, b, c != i, tires au hasard
trial[j]  = (u <= CR ou j == jrand) ? mutant[j] : pop[i][j]
si fitness(trial) <= fitness(pop[i]) : pop[i] = trial   // selection gloutonne
```
Le DE combine trois individus tires au hasard dans la population pour
generer un candidat (mutation), le mixe avec l'individu courant
(croisement binomial), puis ne garde le resultat que s'il est meilleur
(selection). Il n'y a pas de notion de trajectoire continue comme en PSO.

## 4. Fonctions objectif

Identiques dans les deux algorithmes : meme `switch (SELECTED_OBJ_FUNC)`
sur Levy, Rastrigin decale, Rosenbrock decale, Griewank decale, Sphere
decalee (`fitness_function` en PSO vs `dev_eval` en DE). Ce sont les
memes fonctions CEC, avec les memes biais et bornes de recherche
(`START_RANGE_MIN`/`MAX`) — c'est ce qui permet de comparer les deux
algorithmes sur un pied d'egalite.

## 5. Structure des kernels GPU

| PSO | DE | Role |
|---|---|---|
| `kernelUpdateParticle` | `k_prepare_rand_indices` + `k_de_step` | Calcule le nouveau candidat par thread |
| `kernelUpdatePBest` | *(integre dans `k_de_step`)* | En DE, l'evaluation et la selection se font dans le meme kernel, pas besoin d'un kernel separe |
| Boucle `for (iter < MAX_ITER)` cote host, sync + copies a chaque iteration | Boucle `while (fes < MAXFES)` cote host, meme structure | Les deux orchestrent depuis le host, un kernel par iteration |
| `curand` absent (rand via host, poids passes en argument) | `curandState` par thread (`curand_init`, `curand_uniform`) | Le DE genere son alea directement sur GPU (indices `r1/r2/r3`, mutation, croisement) |

## 6. Ce qui est reutilisable tel quel

- Les fonctions objectif (deja portees dans `src/seq/benchmarks.h` cote CPU)
- La structure "host lance un kernel par iteration, synchronise, cherche le meilleur"
- Les constantes de plage de recherche (`START_RANGE_MIN/MAX`) et les biais CEC

## 7. Ce qui doit etre reecrit

- Toute la logique de deplacement (vitesse/attraction -> mutation/croisement/selection)
- La generation aleatoire (le PSO tire cote host, le DE doit tirer cote device avec `curand` car chaque individu a besoin de 3 indices distincts par iteration)
- Le critere d'arret (iterations -> evaluations de fonction, convention CEC)
