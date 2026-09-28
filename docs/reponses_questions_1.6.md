# Reponses aux 5 questions de la section 1.6

Notebook source : `notebooks/Etape1_PSO_diagnostic.ipynb`, section 1.6.
Reponses basees sur `src/pso_original/kernel.cu` et `kernel.cpp` (config
par defaut : `NUM_OF_PARTICLES = 512`, `NUM_OF_DIMENSIONS = 3`, 32
threads/bloc).

## 1. Dans `kernelUpdateParticle`, combien de threads sont lances en tout ? Combien travaillent vraiment ?

`size = NUM_OF_PARTICLES * NUM_OF_DIMENSIONS = 512 * 3 = 1536`.
`threadsNum = 32`, `blocksNum = ceil(size / threadsNum) = 48`.

Threads lances = `blocksNum * threadsNum = 48 * 32 = 1536`.

Ici `1536 / 32 = 48` tombe pile, donc les 1536 threads lances passent le
garde-fou `if (i >= size) return;` et travaillent **tous** : chaque
thread met a jour une coordonnee d'une particule. Ce n'est pas garanti
en general (bug #12 releve dans le diagnostic : `ceil(size/threadsNum)`
avec une division entiere peut tronquer et lancer moins de threads que
necessaire si `size` n'est pas un multiple de `threadsNum`).

## 2. Pourquoi `gBest[i % NUM_OF_DIMENSIONS]` et pas `gBest[i]` ?

`gBest` ne contient qu'**un seul point** de dimension `D` (le meilleur
trouve par tout l'essaim), donc un tableau de taille `D`, alors que `i`
parcourt `N * D` indices (une particule = D coordonnees consecutives).

`i % D` retombe toujours dans `[0, D-1]` : ca convertit l'indice global
du thread en indice de la **dimension** qu'il traite, quelle que soit
la particule a laquelle il appartient. Utiliser `gBest[i]` directement
lirait hors des bornes du tableau des que `i >= D` (soit pour 511 des
512 particules ici).

## 3. Avec D = 100 et N = 500, combien de `cudaMemcpy` par iteration, et de quelle taille ?

Dans la boucle `for (iter < MAX_ITER)` de `cuda_pso` (`kernel.cu:169-199`),
il y a exactement **2 `cudaMemcpy` par iteration** (les copies initiales
et finales sont hors boucle, faites une seule fois) :

1. `cudaMemcpy(pBests, devPBest, ..., DeviceToHost)` — taille
   `N * D * sizeof(float) = 500 * 100 * 4 = 200 000 octets` (~195 Kio)
2. `cudaMemcpy(devGBest, gBest, ..., HostToDevice)` — taille
   `D * sizeof(float) = 100 * 4 = 400 octets`

Le premier copie **toute la population** a chaque iteration rien que
pour chercher un maximum cote CPU — c'est le probleme de performance
souligne au point 10 du diagnostic (a remplacer par une reduction GPU).

## 4. Pourquoi une race condition ne fait-elle pas forcement planter le programme, et pourquoi c'est pire qu'un plantage ?

`tempParticle1`/`tempParticle2` (`kernel.cu:8-9`) sont des adresses
memoire valides et de la bonne taille : chaque thread lit/ecrit dedans
sans jamais sortir des bornes allouees. Le GPU ne detecte donc **aucune
erreur memoire** (pas de segfault, pas d'acces illegal) — le probleme
n'est pas *ou* on ecrit, mais *quand*, plusieurs threads se marchant
dessus sans synchronisation.

C'est pire qu'un plantage parce que :
- Le programme tourne et **affiche un resultat qui a l'air plausible**
  (pas d'erreur, pas de crash) : rien n'alerte qu'il y a un bug.
- Le resultat est **non deterministe** : il peut varier d'une execution
  a l'autre selon l'ordonnancement des threads, ce qui rend le bug
  difficile a reproduire et a deboguer.
- Une fois ce bug non detecte, il **contamine tout ce qui est construit
  dessus** (comparaisons de performance, article, conclusions) sans que
  personne ne s'en rende compte.

## 5. Avec le budget correct (10^4 x D FE), combien d'iterations pour D = 10, N = 50 ? Et N = 500 ? Implication ?

Budget total : `maxFE = 10^4 * D = 10^4 * 10 = 100 000` evaluations de
fonction. Avec la correction du point 7 du diagnostic
(`max_iter = maxFE / N`, une evaluation par particule et par
iteration) :

- **N = 50** : `100 000 / 50 = 2 000` iterations
- **N = 500** : `100 000 / 500 = 200` iterations

**Implication** : a budget de calcul fixe, augmenter la population
reduit proportionnellement le nombre d'iterations. C'est un compromis
largeur vs profondeur de recherche :
- N = 50 → peu de diversite par iteration mais 10x plus de generations
  pour affiner (exploiter) les meilleures solutions trouvees.
- N = 500 → forte diversite/parallelisme par iteration, mais seulement
  200 generations pour converger, ce qui peut etre insuffisant pour que
  l'algorithme affine sa solution (risque de s'arreter avant d'avoir
  vraiment converge, malgre un GPU mieux occupe a chaque iteration).

Il n'y a pas de population "gratuite" : plus de particules par
iteration ne compense pas automatiquement moins d'iterations, il faut
trouver un compromis empirique (c'est justement ce que la campagne
d'experiences de l'etape 5 devra mesurer).
