// cuda_utils.h -- outils communs aux versions GPU du DE (issue #9, tache 3.1)
//
// CUDA_CHECK(appel)  : verifie le code de retour d'un appel CUDA (cudaMalloc,
//                      cudaMemcpy...) et arrete le programme avec un message clair.
// CUDA_CHECK_KERNEL(): a placer juste apres chaque lancement <<<...>>>. Un lancement
//                      ne renvoie rien : on recupere l'erreur avec cudaGetLastError().
//                      Compile avec -DDE_DEBUG, on attend aussi la fin du kernel
//                      (cudaDeviceSynchronize) pour attraper les erreurs d'execution
//                      a la bonne ligne (acces memoire hors limites, etc.).
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "Erreur CUDA %s:%d : %s\n  dans : %s\n",         \
                         __FILE__, __LINE__, cudaGetErrorString(err_), #call);    \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

#ifdef DE_DEBUG
#define CUDA_CHECK_KERNEL()                                                       \
    do { CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); } while (0)
#else
#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())
#endif

// Nombre de blocs pour couvrir n elements avec t threads par bloc :
// (n + t - 1) / t, c'est-a-dire ceil(n / t) en entiers (bug 12 du PSO corrige).
inline int nb_blocs(int n, int t) { return (n + t - 1) / t; }
