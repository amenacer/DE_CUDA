// de_cuda_v1.cu -- DE/rand/1/bin sur GPU, version 1 "naive" (issues #9 et #10)
//
// DE/rand/1/bin comme le DE sequentiel (src/seq/sde.cpp), memes fonctions
// (src/common/benchmarks.h), meme budget, meme format CSV. Difference : ici la
// population est remplacee en fin de generation (DE "generationnel", obligatoire
// en parallele), alors que sde.cpp remplace x_i des que l'essai est meilleur.
// Une generation =
// 3 kernels separes, un thread par individu :
//
//   k_mutation_croisement : trial[i] = croisement(pop[i], pop[r1] + F (pop[r2] - pop[r3]))
//   k_eval                : fit_trial[i] = f(trial[i])
//   k_selection           : si fit_trial[i] <= fit[i] alors pop[i] <- trial[i]
//
// Les essais sont ecrits dans un tableau separe (trial) et la selection n'a lieu
// qu'apres : aucun thread ne lit un individu qu'un autre est en train de remplacer.
// Aucun transfert CPU <-> GPU dans la boucle : le CPU ne fait que lancer les kernels
// et compter les evaluations.
//
// Usage :
//   ./de_cuda_v1 <fonction> <D> <N> <seed> [fichier.csv] [--trace] [--threads T]
//   ./de_cuda_v1 --test-erreur      (lancement volontairement faux -> message clair)
// Compilation (Colab, GPU T4) :
//   nvcc -O3 -arch=sm_75 -o de_cuda_v1 de_cuda_v1.cu
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <curand_kernel.h>
#include "../common/benchmarks.h"
#include "../common/results_csv.h"
#include "cuda_utils.h"

static const double DE_F  = 0.5;
static const double DE_CR = 0.3;
static const int    MAX_D = 1024;   // 8 Ko de memoire constante (limite : 64 Ko)

// Le decalage o est lu par tous les threads au meme indice j en meme temps :
// la memoire constante le diffuse en une seule lecture a tout le warp.
__constant__ double c_shift[MAX_D];

// Un generateur cuRAND par individu (Philox : initialisation rapide, sequences
// independantes). Meme graine -> memes nombres -> meme resultat a chaque run.
typedef curandStatePhilox4_32_10_t rng_t;

// --------------------------------------------------------------------- kernels
__global__ void k_init_rng(rng_t* rng, unsigned long long seed, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    curand_init(seed, (unsigned long long)i, 0ULL, &rng[i]);   // sous-sequence i
}

__global__ void k_init_pop(double* pop, rng_t* rng, int N, int D, double lo, double hi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    rng_t s = rng[i];                       // copie locale (registres), plus rapide
    for (int j = 0; j < D; j++)
        pop[i * D + j] = lo + (hi - lo) * curand_uniform_double(&s);
    rng[i] = s;                             // on sauvegarde l'etat pour la suite
}

__global__ void k_eval(const double* x, double* fit, int n, int D, int f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fit[i] = evaluate(f, &x[i * D], c_shift, D);
}

__device__ __forceinline__ int tirer_indice(rng_t* s, int N) {
    return (int)(curand(s) % (unsigned)N);  // biais du modulo < N / 2^32 : negligeable
}

__global__ void k_mutation_croisement(const double* pop, double* trial, rng_t* rng,
                                      int N, int D, double lo, double hi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    rng_t s = rng[i];
    int r1, r2, r3;                                  // 3 individus distincts, tous != i
    do { r1 = tirer_indice(&s, N); } while (r1 == i);
    do { r2 = tirer_indice(&s, N); } while (r2 == i || r2 == r1);
    do { r3 = tirer_indice(&s, N); } while (r3 == i || r3 == r1 || r3 == r2);
    int jrand = tirer_indice(&s, D);                 // au moins un gene vient du mutant
    for (int j = 0; j < D; j++) {
        double u;
        if (curand_uniform_double(&s) < DE_CR || j == jrand) {
            double v = pop[r1 * D + j] + DE_F * (pop[r2 * D + j] - pop[r3 * D + j]);
            u = fmin(fmax(v, lo), hi);               // on reste dans le domaine
        } else {
            u = pop[i * D + j];
        }
        trial[i * D + j] = u;
    }
    rng[i] = s;
}

__global__ void k_selection(double* pop, double* fit, const double* trial,
                            const double* fit_trial, int n, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (fit_trial[i] <= fit[i]) {                    // <= : on accepte un essai aussi bon
        for (int j = 0; j < D; j++) pop[i * D + j] = trial[i * D + j];
        fit[i] = fit_trial[i];
    }
}

// ---------------------------------------------------------------------- outils
static double min_of(const std::vector<double>& v) {
    double b = v[0];
    for (double x : v) if (x < b) b = x;
    return b;
}

static bool health_check(const std::vector<double>& pop, const std::vector<double>& fit,
                         int N, int D, double lo, double hi, long long fe, long long maxFE) {
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (!std::isfinite(fit[i])) { fprintf(stderr, "SANTE: fit[%d] non finie\n", i); ok = false; }
        for (int j = 0; j < D; j++) {
            double v = pop[(size_t)i * D + j];
            if (!std::isfinite(v) || v < lo || v > hi) {
                fprintf(stderr, "SANTE: pop[%d][%d]=%g hors de [%g, %g]\n", i, j, v, lo, hi);
                ok = false;
            }
        }
    }
    if (fe > maxFE) { fprintf(stderr, "SANTE: FE=%lld > maxFE=%lld\n", fe, maxFE); ok = false; }
    return ok;
}

// Lancement volontairement faux (2048 threads par bloc, maximum = 1024) :
// CUDA_CHECK_KERNEL doit l'attraper et afficher un message clair (critere #9).
static int test_erreur() {
    double* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, sizeof(double)));
    printf("Lancement de k_eval avec 2048 threads par bloc (maximum autorise : 1024)...\n");
    fflush(stdout);
    k_eval<<<1, 2048>>>(d, d, 1, 1, 0);
    CUDA_CHECK_KERNEL();                      // -> message clair (fichier, ligne, erreur CUDA)
    printf("ERREUR : le lancement faux n'a pas ete detecte\n");
    return 1;
}

static void usage(const char* p) {
    fprintf(stderr, "Usage: %s <fonction> <D> <N> <seed> [fichier.csv] [--trace] [--threads T]\n", p);
    fprintf(stderr, "       %s --test-erreur\n", p);
    fprintf(stderr, "  fonction in {sphere, rosenbrock, griewank, rastrigin}\n");
}

// ------------------------------------------------------------------------ main
int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--test-erreur") == 0) return test_erreur();
    if (argc < 5) { usage(argv[0]); return 1; }

    const int f = func_id(argv[1]);
    const int D = atoi(argv[2]);
    const int N = atoi(argv[3]);
    const unsigned seed = (unsigned)strtoul(argv[4], nullptr, 10);
    if (f < 0)  { fprintf(stderr, "Fonction inconnue : %s\n", argv[1]); return 1; }
    if (D < 1 || D > MAX_D || N < 4) {
        fprintf(stderr, "Il faut 1 <= D <= %d et N >= 4 (recu D=%d, N=%d)\n", MAX_D, D, N);
        return 1;
    }
    const char* csv = nullptr;
    bool trace = false;
    int T = 128;                                    // threads par bloc
    for (int a = 5; a < argc; a++) {
        if (!std::strcmp(argv[a], "--trace")) trace = true;
        else if (!std::strcmp(argv[a], "--threads") && a + 1 < argc) T = atoi(argv[++a]);
        else csv = argv[a];
    }

    if (T < 1 || T > 1024) {
        fprintf(stderr, "--threads doit etre entre 1 et 1024 (recu %d)\n", T);
        return 1;
    }

    const long long maxFE = max_fe(D);
    const double lo = lower_bound(f), hi = upper_bound(f);
    const std::vector<double> o = make_shift(f, D);
    const size_t ND = (size_t)N * D;

    // Le premier appel CUDA cree le contexte (~100 ms a 1 s) : on le fait AVANT
    // de chronometrer pour ne mesurer que l'algorithme.
    CUDA_CHECK(cudaFree(0));

    double *d_pop, *d_trial, *d_fit, *d_fit_t;
    rng_t* d_rng;
    CUDA_CHECK(cudaMalloc(&d_pop,   ND * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_trial, ND * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fit,   N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fit_t, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rng,   N * sizeof(rng_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_shift, o.data(), D * sizeof(double)));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    const int B = nb_blocs(N, T);
    std::vector<double> h_fit(N);

    // ------------------------------------------------ algorithme chronometre
    CUDA_CHECK(cudaEventRecord(e0));
    k_init_rng<<<B, T>>>(d_rng, seed, N);                 CUDA_CHECK_KERNEL();
    k_init_pop<<<B, T>>>(d_pop, d_rng, N, D, lo, hi);     CUDA_CHECK_KERNEL();
    k_eval<<<B, T>>>(d_pop, d_fit, N, D, f);              CUDA_CHECK_KERNEL();
    long long fe = N;                                     // l'initialisation compte
    long long gen = 0;
    if (trace) {
        CUDA_CHECK(cudaMemcpy(h_fit.data(), d_fit, N * sizeof(double), cudaMemcpyDeviceToHost));
        printf("TRACE gen=0 FE=%lld best_error=%.9e\n", fe, min_of(h_fit));
    }
    while (fe < maxFE) {
        // derniere generation eventuellement partielle : on n'evalue que ce que
        // le budget permet (ne se produit pas quand N divise 10^4 x D)
        const int n = (int)((maxFE - fe < N) ? (maxFE - fe) : N);
        const int Bn = nb_blocs(n, T);
        k_mutation_croisement<<<B, T>>>(d_pop, d_trial, d_rng, N, D, lo, hi);  CUDA_CHECK_KERNEL();
        k_eval<<<Bn, T>>>(d_trial, d_fit_t, n, D, f);                          CUDA_CHECK_KERNEL();
        k_selection<<<Bn, T>>>(d_pop, d_fit, d_trial, d_fit_t, n, D);          CUDA_CHECK_KERNEL();
        fe += n;
        gen++;
        if (trace) {   // copie a chaque generation : uniquement pour tracer, pas pour chronometrer
            CUDA_CHECK(cudaMemcpy(h_fit.data(), d_fit, N * sizeof(double), cudaMemcpyDeviceToHost));
            printf("TRACE gen=%lld FE=%lld best_error=%.9e\n", gen, fe, min_of(h_fit));
        }
    }
    // une seule copie GPU -> CPU : les N fitness, pour trouver la meilleure
    CUDA_CHECK(cudaMemcpy(h_fit.data(), d_fit, N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    // ------------------------------------------------------------------------
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    const double time_s = ms / 1000.0;
    const double best = min_of(h_fit);

    // controle de sante (hors chrono) : population finale dans le domaine, valeurs finies
    std::vector<double> h_pop(ND);
    CUDA_CHECK(cudaMemcpy(h_pop.data(), d_pop, ND * sizeof(double), cudaMemcpyDeviceToHost));
    const bool healthy = health_check(h_pop, h_fit, N, D, lo, hi, fe, maxFE);

    printf("algo=cuda_v1 func=%s D=%d N=%d seed=%u best_error=%.9e FE=%lld maxFE=%lld "
           "time_s=%.6f threads=%d sante=%s\n",
           func_name(f), D, N, seed, best, fe, maxFE, time_s, T, healthy ? "OK" : "ANOMALIE");
    if (csv && !append_result(csv, "cuda_v1", func_name(f), D, N, seed, best, time_s, fe)) {
        fprintf(stderr, "Impossible d'ecrire dans %s\n", csv);
        return 1;
    }

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(d_pop));   CUDA_CHECK(cudaFree(d_trial));
    CUDA_CHECK(cudaFree(d_fit));   CUDA_CHECK(cudaFree(d_fit_t));
    CUDA_CHECK(cudaFree(d_rng));
    return healthy ? 0 : 2;
}
