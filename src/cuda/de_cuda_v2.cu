// de_cuda_v2.cu -- DE/rand/1/bin sur GPU, version 2 optimisee (issues #12 et #13)
//
// Ce qui change par rapport a la v1 (src/cuda/de_cuda_v1.cu) :
//
//  1. Kernel P (indices) : un thread par individu tire r1, r2, r3 et jrand et les
//     range dans une matrice idx[N] (int4). Le travail aleatoire "par individu"
//     est separe du travail "par gene".
//  2. Kernel fusionne MCER (Mutation, Croisement, Evaluation, Remplacement) :
//     UN SEUL kernel par generation fait tout le reste. Un bloc par individu,
//     un thread par gene (boucle si D > threads) :
//       - l'essai u est construit en memoire partagee (rapide, jamais ecrit en
//         memoire globale) ;
//       - f(u) est calculee par une reduction parallele dans le bloc ;
//       - le gagnant (u ou x_i) est ecrit dans un SECOND tableau pop_out.
//     Remarque : les generateurs "par gene" sont indexes par i*T + t ; le resultat
//     d'une graine est donc reproductible pour un T donne (pas d'un T a l'autre).
//     Double tampon : on lit pop_in, on ecrit pop_out, puis on echange les deux
//     pointeurs. Indispensable : un bloc lit x_r1, x_r2, x_r3 pendant que d'autres
//     blocs ecrivent leur gagnant ; ecrire dans pop_in creerait une race condition.
//     Les threads d'un bloc lisent des genes consecutifs : acces memoire coalesces
//     (en v1, un thread par individu lisait avec un pas de D).
//  3. Le meilleur individu est trouve sur GPU par une reduction argmin : une seule
//     copie de 16 octets (struct Best : double + int + alignement) vers le CPU a la fin.
//  4. Configuration automatique du nombre de threads par bloc selon D, N et le
//     nombre de SM (regle mesuree, voir threads_auto) ; --occupation affiche
//     l'occupation theorique de chaque choix (cudaOccupancyMaxActiveBlocksPerMultiprocessor).
//
// Par generation : 2 lancements de kernel (v1 : 3), 0 transfert CPU <-> GPU.
//
// Usage :
//   ./de_cuda_v2 <fonction> <D> <N> <seed> [fichier.csv] [--trace] [--threads T]
//   ./de_cuda_v2 --occupation <D>
// Compilation (Colab, GPU T4) :
//   nvcc -O3 -arch=sm_75 -o de_cuda_v2 de_cuda_v2.cu
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
static const int    MAX_D = 1024;

__constant__ double c_shift[MAX_D];
typedef curandStatePhilox4_32_10_t rng_t;

struct Best { double val; int idx; };

// Memoire partagee dynamique du kernel MCER : essai (D) + 2 tampons de reduction (T).
static size_t shmem_mcer(int D, int T) { return (size_t)(D + 2 * T) * sizeof(double); }

// ------------------------------------------------------------ generateurs
// Sous-sequences : [0, N) pour les individus (kernel P), N + i*T + t pour les genes.
__global__ void k_init_rng(rng_t* rng_ind, rng_t* rng_gene, unsigned long long seed, int N, int T) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < N) curand_init(seed, (unsigned long long)g, 0ULL, &rng_ind[g]);
    if (g < N * T) curand_init(seed, (unsigned long long)(N + g), 0ULL, &rng_gene[g]);
}

__global__ void k_init_pop(double* pop, rng_t* rng_gene, int N, int D, int T, double lo, double hi) {
    int i = blockIdx.x, t = threadIdx.x;              // un bloc par individu
    rng_t s = rng_gene[i * T + t];
    for (int j = t; j < D; j += T)
        pop[(size_t)i * D + j] = lo + (hi - lo) * curand_uniform_double(&s);
    rng_gene[i * T + t] = s;
}

// ------------------------------------------------------------ evaluation par bloc
// Contribution du gene j a f(x). La somme de ces termes redonne exactement les
// formules de benchmarks.h (seul l'ordre des additions change).
__device__ __forceinline__ void terme(int f, const double* x, int j, int D,
                                      double& somme, double& produit) {
    const double z = x[j] - c_shift[j];
    switch (f) {
        case F_SPHERE:    somme += z * z; break;
        case F_RASTRIGIN: somme += z * z - 10.0 * cos(2.0 * DE_PI * z) + 10.0; break;
        case F_GRIEWANK:  somme += z * z / 4000.0;
                          produit *= cos(z / sqrt((double)(j + 1))); break;
        default: /* rosenbrock : couple (j, j+1), pour j < D-1 */
            if (j < D - 1) {
                const double zi = z + 1.0, zi1 = x[j + 1] - c_shift[j + 1] + 1.0;
                const double a = zi * zi - zi1, b = zi - 1.0;
                somme += 100.0 * a * a + b * b;
            }
    }
}

// f(x) calculee par les T threads du bloc (T puissance de 2). Tous les threads
// doivent appeler cette fonction ; tous recoivent le resultat.
__device__ double eval_bloc(int f, const double* x, int D, double* s_sum, double* s_prod) {
    const int t = threadIdx.x, T = blockDim.x;
    double somme = 0.0, produit = 1.0;
    for (int j = t; j < D; j += T) terme(f, x, j, D, somme, produit);
    s_sum[t] = somme;
    s_prod[t] = produit;
    __syncthreads();
    for (int pas = T / 2; pas > 0; pas >>= 1) {       // reduction en arbre : log2(T) etapes
        if (t < pas) {
            s_sum[t]  += s_sum[t + pas];
            s_prod[t] *= s_prod[t + pas];
        }
        __syncthreads();
    }
    const double r = (f == F_GRIEWANK) ? s_sum[0] - s_prod[0] + 1.0 : s_sum[0];
    __syncthreads();                                  // s_sum peut etre reutilise ensuite
    return r;
}

__global__ void k_eval_pop(const double* pop, double* fit, int D, int f) {
    extern __shared__ double sh[];
    const int i = blockIdx.x;
    double r = eval_bloc(f, &pop[(size_t)i * D], D, sh, sh + blockDim.x);
    if (threadIdx.x == 0) fit[i] = r;
}

// ------------------------------------------------------------ kernel P : indices
__device__ __forceinline__ int tirer_indice(rng_t* s, int N) {
    return (int)(curand(s) % (unsigned)N);
}

__global__ void k_P(int4* idx, rng_t* rng_ind, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    rng_t s = rng_ind[i];
    int r1, r2, r3;
    do { r1 = tirer_indice(&s, N); } while (r1 == i);
    do { r2 = tirer_indice(&s, N); } while (r2 == i || r2 == r1);
    do { r3 = tirer_indice(&s, N); } while (r3 == i || r3 == r1 || r3 == r2);
    idx[i] = make_int4(r1, r2, r3, tirer_indice(&s, D));   // w = jrand
    rng_ind[i] = s;
}

// ------------------------------------------------------------ kernel fusionne MCER
__global__ void k_MCER(const double* __restrict__ pop_in, const double* __restrict__ fit_in,
                       double* __restrict__ pop_out, double* __restrict__ fit_out,
                       const int4* __restrict__ idx, rng_t* rng_gene,
                       int n_actifs, int D, int f, double lo, double hi) {
    extern __shared__ double sh[];
    double* s_trial = sh;                          // D doubles
    double* s_sum   = sh + D;                      // T doubles
    double* s_prod  = sh + D + blockDim.x;         // T doubles
    __shared__ int s_gagne;

    const int i = blockIdx.x, t = threadIdx.x, T = blockDim.x;
    const double* xi = pop_in + (size_t)i * D;
    double* yi = pop_out + (size_t)i * D;

    if (i >= n_actifs) {                           // budget epuise pour cet individu :
        for (int j = t; j < D; j += T) yi[j] = xi[j];   // on le recopie tel quel
        if (t == 0) fit_out[i] = fit_in[i];
        return;                                    // (tout le bloc sort ensemble)
    }

    // M + C : mutation et croisement, un gene par thread, essai en memoire partagee
    const int4 r = idx[i];
    const double* x1 = pop_in + (size_t)r.x * D;
    const double* x2 = pop_in + (size_t)r.y * D;
    const double* x3 = pop_in + (size_t)r.z * D;
    rng_t s = rng_gene[i * T + t];
    for (int j = t; j < D; j += T) {
        double u;
        if (curand_uniform_double(&s) < DE_CR || j == r.w) {
            const double v = x1[j] + DE_F * (x2[j] - x3[j]);
            u = fmin(fmax(v, lo), hi);
        } else {
            u = xi[j];
        }
        s_trial[j] = u;
    }
    rng_gene[i * T + t] = s;
    __syncthreads();

    // E : evaluation de l'essai par reduction parallele
    const double fu = eval_bloc(f, s_trial, D, s_sum, s_prod);

    // R : remplacement, ecrit dans le second tampon
    if (t == 0) {
        s_gagne = (fu <= fit_in[i]);
        fit_out[i] = s_gagne ? fu : fit_in[i];
    }
    __syncthreads();
    const bool g = s_gagne;
    for (int j = t; j < D; j += T) yi[j] = g ? s_trial[j] : xi[j];
}

// ------------------------------------------------------------ argmin sur GPU
// Un seul bloc de 512 threads : chaque thread parcourt N/512 valeurs, puis
// reduction en arbre. A egalite, le plus petit indice gagne (resultat deterministe).
__global__ void k_argmin(const double* fit, int N, Best* out) {
    __shared__ double sv[512];
    __shared__ int si[512];
    const int t = threadIdx.x;
    double bv = INFINITY; int bi = -1;
    for (int k = t; k < N; k += blockDim.x)
        if (fit[k] < bv) { bv = fit[k]; bi = k; }
    sv[t] = bv; si[t] = bi;
    __syncthreads();
    for (int pas = blockDim.x / 2; pas > 0; pas >>= 1) {
        if (t < pas) {
            const double v2 = sv[t + pas]; const int i2 = si[t + pas];
            if (i2 >= 0 && (v2 < sv[t] || (v2 == sv[t] && i2 < si[t]) || si[t] < 0)) {
                sv[t] = v2; si[t] = i2;
            }
        }
        __syncthreads();
    }
    if (t == 0) { out->val = sv[0]; out->idx = si[0]; }
}

// ------------------------------------------------------------ configuration
// Regle fixee par le balayage mesure sur le T4 (results/taille_bloc_v2.csv) :
//  - D <= 32 : un warp (32 threads) couvre tous les genes ;
//  - peu de blocs (N < 4 x nombre de SM) : le GPU n'est pas rempli, 64 threads
//    par bloc donnent plus de parallelisme dans chaque bloc ;
//  - beaucoup de blocs (N >= 4 x SM) : le GPU est deja rempli, 32 threads par bloc
//    limitent le cout des __syncthreads et de la reduction (log2 T etapes).
// Les blocs plus gros (128, 256) sont toujours plus lents que le meilleur choix entre
// 32 et 64, meme a 100 % d'occupation. Limites : regle ajustee sur 12 cas (D = 50/100,
// N = 50/100/500) ; le seuil 4 x SM (160 sur un T4) est seulement situe entre N = 100 et 500.
static int threads_auto(int D, int N, int nb_sm) {
    if (D <= 32 || N >= 4 * nb_sm) return 32;
    return 64;
}

static int nb_sm() {
    int dev = 0, n = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
    return n;
}

static int occupation(int D) {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("GPU : %s, %d SM, %d threads max par SM, %zu octets de memoire partagee par bloc\n",
           p.name, p.multiProcessorCount, p.maxThreadsPerMultiProcessor, p.sharedMemPerBlock);
    printf("Kernel k_MCER, D = %d (choix automatique : %d threads si N < %d, %d threads sinon)\n",
           D, threads_auto(D, 1, p.multiProcessorCount), 4 * p.multiProcessorCount,
           threads_auto(D, 4 * p.multiProcessorCount, p.multiProcessorCount));
    printf("threads/bloc,shmem_octets,blocs_par_SM,warps_actifs_par_SM,occupation_%%,threads_utiles_%%\n");
    for (int T = 32; T <= 512; T *= 2) {
        int nb = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_MCER, T, shmem_mcer(D, T)));
        const double occ = 100.0 * nb * T / p.maxThreadsPerMultiProcessor;
        // part des threads qui ont un gene a traiter au premier tour de boucle
        const double utiles = 100.0 * (D < T ? D : T) / T;
        printf("%d,%zu,%d,%d,%.1f,%.1f\n", T, shmem_mcer(D, T), nb, nb * T / 32, occ, utiles);
    }
    return 0;
}

// ------------------------------------------------------------ sante
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

static void usage(const char* p) {
    fprintf(stderr, "Usage: %s <fonction> <D> <N> <seed> [fichier.csv] [--trace] [--threads T]\n", p);
    fprintf(stderr, "       %s --occupation <D>\n", p);
}

// ------------------------------------------------------------ main
int main(int argc, char** argv) {
    if (argc >= 3 && !std::strcmp(argv[1], "--occupation")) return occupation(atoi(argv[2]));
    if (argc < 5) { usage(argv[0]); return 1; }

    const int f = func_id(argv[1]);
    const int D = atoi(argv[2]);
    const int N = atoi(argv[3]);
    const unsigned seed = (unsigned)strtoul(argv[4], nullptr, 10);
    if (f < 0) { fprintf(stderr, "Fonction inconnue : %s\n", argv[1]); return 1; }
    if (D < 1 || D > MAX_D || N < 4) {
        fprintf(stderr, "Il faut 1 <= D <= %d et N >= 4 (recu D=%d, N=%d)\n", MAX_D, D, N);
        return 1;
    }
    const char* csv = nullptr;
    bool trace = false;
    int T = -1;                                  // -1 : choix automatique
    for (int a = 5; a < argc; a++) {
        if (!std::strcmp(argv[a], "--trace")) trace = true;
        else if (!std::strcmp(argv[a], "--threads") && a + 1 < argc) T = atoi(argv[++a]);
        else csv = argv[a];
    }
    if (T < 0) T = threads_auto(D, N, nb_sm());
    if (T < 32 || T > 512 || (T & (T - 1))) {
        fprintf(stderr, "--threads doit etre une puissance de 2 entre 32 et 512 (recu %d)\n", T);
        return 1;
    }

    const long long maxFE = max_fe(D);
    const double lo = lower_bound(f), hi = upper_bound(f);
    const std::vector<double> o = make_shift(f, D);
    const size_t ND = (size_t)N * D;
    const size_t sh = shmem_mcer(D, T);

    CUDA_CHECK(cudaFree(0));                     // contexte cree hors chrono

    double *d_popA, *d_popB, *d_fitA, *d_fitB;
    int4* d_idx; rng_t *d_rng_ind, *d_rng_gene; Best* d_best;
    CUDA_CHECK(cudaMalloc(&d_popA, ND * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_popB, ND * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fitA, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fitB, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_idx, N * sizeof(int4)));
    CUDA_CHECK(cudaMalloc(&d_rng_ind, N * sizeof(rng_t)));
    CUDA_CHECK(cudaMalloc(&d_rng_gene, (size_t)N * T * sizeof(rng_t)));
    CUDA_CHECK(cudaMalloc(&d_best, sizeof(Best)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_shift, o.data(), D * sizeof(double)));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    const int TP = 128;                           // kernel P et init : un thread par element
    Best h_best;

    // ------------------------------------------------ algorithme chronometre
    CUDA_CHECK(cudaEventRecord(e0));
    k_init_rng<<<nb_blocs(N * T, TP), TP>>>(d_rng_ind, d_rng_gene, seed, N, T);  CUDA_CHECK_KERNEL();
    k_init_pop<<<N, T>>>(d_popA, d_rng_gene, N, D, T, lo, hi);                    CUDA_CHECK_KERNEL();
    k_eval_pop<<<N, T, 2 * T * sizeof(double)>>>(d_popA, d_fitA, D, f);           CUDA_CHECK_KERNEL();
    long long fe = N, gen = 0;
    double *pin = d_popA, *pout = d_popB, *fin = d_fitA, *fout = d_fitB;
    if (trace) {
        k_argmin<<<1, 512>>>(fin, N, d_best); CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(&h_best, d_best, sizeof(Best), cudaMemcpyDeviceToHost));
        printf("TRACE gen=0 FE=%lld best_error=%.9e\n", fe, h_best.val);
    }
    while (fe < maxFE) {
        const int n = (int)((maxFE - fe < N) ? (maxFE - fe) : N);
        k_P<<<nb_blocs(N, TP), TP>>>(d_idx, d_rng_ind, N, D);                          CUDA_CHECK_KERNEL();
        k_MCER<<<N, T, sh>>>(pin, fin, pout, fout, d_idx, d_rng_gene, n, D, f, lo, hi); CUDA_CHECK_KERNEL();
        std::swap(pin, pout);                     // le tampon ecrit devient la population courante
        std::swap(fin, fout);
        fe += n;
        gen++;
        if (trace) {
            k_argmin<<<1, 512>>>(fin, N, d_best); CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaMemcpy(&h_best, d_best, sizeof(Best), cudaMemcpyDeviceToHost));
            printf("TRACE gen=%lld FE=%lld best_error=%.9e\n", gen, fe, h_best.val);
        }
    }
    k_argmin<<<1, 512>>>(fin, N, d_best);                                          CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(&h_best, d_best, sizeof(Best), cudaMemcpyDeviceToHost)); // 16 octets
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    // ------------------------------------------------------------------------
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    const double time_s = ms / 1000.0;

    std::vector<double> h_pop(ND), h_fit(N);
    CUDA_CHECK(cudaMemcpy(h_pop.data(), pin, ND * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fit.data(), fin, N * sizeof(double), cudaMemcpyDeviceToHost));
    bool healthy = health_check(h_pop, h_fit, N, D, lo, hi, fe, maxFE);
    // controle de l'argmin GPU contre un min calcule sur CPU
    double cpu_min = h_fit[0];
    for (double v : h_fit) if (v < cpu_min) cpu_min = v;
    if (cpu_min != h_best.val) {
        fprintf(stderr, "SANTE: argmin GPU %.17g != min CPU %.17g\n", h_best.val, cpu_min);
        healthy = false;
    }
    if (h_best.idx < 0 || h_best.idx >= N) {      // toutes les fitness non finies
        fprintf(stderr, "SANTE: argmin GPU invalide (indice %d)\n", h_best.idx);
        return 2;
    }
    // controle de la reduction parallele : f(meilleur) recalculee avec evaluate()
    const double ref = evaluate(f, &h_pop[(size_t)h_best.idx * D], o.data(), D);
    if (std::fabs(ref - h_best.val) > 1e-9 * std::fmax(1.0, std::fabs(ref))) {
        fprintf(stderr, "SANTE: f GPU %.17g != f CPU %.17g\n", h_best.val, ref);
        healthy = false;
    }

    printf("algo=cuda_v2 func=%s D=%d N=%d seed=%u best_error=%.9e FE=%lld maxFE=%lld "
           "time_s=%.6f threads=%d sante=%s\n",
           func_name(f), D, N, seed, h_best.val, fe, maxFE, time_s, T, healthy ? "OK" : "ANOMALIE");
    if (csv && !append_result(csv, "cuda_v2", func_name(f), D, N, seed, h_best.val, time_s, fe)) {
        fprintf(stderr, "Impossible d'ecrire dans %s\n", csv);
        return 1;
    }

    CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(d_popA)); CUDA_CHECK(cudaFree(d_popB));
    CUDA_CHECK(cudaFree(d_fitA)); CUDA_CHECK(cudaFree(d_fitB));
    CUDA_CHECK(cudaFree(d_idx));  CUDA_CHECK(cudaFree(d_rng_ind));
    CUDA_CHECK(cudaFree(d_rng_gene)); CUDA_CHECK(cudaFree(d_best));
    return healthy ? 0 : 2;
}
