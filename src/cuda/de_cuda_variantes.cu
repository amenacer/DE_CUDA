// de_cuda_variantes.cu -- variantes du DE sur GPU (issue #17, partie GPU)
//
// Meme organisation que la v2 (src/cuda/de_cuda_v2.cu) : kernel P (tirages par
// individu), kernel fusionne MCER (un bloc par individu, un thread par gene,
// essai en memoire partagee, f par reduction, double tampon), argmin sur GPU.
// La v2 n'est pas modifiee : ses resultats restent reproductibles.
//
// Strategies (--strategie) :
//   rand1 : DE/rand/1/bin         v = x_r1 + F (x_r2 - x_r3)               (= v2)
//   ctb1  : DE/current-to-best/1/bin
//           v = x_i + F (x_best - x_i) + F (x_r1 - x_r2)
//           x_best = meilleur individu de la generation courante, trouve sur GPU
//           par k_argmin AVANT le kernel MCER (un lancement de plus par generation).
//   jde   : jDE (Brest et al., IEEE TEC 2006), DE/rand/1/bin auto-adaptatif.
//           Chaque individu porte ses propres F_i et CR_i (initialises a 0,5 et 0,9).
//           A chaque generation, le kernel P propose :
//             F'  = 0,1 + 0,9 rand   avec probabilite tau1 = 0,1, sinon F' = F_i
//             CR' = rand             avec probabilite tau2 = 0,1, sinon CR' = CR_i
//           (rand = curand_uniform_double, dans ]0, 1] : F' dans ]0,1 ; 1], CR' dans ]0 ; 1])
//           L'essai est construit avec (F', CR'). S'il remplace x_i, (F', CR') sont
//           gardes ; sinon on garde (F_i, CR_i). F et CR sont donc aussi en double tampon.
// --F et --CR fixent F et CR pour rand1 et ctb1 (defaut 0,5 et 0,3 : parametres du projet).
//
// Usage :
//   ./de_cuda_variantes <fonction> <D> <N> <seed> [fichier.csv]
//        [--strategie rand1|ctb1|jde] [--F x] [--CR x] [--trace] [--threads T]
// Le nom d'algorithme ecrit dans le CSV decrit la variante, par exemple
//   gpu_rand1_F0.5_CR0.3, gpu_ctb1_F0.5_CR0.3, gpu_jde.
// Compilation (Colab, GPU T4) :
//   nvcc -O3 -arch=sm_75 -o de_cuda_variantes de_cuda_variantes.cu
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <curand_kernel.h>
#include "../common/benchmarks.h"
#include "../common/results_csv.h"
#include "cuda_utils.h"

static const int MAX_D = 1024;
enum Strategie { S_RAND1 = 0, S_CTB1 = 1, S_JDE = 2 };

// parametres de jDE (valeurs de l'article de Brest et al.)
static const double JDE_TAU1 = 0.1, JDE_TAU2 = 0.1, JDE_FL = 0.1, JDE_FU = 0.9;
static const double JDE_F0 = 0.5, JDE_CR0 = 0.9;

__constant__ double c_shift[MAX_D];
typedef curandStatePhilox4_32_10_t rng_t;

struct Best { double val; int idx; };

static size_t shmem_mcer(int D, int T) { return (size_t)(D + 2 * T) * sizeof(double); }

// ------------------------------------------------------------ generateurs (identique v2)
__global__ void k_init_rng(rng_t* rng_ind, rng_t* rng_gene, unsigned long long seed, int N, int T) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < N) curand_init(seed, (unsigned long long)g, 0ULL, &rng_ind[g]);
    if (g < N * T) curand_init(seed, (unsigned long long)(N + g), 0ULL, &rng_gene[g]);
}

__global__ void k_init_pop(double* pop, rng_t* rng_gene, int N, int D, int T, double lo, double hi) {
    int i = blockIdx.x, t = threadIdx.x;
    rng_t s = rng_gene[i * T + t];
    for (int j = t; j < D; j += T)
        pop[(size_t)i * D + j] = lo + (hi - lo) * curand_uniform_double(&s);
    rng_gene[i * T + t] = s;
}

// jDE : F_i = 0,5 et CR_i = 0,9 au depart
__global__ void k_init_param(double* F, double* CR, int N, double F0, double CR0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { F[i] = F0; CR[i] = CR0; }
}

// ------------------------------------------------------------ evaluation par bloc (identique v2)
__device__ __forceinline__ void terme(int f, const double* x, int j, int D,
                                      double& somme, double& produit) {
    const double z = x[j] - c_shift[j];
    switch (f) {
        case F_SPHERE:    somme += z * z; break;
        case F_RASTRIGIN: somme += z * z - 10.0 * cos(2.0 * DE_PI * z) + 10.0; break;
        case F_GRIEWANK:  somme += z * z / 4000.0;
                          produit *= cos(z / sqrt((double)(j + 1))); break;
        default:
            if (j < D - 1) {
                const double zi = z + 1.0, zi1 = x[j + 1] - c_shift[j + 1] + 1.0;
                const double a = zi * zi - zi1, b = zi - 1.0;
                somme += 100.0 * a * a + b * b;
            }
    }
}

__device__ double eval_bloc(int f, const double* x, int D, double* s_sum, double* s_prod) {
    const int t = threadIdx.x, T = blockDim.x;
    double somme = 0.0, produit = 1.0;
    for (int j = t; j < D; j += T) terme(f, x, j, D, somme, produit);
    s_sum[t] = somme;
    s_prod[t] = produit;
    __syncthreads();
    for (int pas = T / 2; pas > 0; pas >>= 1) {
        if (t < pas) {
            s_sum[t]  += s_sum[t + pas];
            s_prod[t] *= s_prod[t + pas];
        }
        __syncthreads();
    }
    const double r = (f == F_GRIEWANK) ? s_sum[0] - s_prod[0] + 1.0 : s_sum[0];
    __syncthreads();
    return r;
}

__global__ void k_eval_pop(const double* pop, double* fit, int D, int f) {
    extern __shared__ double sh[];
    const int i = blockIdx.x;
    double r = eval_bloc(f, &pop[(size_t)i * D], D, sh, sh + blockDim.x);
    if (threadIdx.x == 0) fit[i] = r;
}

// ------------------------------------------------------------ kernel P : indices (+ F', CR' pour jDE)
__device__ __forceinline__ int tirer_indice(rng_t* s, int N) {
    return (int)(curand(s) % (unsigned)N);
}

__global__ void k_P(int4* idx, rng_t* rng_ind, int N, int D, int strat,
                    const double* F_in, const double* CR_in, double* F_tr, double* CR_tr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    rng_t s = rng_ind[i];
    int r1, r2, r3;
    do { r1 = tirer_indice(&s, N); } while (r1 == i);
    do { r2 = tirer_indice(&s, N); } while (r2 == i || r2 == r1);
    do { r3 = tirer_indice(&s, N); } while (r3 == i || r3 == r1 || r3 == r2);
    idx[i] = make_int4(r1, r2, r3, tirer_indice(&s, D));   // w = jrand
    if (strat == S_JDE) {
        const double a = curand_uniform_double(&s), b = curand_uniform_double(&s);
        const double c = curand_uniform_double(&s), d = curand_uniform_double(&s);
        F_tr[i]  = (b < JDE_TAU1) ? JDE_FL + JDE_FU * a : F_in[i];
        CR_tr[i] = (d < JDE_TAU2) ? c : CR_in[i];
    }
    rng_ind[i] = s;
}

// ------------------------------------------------------------ kernel fusionne MCER
// strat = S_RAND1 / S_JDE : v = x_r1 + F (x_r2 - x_r3)
// strat = S_CTB1          : v = x_i + F (x_best - x_i) + F (x_r1 - x_r2)
// F et CR : constantes (rand1, ctb1) ou F_tr[i], CR_tr[i] (jDE).
__global__ void k_MCER(const double* __restrict__ pop_in, const double* __restrict__ fit_in,
                       double* __restrict__ pop_out, double* __restrict__ fit_out,
                       const int4* __restrict__ idx, rng_t* rng_gene,
                       int n_actifs, int D, int f, double lo, double hi,
                       int strat, double F, double CR, const Best* __restrict__ best,
                       const double* F_in, const double* CR_in,
                       const double* F_tr, const double* CR_tr,
                       double* F_out, double* CR_out) {
    extern __shared__ double sh[];
    double* s_trial = sh;
    double* s_sum   = sh + D;
    double* s_prod  = sh + D + blockDim.x;
    __shared__ int s_gagne;

    const int i = blockIdx.x, t = threadIdx.x, T = blockDim.x;
    const double* xi = pop_in + (size_t)i * D;
    double* yi = pop_out + (size_t)i * D;
    const bool jde = (strat == S_JDE);

    if (i >= n_actifs) {                           // budget epuise : recopie
        for (int j = t; j < D; j += T) yi[j] = xi[j];
        if (t == 0) {
            fit_out[i] = fit_in[i];
            if (jde) { F_out[i] = F_in[i]; CR_out[i] = CR_in[i]; }
        }
        return;
    }
    if (jde) { F = F_tr[i]; CR = CR_tr[i]; }

    const int4 r = idx[i];
    const double* x1 = pop_in + (size_t)r.x * D;
    const double* x2 = pop_in + (size_t)r.y * D;
    const double* x3 = pop_in + (size_t)r.z * D;
    // ctb1 : si toutes les fitness etaient non finies, argmin renvoie -1 ; on retombe
    // alors sur x_i (le controle de sante final signalera l'anomalie).
    const double* xb = nullptr;
    if (strat == S_CTB1) xb = (best->idx >= 0) ? pop_in + (size_t)best->idx * D : xi;
    rng_t s = rng_gene[i * T + t];
    for (int j = t; j < D; j += T) {
        double u;
        if (curand_uniform_double(&s) < CR || j == r.w) {
            const double v = (strat == S_CTB1)
                ? xi[j] + F * (xb[j] - xi[j]) + F * (x1[j] - x2[j])
                : x1[j] + F * (x2[j] - x3[j]);
            u = fmin(fmax(v, lo), hi);
        } else {
            u = xi[j];
        }
        s_trial[j] = u;
    }
    rng_gene[i * T + t] = s;
    __syncthreads();

    const double fu = eval_bloc(f, s_trial, D, s_sum, s_prod);

    if (t == 0) {
        s_gagne = (fu <= fit_in[i]);
        fit_out[i] = s_gagne ? fu : fit_in[i];
        if (jde) {                                 // jDE : les parametres gagnants survivent
            F_out[i]  = s_gagne ? F_tr[i]  : F_in[i];
            CR_out[i] = s_gagne ? CR_tr[i] : CR_in[i];
        }
    }
    __syncthreads();
    const bool g = s_gagne;
    for (int j = t; j < D; j += T) yi[j] = g ? s_trial[j] : xi[j];
}

// ------------------------------------------------------------ argmin sur GPU (identique v2)
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

// ------------------------------------------------------------ configuration (identique v2)
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
    fprintf(stderr, "Usage: %s <fonction> <D> <N> <seed> [fichier.csv] "
                    "[--strategie rand1|ctb1|jde] [--F x] [--CR x] [--trace] [--threads T]\n", p);
}

// ------------------------------------------------------------ main
int main(int argc, char** argv) {
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
    int T = -1, strat = S_RAND1;
    double F = 0.5, CR = 0.3;
    bool F_donne = false, CR_donne = false;
    for (int a = 5; a < argc; a++) {
        if (!std::strcmp(argv[a], "--trace")) trace = true;
        else if (!std::strcmp(argv[a], "--threads") && a + 1 < argc) T = atoi(argv[++a]);
        else if (!std::strcmp(argv[a], "--F") && a + 1 < argc) { F = atof(argv[++a]); F_donne = true; }
        else if (!std::strcmp(argv[a], "--CR") && a + 1 < argc) { CR = atof(argv[++a]); CR_donne = true; }
        else if (!std::strcmp(argv[a], "--strategie") && a + 1 < argc) {
            const char* s = argv[++a];
            if      (!std::strcmp(s, "rand1")) strat = S_RAND1;
            else if (!std::strcmp(s, "ctb1"))  strat = S_CTB1;
            else if (!std::strcmp(s, "jde"))   strat = S_JDE;
            else { fprintf(stderr, "Strategie inconnue : %s\n", s); return 1; }
        }
        else if (argv[a][0] == '-') { fprintf(stderr, "Option inconnue : %s\n", argv[a]); usage(argv[0]); return 1; }
        else if (csv) { fprintf(stderr, "Argument en trop : %s\n", argv[a]); usage(argv[0]); return 1; }
        else csv = argv[a];
    }
    if (strat == S_JDE && (F_donne || CR_donne)) {
        fprintf(stderr, "jde adapte F et CR lui-meme : --F et --CR ne sont pas acceptes avec --strategie jde\n");
        return 1;
    }
    if (!(F > 0.0 && F <= 2.0) || !(CR >= 0.0 && CR <= 1.0)) {
        fprintf(stderr, "Il faut 0 < F <= 2 et 0 <= CR <= 1 (recu F=%g, CR=%g)\n", F, CR);
        return 1;
    }
    if (T < 0) T = threads_auto(D, N, nb_sm());
    if (T < 32 || T > 512 || (T & (T - 1))) {
        fprintf(stderr, "--threads doit etre une puissance de 2 entre 32 et 512 (recu %d)\n", T);
        return 1;
    }
    char algo[64];
    if (strat == S_JDE) snprintf(algo, sizeof algo, "gpu_jde");
    else snprintf(algo, sizeof algo, "gpu_%s_F%g_CR%g", strat == S_CTB1 ? "ctb1" : "rand1", F, CR);

    const long long maxFE = max_fe(D);
    const double lo = lower_bound(f), hi = upper_bound(f);
    const std::vector<double> o = make_shift(f, D);
    const size_t ND = (size_t)N * D;
    const size_t sh = shmem_mcer(D, T);

    CUDA_CHECK(cudaFree(0));

    double *d_popA, *d_popB, *d_fitA, *d_fitB;
    double *d_FA = nullptr, *d_FB = nullptr, *d_CRA = nullptr, *d_CRB = nullptr, *d_Ftr = nullptr, *d_CRtr = nullptr;
    int4* d_idx; rng_t *d_rng_ind, *d_rng_gene; Best* d_best;
    CUDA_CHECK(cudaMalloc(&d_popA, ND * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_popB, ND * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fitA, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fitB, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_idx, N * sizeof(int4)));
    CUDA_CHECK(cudaMalloc(&d_rng_ind, N * sizeof(rng_t)));
    CUDA_CHECK(cudaMalloc(&d_rng_gene, (size_t)N * T * sizeof(rng_t)));
    CUDA_CHECK(cudaMalloc(&d_best, sizeof(Best)));
    if (strat == S_JDE) {
        CUDA_CHECK(cudaMalloc(&d_FA, N * sizeof(double)));  CUDA_CHECK(cudaMalloc(&d_FB, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_CRA, N * sizeof(double))); CUDA_CHECK(cudaMalloc(&d_CRB, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Ftr, N * sizeof(double))); CUDA_CHECK(cudaMalloc(&d_CRtr, N * sizeof(double)));
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_shift, o.data(), D * sizeof(double)));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    const int TP = 128;
    Best h_best;

    // ------------------------------------------------ algorithme chronometre
    CUDA_CHECK(cudaEventRecord(e0));
    k_init_rng<<<nb_blocs(N * T, TP), TP>>>(d_rng_ind, d_rng_gene, seed, N, T);  CUDA_CHECK_KERNEL();
    k_init_pop<<<N, T>>>(d_popA, d_rng_gene, N, D, T, lo, hi);                    CUDA_CHECK_KERNEL();
    k_eval_pop<<<N, T, 2 * T * sizeof(double)>>>(d_popA, d_fitA, D, f);           CUDA_CHECK_KERNEL();
    if (strat == S_JDE) {
        k_init_param<<<nb_blocs(N, TP), TP>>>(d_FA, d_CRA, N, JDE_F0, JDE_CR0);   CUDA_CHECK_KERNEL();
    }
    long long fe = N, gen = 0;
    double *pin = d_popA, *pout = d_popB, *fin = d_fitA, *fout = d_fitB;
    double *Fin = d_FA, *Fout = d_FB, *CRin = d_CRA, *CRout = d_CRB;
    if (trace) {
        k_argmin<<<1, 512>>>(fin, N, d_best); CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaMemcpy(&h_best, d_best, sizeof(Best), cudaMemcpyDeviceToHost));
        printf("TRACE gen=0 FE=%lld best_error=%.9e\n", fe, h_best.val);
    }
    while (fe < maxFE) {
        const int n = (int)((maxFE - fe < N) ? (maxFE - fe) : N);
        if (strat == S_CTB1) { k_argmin<<<1, 512>>>(fin, N, d_best); CUDA_CHECK_KERNEL(); }
        k_P<<<nb_blocs(N, TP), TP>>>(d_idx, d_rng_ind, N, D, strat, Fin, CRin, d_Ftr, d_CRtr); CUDA_CHECK_KERNEL();
        k_MCER<<<N, T, sh>>>(pin, fin, pout, fout, d_idx, d_rng_gene, n, D, f, lo, hi,
                             strat, F, CR, d_best, Fin, CRin, d_Ftr, d_CRtr, Fout, CRout);
        CUDA_CHECK_KERNEL();
        std::swap(pin, pout);
        std::swap(fin, fout);
        std::swap(Fin, Fout);
        std::swap(CRin, CRout);
        fe += n;
        gen++;
        if (trace) {
            k_argmin<<<1, 512>>>(fin, N, d_best); CUDA_CHECK_KERNEL();
            CUDA_CHECK(cudaMemcpy(&h_best, d_best, sizeof(Best), cudaMemcpyDeviceToHost));
            printf("TRACE gen=%lld FE=%lld best_error=%.9e\n", gen, fe, h_best.val);
        }
    }
    k_argmin<<<1, 512>>>(fin, N, d_best);                                          CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpy(&h_best, d_best, sizeof(Best), cudaMemcpyDeviceToHost));
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
    double cpu_min = h_fit[0];
    for (double v : h_fit) if (v < cpu_min) cpu_min = v;
    if (cpu_min != h_best.val) {
        fprintf(stderr, "SANTE: argmin GPU %.17g != min CPU %.17g\n", h_best.val, cpu_min);
        healthy = false;
    }
    if (h_best.idx < 0 || h_best.idx >= N) {
        fprintf(stderr, "SANTE: argmin GPU invalide (indice %d)\n", h_best.idx);
        return 2;
    }
    const double ref = evaluate(f, &h_pop[(size_t)h_best.idx * D], o.data(), D);
    if (std::fabs(ref - h_best.val) > 1e-9 * std::fmax(1.0, std::fabs(ref))) {
        fprintf(stderr, "SANTE: f GPU %.17g != f CPU %.17g\n", h_best.val, ref);
        healthy = false;
    }
    // jDE : moyenne finale de F et CR dans la population (pour voir l'adaptation)
    char extra[96] = "";
    if (strat == S_JDE) {
        std::vector<double> hF(N), hCR(N);
        CUDA_CHECK(cudaMemcpy(hF.data(), Fin, N * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hCR.data(), CRin, N * sizeof(double), cudaMemcpyDeviceToHost));
        double mF = 0, mCR = 0;
        for (int i = 0; i < N; i++) {
            if (!(hF[i] >= JDE_FL && hF[i] <= JDE_FL + JDE_FU) || !(hCR[i] >= 0.0 && hCR[i] <= 1.0)) {
                fprintf(stderr, "SANTE: parametres jDE hors bornes (F=%g, CR=%g)\n", hF[i], hCR[i]);
                healthy = false;
            }
            mF += hF[i] / N; mCR += hCR[i] / N;
        }
        snprintf(extra, sizeof extra, " F_moyen=%.4f CR_moyen=%.4f", mF, mCR);
    }

    printf("algo=%s func=%s D=%d N=%d seed=%u best_error=%.9e FE=%lld maxFE=%lld "
           "time_s=%.6f threads=%d%s sante=%s\n",
           algo, func_name(f), D, N, seed, h_best.val, fe, maxFE, time_s, T, extra,
           healthy ? "OK" : "ANOMALIE");
    if (csv && !append_result(csv, algo, func_name(f), D, N, seed, h_best.val, time_s, fe)) {
        fprintf(stderr, "Impossible d'ecrire dans %s\n", csv);
        return 1;
    }

    CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(d_popA)); CUDA_CHECK(cudaFree(d_popB));
    CUDA_CHECK(cudaFree(d_fitA)); CUDA_CHECK(cudaFree(d_fitB));
    CUDA_CHECK(cudaFree(d_idx));  CUDA_CHECK(cudaFree(d_rng_ind));
    CUDA_CHECK(cudaFree(d_rng_gene)); CUDA_CHECK(cudaFree(d_best));
    if (strat == S_JDE) {
        CUDA_CHECK(cudaFree(d_FA)); CUDA_CHECK(cudaFree(d_FB));
        CUDA_CHECK(cudaFree(d_CRA)); CUDA_CHECK(cudaFree(d_CRB));
        CUDA_CHECK(cudaFree(d_Ftr)); CUDA_CHECK(cudaFree(d_CRtr));
    }
    return healthy ? 0 : 2;
}
