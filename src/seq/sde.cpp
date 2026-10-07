// sde.cpp -- DE/rand/1/bin sequentiel (issues #7 et #8, 2.4-2.9)
//
// Reference de correction du projet : si le DE GPU (src/cuda) donne un
// resultat different de celui-ci sur les memes (fonction, D, N, seed), c'est
// que le GPU a un bug. Utilise l'interface commune (src/common/benchmarks.h)
// pour que les deux versions optimisent exactement les memes fonctions.
//
// Usage : ./sde <fonction> <D> <N> <seed> [fichier.csv] [--trace]
// Compilation : g++ -std=c++14 -O2 -o sde sde.cpp
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "../common/benchmarks.h"
#include "../common/results_csv.h"

static const double DE_F = 0.5;
static const double DE_CR = 0.3;

static double clampd(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Un indice dans [0, N), different de i et des indices deja tires.
static int pick_distinct(std::mt19937& rng, std::uniform_int_distribution<int>& dist,
                          int i, int a = -1, int b = -1) {
    int r;
    do { r = dist(rng); } while (r == i || r == a || r == b);
    return r;
}

// Controle de sante (2.9) : verifie que rien d'anormal ne s'est glisse dans
// le resultat final -- NaN/Inf (souvent un signe de mutation qui diverge) ou
// une coordonnee hors des bornes du domaine (la clamp aurait du l'empecher).
static bool health_check(const std::vector<double>& pop, const std::vector<double>& fit,
                          int N, int D, double lo, double hi, long long fe, long long maxFE) {
    bool healthy = true;
    for (int i = 0; i < N; i++) {
        if (!std::isfinite(fit[i])) {
            fprintf(stderr, "SANTE: fitness non finie pour l'individu %d (%.6g)\n", i, fit[i]);
            healthy = false;
        }
        for (int j = 0; j < D; j++) {
            double v = pop[i * D + j];
            if (!std::isfinite(v) || v < lo || v > hi) {
                fprintf(stderr, "SANTE: pop[%d][%d]=%.6g hors de [%.6g, %.6g]\n", i, j, v, lo, hi);
                healthy = false;
            }
        }
    }
    if (fe > maxFE) {
        fprintf(stderr, "SANTE: budget depasse, FE=%lld > maxFE=%lld\n", fe, maxFE);
        healthy = false;
    }
    return healthy;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <fonction> <D> <N> <seed> [fichier.csv] [--trace]\n", argv[0]);
        fprintf(stderr, "  fonction in {sphere, rosenbrock, griewank, rastrigin}\n");
        return 1;
    }

    int fid = func_id(argv[1]);
    if (fid < 0) {
        fprintf(stderr, "Fonction inconnue : %s\n", argv[1]);
        return 1;
    }
    int D = atoi(argv[2]);
    int N = atoi(argv[3]);
    unsigned seed = (unsigned)strtoul(argv[4], nullptr, 10);

    if (D < 1 || N < 4) {
        // DE/rand/1/bin a besoin de 3 individus distincts, differents de i.
        fprintf(stderr, "D doit etre >= 1 et N >= 4 (recu D=%d, N=%d)\n", D, N);
        return 1;
    }

    const char* csv_path = nullptr;
    bool trace = false;
    for (int a = 5; a < argc; a++) {
        if (std::strcmp(argv[a], "--trace") == 0) trace = true;
        else csv_path = argv[a];
    }

    const long long maxFE = max_fe(D);
    const double lo = lower_bound(fid), hi = upper_bound(fid);
    const std::vector<double> o = make_shift(fid, D);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> init_dist(lo, hi);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_int_distribution<int> idx_dist(0, N - 1);
    std::uniform_int_distribution<int> dim_dist(0, D - 1);

    auto t0 = std::chrono::steady_clock::now();

    std::vector<double> pop(N * D);
    std::vector<double> fit(N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) pop[i * D + j] = init_dist(rng);
        fit[i] = evaluate(fid, &pop[i * D], o.data(), D);
    }
    long long fe = N;

    auto best_of = [&]() {
        double b = fit[0];
        for (int i = 1; i < N; i++) if (fit[i] < b) b = fit[i];
        return b;
    };
    if (trace) printf("TRACE gen=0 FE=%lld best_error=%.9e\n", fe, best_of());

    std::vector<double> trial(D);
    long long gen = 0;
    while (fe < maxFE) {
        for (int i = 0; i < N && fe < maxFE; i++) {
            int r1 = pick_distinct(rng, idx_dist, i);
            int r2 = pick_distinct(rng, idx_dist, i, r1);
            int r3 = pick_distinct(rng, idx_dist, i, r1, r2);

            int jrand = dim_dist(rng);
            for (int j = 0; j < D; j++) {
                if (unit(rng) < DE_CR || j == jrand) {
                    double v = pop[r1 * D + j] + DE_F * (pop[r2 * D + j] - pop[r3 * D + j]);
                    trial[j] = clampd(v, lo, hi);
                } else {
                    trial[j] = pop[i * D + j];
                }
            }

            double trial_fit = evaluate(fid, trial.data(), o.data(), D);
            fe++;

            if (trial_fit <= fit[i]) {
                for (int j = 0; j < D; j++) pop[i * D + j] = trial[j];
                fit[i] = trial_fit;
            }
        }
        gen++;
        if (trace) printf("TRACE gen=%lld FE=%lld best_error=%.9e\n", gen, fe, best_of());
    }

    auto t1 = std::chrono::steady_clock::now();
    double time_s = std::chrono::duration<double>(t1 - t0).count();
    double best = best_of();

    bool healthy = health_check(pop, fit, N, D, lo, hi, fe, maxFE);

    printf("func=%s D=%d N=%d seed=%u best_error=%.9e FE=%lld maxFE=%lld time_s=%.6f sante=%s\n",
           func_name(fid), D, N, seed, best, fe, maxFE, time_s, healthy ? "OK" : "ANOMALIE");

    if (csv_path) {
        if (!append_result(csv_path, "sde", func_name(fid), D, N, seed, best, time_s, fe)) {
            fprintf(stderr, "Impossible d'ecrire dans %s\n", csv_path);
            return 1;
        }
    }

    return healthy ? 0 : 2;
}
