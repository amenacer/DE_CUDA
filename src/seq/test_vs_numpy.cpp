// test_vs_numpy.cpp -- genere des points de reference pour l'issue #6
// (2.3 : tester les 4 fonctions contre NumPy).
//
// Pour chaque fonction, tire 5 points aleatoires dans le domaine et calcule
// evaluate(f, x, o, D) avec l'interface commune (src/common/benchmarks.h).
// Imprime une ligne par point ; scripts/verify_benchmarks_numpy.py recalcule
// independamment avec NumPy et compare (ecart relatif attendu < 1e-12).
//
// Compilation : g++ -std=c++14 -O2 -I../common -o test_vs_numpy test_vs_numpy.cpp
#include <cstdio>
#include <random>
#include "../common/benchmarks.h"

static const int D = 10;
static const int NB_POINTS = 5;
static const unsigned SEED = 42;

int main() {
    std::mt19937 rng(SEED);

    for (int f = 0; f < NUM_FUNCS; f++) {
        std::vector<double> o = make_shift(f, D);
        std::uniform_real_distribution<double> dist(lower_bound(f), upper_bound(f));

        for (int p = 0; p < NB_POINTS; p++) {
            std::vector<double> x(D);
            for (int i = 0; i < D; i++) x[i] = dist(rng);

            double v = evaluate(f, x.data(), o.data(), D);

            printf("PT %s %d %d", func_name(f), D, p);
            for (int i = 0; i < D; i++) printf(" %.17g", o[i]);
            for (int i = 0; i < D; i++) printf(" %.17g", x[i]);
            printf(" %.17g\n", v);
        }
    }
    return 0;
}
