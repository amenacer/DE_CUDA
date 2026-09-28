// test_benchmarks.cpp -- valide l'interface commune (critere de l'issue #5)
#include <cstdio>
#include "benchmarks.h"
#include "results_csv.h"

int main() {
    int dims[3] = {10, 50, 100};
    int ok = 1;
    printf("Test 1 : f(o) = 0 exactement\n");
    for (int f = 0; f < NUM_FUNCS; f++)
        for (int d : dims) {
            std::vector<double> o = make_shift(f, d);
            double v = evaluate(f, o.data(), o.data(), d);
            bool good = (v == 0.0);
            ok &= good;
            printf("  %-10s D=%3d  f(o) = %.3e  %s\n", func_name(f), d, v, good ? "OK" : "ECHEC");
        }
    printf("Test 2 : le decalage reste dans 80 %% du domaine\n");
    for (int f = 0; f < NUM_FUNCS; f++) {
        std::vector<double> o = make_shift(f, 100);
        double mn = o[0], mx = o[0];
        for (double v : o) { if (v < mn) mn = v; if (v > mx) mx = v; }
        bool good = mn >= 0.8 * lower_bound(f) && mx <= 0.8 * upper_bound(f);
        ok &= good;
        printf("  %-10s domaine [%g, %g]  o dans [%.2f, %.2f]  %s\n", func_name(f),
               lower_bound(f), upper_bound(f), mn, mx, good ? "OK" : "ECHEC");
    }
    printf("Test 3 : valeurs de reference (a comparer avec NumPy)\n");
    for (int f = 0; f < NUM_FUNCS; f++) {
        std::vector<double> o = make_shift(f, 10), x(10);
        for (int i = 0; i < 10; i++) x[i] = o[i] + 0.1 * (i + 1);   // point connu
        printf("  REF %s %.17g\n", func_name(f), evaluate(f, x.data(), o.data(), 10));
    }
    printf("Test 4 : ecriture CSV\n");
    std::remove("test_resultats.csv");
    append_result("test_resultats.csv", "test", "sphere", 10, 50, 1, 1.234e-9, 0.5, 100000);
    append_result("test_resultats.csv", "test", "rastrigin", 10, 50, 2, 3.5, 0.6, 100000);
    FILE* fp = fopen("test_resultats.csv", "r"); char line[256];
    while (fgets(line, sizeof line, fp)) printf("  %s", line);
    fclose(fp);
    printf("\n%s\n", ok ? "=== TOUS LES TESTS PASSENT ===" : "=== ECHEC ===");
    return ok ? 0 : 1;
}
