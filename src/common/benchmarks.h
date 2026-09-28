// benchmarks.h -- INTERFACE COMMUNE du projet DE_CUDA (issue #5)
// Utilise par la version sequentielle (src/seq) ET par la version GPU (src/cuda).
// Ne pas modifier sans prevenir les deux autres membres du groupe.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// Une meme fonction compilee pour le CPU et pour le GPU :
// sous nvcc (__CUDACC__ defini) -> __host__ __device__ ; sous g++ -> rien.
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// ---------------------------------------------------------------- identifiants
enum FuncId { F_SPHERE = 0, F_ROSENBROCK = 1, F_GRIEWANK = 2, F_RASTRIGIN = 3 };
static const int NUM_FUNCS = 4;
#define DE_PI 3.14159265358979323846   // macro : utilisable sur CPU et GPU

inline const char* func_name(int f) {
    static const char* names[] = {"sphere", "rosenbrock", "griewank", "rastrigin"};
    return (f >= 0 && f < NUM_FUNCS) ? names[f] : "inconnue";
}
// "sphere" -> 0, ... ; -1 si inconnu
inline int func_id(const char* name) {
    for (int f = 0; f < NUM_FUNCS; f++) if (std::strcmp(name, func_name(f)) == 0) return f;
    return -1;
}

// ---------------------------------------------------------------- domaines (CEC 2005)
HD inline double lower_bound(int f) {
    switch (f) {
        case F_GRIEWANK:  return -600.0;
        case F_RASTRIGIN: return -5.0;
        default:          return -100.0;   // sphere, rosenbrock
    }
}
HD inline double upper_bound(int f) { return -lower_bound(f); }

// ---------------------------------------------------------------- les 4 fonctions
// Toutes renvoient l'ERREUR f(x) - f(x*) : 0 exactement a l'optimum x = o.
// Le biais CEC (-450, 390, -180, -330) n'est PAS ajoute (voir bug 9 :
// en float, pres de 450, on ne peut pas mesurer une erreur < 3e-5).

// Sphere : somme z_i^2, z = x - o
HD inline double sphere(const double* x, const double* o, int D) {
    double s = 0.0;
    for (int i = 0; i < D; i++) { double z = x[i] - o[i]; s += z * z; }
    return s;
}
// Rosenbrock : somme_{i<D-1} 100 (z_i^2 - z_{i+1})^2 + (z_i - 1)^2, z = x - o + 1
HD inline double rosenbrock(const double* x, const double* o, int D) {
    double s = 0.0;
    for (int i = 0; i < D - 1; i++) {
        double zi  = x[i]     - o[i]     + 1.0;
        double zi1 = x[i + 1] - o[i + 1] + 1.0;
        double a = zi * zi - zi1, b = zi - 1.0;
        s += 100.0 * a * a + b * b;
    }
    return s;
}
// Griewank : somme z_i^2 / 4000 - produit cos(z_i / sqrt(i+1)) + 1, z = x - o
HD inline double griewank(const double* x, const double* o, int D) {
    double s = 0.0, p = 1.0;                       // p = 1 (bug 2 corrige)
    for (int i = 0; i < D; i++) {
        double z = x[i] - o[i];
        s += z * z / 4000.0;
        p *= cos(z / sqrt((double)(i + 1)));
    }
    return s - p + 1.0;
}
// Rastrigin : somme z_i^2 - 10 cos(2 pi z_i) + 10, z = x - o
HD inline double rastrigin(const double* x, const double* o, int D) {
    double s = 0.0;
    for (int i = 0; i < D; i++) {
        double z = x[i] - o[i];
        s += z * z - 10.0 * cos(2.0 * DE_PI * z) + 10.0;
    }
    return s;
}
// Point d'entree unique : c'est CETTE fonction que le DE appelle.
HD inline double evaluate(int f, const double* x, const double* o, int D) {
    switch (f) {
        case F_SPHERE:     return sphere(x, o, D);
        case F_ROSENBROCK: return rosenbrock(x, o, D);
        case F_GRIEWANK:   return griewank(x, o, D);
        default:           return rastrigin(x, o, D);
    }
}

// ---------------------------------------------------------------- decalage o
// Generateur deterministe (splitmix64) : meme graine -> memes nombres,
// sur n'importe quelle machine. CPU et GPU utilisent donc EXACTEMENT le meme o.
inline uint64_t splitmix64(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
// o_i tire uniformement dans 80 % du domaine, graine = 1000*f + D (fixee).
inline std::vector<double> make_shift(int f, int D) {
    uint64_t s = 1000ULL * (uint64_t)f + (uint64_t)D;
    double lo = 0.8 * lower_bound(f), hi = 0.8 * upper_bound(f);
    std::vector<double> o(D);
    for (int i = 0; i < D; i++) {
        double u = (double)(splitmix64(s) >> 11) * (1.0 / 9007199254740992.0); // [0,1)
        o[i] = lo + (hi - lo) * u;
    }
    return o;
}

// ---------------------------------------------------------------- budget
inline long long max_fe(int D) { return 10000LL * D; }   // 10^4 x D (sujet)
