// results_csv.h -- FORMAT COMMUN des resultats (issue #5)
// Une ligne par run :  algo,func,D,N,seed,best_error,time_s,FE
#pragma once
#include <cstdio>

static const char* CSV_HEADER = "algo,func,D,N,seed,best_error,time_s,FE";

// Ajoute une ligne au fichier `path` (cree l'en-tete si le fichier est neuf).
inline bool append_result(const char* path, const char* algo, const char* func,
                          int D, int N, unsigned seed, double best_error,
                          double time_s, long long fe) {
    FILE* t = std::fopen(path, "r");
    bool fresh = (t == nullptr);
    if (t) std::fclose(t);
    FILE* fp = std::fopen(path, "a");
    if (!fp) return false;
    if (fresh) std::fprintf(fp, "%s\n", CSV_HEADER);
    std::fprintf(fp, "%s,%s,%d,%d,%u,%.6e,%.6f,%lld\n",
                 algo, func, D, N, seed, best_error, time_s, fe);
    std::fclose(fp);
    return true;
}
