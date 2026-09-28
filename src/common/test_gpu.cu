// test_gpu.cu -- verifie que les fonctions donnent le MEME resultat sur GPU et sur CPU
#include <cstdio>
#include <cmath>
#include "benchmarks.h"

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    printf("Erreur CUDA %s ligne %d : %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

// Un seul thread suffit ici : on teste la fonction, pas la performance.
__global__ void k_eval(int f, const double* x, const double* o, int D, double* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *out = evaluate(f, x, o, D);
}

int main() {
    const int D = 100;
    int ok = 1;
    double *dx, *dout, *dshift;
    CUDA_CHECK(cudaMalloc(&dx, D * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dshift, D * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dout, sizeof(double)));
    for (int f = 0; f < NUM_FUNCS; f++) {
        std::vector<double> o = make_shift(f, D), x(D);
        for (int i = 0; i < D; i++) x[i] = o[i] + 0.01 * (i + 1);
        double cpu = evaluate(f, x.data(), o.data(), D);
        CUDA_CHECK(cudaMemcpy(dx, x.data(), D * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dshift, o.data(), D * sizeof(double), cudaMemcpyHostToDevice));
        k_eval<<<1, 1>>>(f, dx, dshift, D, dout);
        CUDA_CHECK(cudaGetLastError());
        double gpu;
        CUDA_CHECK(cudaMemcpy(&gpu, dout, sizeof(double), cudaMemcpyDeviceToHost));
        double rel = fabs(gpu - cpu) / fmax(1.0, fabs(cpu));
        bool good = rel < 1e-12;
        ok &= good;
        printf("%-10s CPU = %.15e  GPU = %.15e  ecart rel = %.1e  %s\n",
               func_name(f), cpu, gpu, rel, good ? "OK" : "ECHEC");
    }
    cudaFree(dx); cudaFree(dshift); cudaFree(dout);
    printf("\n%s\n", ok ? "=== CPU et GPU donnent le meme resultat ===" : "=== ECHEC ===");
    return ok ? 0 : 1;
}
