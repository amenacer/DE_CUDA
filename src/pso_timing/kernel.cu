#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>

#include "kernel.h"
#include <chrono>


__device__ float tempParticle1[NUM_OF_DIMENSIONS];
__device__ float tempParticle2[NUM_OF_DIMENSIONS];

/* Objective function
0: Levy 3-dimensional
1: Shifted Rastigrin's Function
2: Shifted Rosenbrock's Function
3: Shifted Griewank's Function
4: Shifted Sphere's Function
*/
/**
 * Runs on the GPU, called from the GPU.
*/
__device__ float fitness_function(float x[]) {
    float res = 0;
    float somme = 0;
    float produit = 0;

    switch (SELECTED_OBJ_FUNC)  {
        case 0: 
            float y1 = 1 + (x[0] - 1)/4;
            float yn = 1 + (x[NUM_OF_DIMENSIONS-1] - 1)/4;

            res += pow(sin(phi*y1), 2);

            for (int i = 0; i < NUM_OF_DIMENSIONS-1; i++) {
                float y = 1 + (x[i] - 1)/4;
                float yp = 1 + (x[i+1] - 1)/4;
                res += pow(y - 1, 2)*(1 + 10*pow(sin(phi*yp), 2)) + pow(yn - 1, 2);
            }
            break;
        case 1: 
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2) - 10*cos(2*phi*zi) + 10;
            }
            res -= 330;
            break;
        
        case 2:
            for (int i = 0; i < NUM_OF_DIMENSIONS-1; i++) {
                float zi = x[i] - 0 + 1;
                float zip1 = x[i+1] - 0 + 1;
                res += 100 * ( pow(pow(zi, 2) - zip1, 2)) + pow(zi - 1, 2);
            }
            res += 390;
            break;
        case 3:
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                somme += pow(zi, 2)/4000;
                produit *= cos(zi/pow(i+1, 0.5));
            }
            res = somme - produit + 1 - 180; 
            break;
        case 4:
            for(int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2);
            }
            res -= 450;
            break;
    }

    return res;
}

/**
 * 
 * Runs on the GPU, called from the CPU or the GPU
*/
__global__ void kernelUpdateParticle(float *positions, float *velocities, 
                                     float *pBests, float *gBest, float r1, 
                                     float r2)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // avoid an out of bound for the array 
    if(i >= NUM_OF_PARTICLES * NUM_OF_DIMENSIONS)
        return;

    //float rp = getRandomClamped();
    //float rg = getRandomClamped();
    
    float rp = r1; // random weight for personnal =>  computed from @getRandomClamped
    float rg = r2; // random weight for global =>  computed from @getRandomClamped


    // Mise à jour de velocities et positions
    velocities[i] = OMEGA * velocities[i] + 
                    c1 * rp * (pBests[i] - positions[i]) + 
                    c2 * rg * (gBest[i % NUM_OF_DIMENSIONS] - positions[i]);

    // Update posisi particle
    //Mise à jour de la position de la particule courante
    //incrémentant la position de la particule courante avec la vitesse de la particule courante
    positions[i] += velocities[i];
}

/**
 * Runs on the GPU, called from the CPU or the GPU
*/
__global__ void kernelUpdatePBest(float *positions, float *pBests, float* gBest)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i >= NUM_OF_PARTICLES * NUM_OF_DIMENSIONS || i % NUM_OF_DIMENSIONS != 0)
        return;

    for (int j = 0; j < NUM_OF_DIMENSIONS; j++)
    {
        tempParticle1[j] = positions[i + j];
        tempParticle2[j] = pBests[i + j];
    }

    if (fitness_function(tempParticle1) < fitness_function(tempParticle2))
    {
        for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
            pBests[i + k] = positions[i + k];
    }
}


extern "C" void cuda_pso(float *positions, float *velocities, float *pBests, float *gBest)
{
    int size = NUM_OF_PARTICLES * NUM_OF_DIMENSIONS;
    float *devPos, *devVel, *devPBest, *devGBest;
    float temp[NUM_OF_DIMENSIONS];

    cudaMalloc((void**)&devPos, sizeof(float) * size);
    cudaMalloc((void**)&devVel, sizeof(float) * size);
    cudaMalloc((void**)&devPBest, sizeof(float) * size);
    cudaMalloc((void**)&devGBest, sizeof(float) * NUM_OF_DIMENSIONS);

    int threadsNum = 32;
    int blocksNum = ceil(size / threadsNum);

    cudaMemcpy(devPos, positions, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devVel, velocities, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devPBest, pBests, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devGBest, gBest, sizeof(float) * NUM_OF_DIMENSIONS, cudaMemcpyHostToDevice);

    // ===== Chronometrage : preparation =====
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    float ms;
    double t_k1 = 0, t_k2 = 0, t_d2h = 0, t_cpu = 0, t_h2d = 0;
    auto T0 = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < MAX_ITER; iter++)
    {
        // (1) kernelUpdateParticle -> t_k1   
        cudaEventRecord(e0);
        kernelUpdateParticle<<<blocksNum, threadsNum>>>(devPos, devVel, devPBest, devGBest,
                                                        getRandomClamped(), getRandomClamped());
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1); t_k1 += ms;

        // (2) kernelUpdatePBest -> t_k2   (meme principe que (1))
        cudaEventRecord(e0);
        kernelUpdatePBest<<<blocksNum, threadsNum>>>(devPos, devPBest, devGBest);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1); t_k2 += ms;

        // (3) copie pBests GPU -> CPU -> t_d2h   (une copie se mesure comme un kernel)
        cudaEventRecord(e0);
        cudaMemcpy(pBests, devPBest, sizeof(float) * size, cudaMemcpyDeviceToHost);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1); t_d2h += ms;

        // (4) boucle CPU du gBest -> t_cpu
        //     C'est du code CPU : on utilise une horloge CPU (std::chrono), pas cudaEvent.
        auto c0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size; i += NUM_OF_DIMENSIONS)
        {
            for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
                temp[k] = pBests[i + k];
            if (host_fitness_function(temp) < host_fitness_function(gBest))
            {
                for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
                    gBest[k] = temp[k];
            }
        }

        auto c1 = std::chrono::high_resolution_clock::now();
        t_cpu += std::chrono::duration<double, std::milli>(c1 - c0).count();

        // (5) copie gBest CPU -> GPU -> t_h2d
        cudaEventRecord(e0);
        cudaMemcpy(devGBest, gBest, sizeof(float) * NUM_OF_DIMENSIONS, cudaMemcpyHostToDevice);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1); t_h2d += ms;
    }

    // ===== Chronometrage : resultats =====
    auto T1 = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double, std::milli>(T1 - T0).count();
    double somme = t_k1 + t_k2 + t_d2h + t_cpu + t_h2d;
    printf("\n%-22s %10s %8s\n", "Morceau", "ms", "%");
    printf("%-22s %10.1f %7.1f%%\n", "kernelUpdateParticle", t_k1,  100*t_k1/total);
    printf("%-22s %10.1f %7.1f%%\n", "kernelUpdatePBest",    t_k2,  100*t_k2/total);
    printf("%-22s %10.1f %7.1f%%\n", "memcpy pBests D->H",   t_d2h, 100*t_d2h/total);
    printf("%-22s %10.1f %7.1f%%\n", "boucle CPU gBest",     t_cpu, 100*t_cpu/total);
    printf("%-22s %10.1f %7.1f%%\n", "memcpy gBest H->D",    t_h2d, 100*t_h2d/total);
    printf("%-22s %10.1f %7.1f%%\n", "SOMME des parties",    somme, 100*somme/total);
    printf("%-22s %10.1f\n",         "TOTAL boucle",         total);
    cudaEventDestroy(e0); cudaEventDestroy(e1);

    cudaMemcpy(positions, devPos, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(velocities, devVel, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(pBests, devPBest, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gBest, devGBest, sizeof(float) * NUM_OF_DIMENSIONS, cudaMemcpyDeviceToHost);

    cudaFree(devPos); cudaFree(devVel); cudaFree(devPBest); cudaFree(devGBest);
}
