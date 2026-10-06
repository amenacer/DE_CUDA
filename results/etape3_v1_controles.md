# Étape 3 — contrôles du DE CUDA v1 (issues #9 et #10)

GPU : Tesla T4 (Google Colab), CPU : Intel(R) Xeon(R) CPU @ 2.00GHz, CUDA 12 (nvcc -O3 -arch=sm_75)

Sortie du notebook `notebooks/Etape3_DE_CUDA_v1.ipynb` :

```
de_cuda_v1.cu : compilation SANS aucun warning
Lancement de k_eval avec 2048 threads par bloc (maximum autorise : 1024)...
Erreur CUDA de_cuda_v1.cu:134 : invalid argument
  dans : cudaGetLastError()
code de retour : 1  =>  erreur detectee : OK
Determinisme (D = 10, N = 50)
  sphere     seed 7 : 0.000000000000000e+00 | seed 7 : 0.000000000000000e+00 -> IDENTIQUES | seed 8 : 0.000000e+00
  rastrigin  seed 7 : 0.000000000000000e+00 | seed 7 : 0.000000000000000e+00 -> IDENTIQUES | seed 8 : 0.000000e+00
  rosenbrock seed 7 : 3.506597420000000e-02 | seed 7 : 3.506597420000000e-02 -> IDENTIQUES | seed 8 : 1.157099e+00
  griewank   seed 7 : 0.000000000000000e+00 | seed 7 : 0.000000000000000e+00 -> IDENTIQUES | seed 8 : 0.000000e+00
Temps (s) de 5 runs identiques : 1.8387, 1.8398, 1.8405, 1.8390, 1.8378
ecart run1/run2 = 0.06 %   ecart max/min sur 5 runs = 0.14 %  =>  OK (< 10 %)
| fonction | D | N | best_error | FE | temps (s) | sante |
|---|---|---|---|---|---|---|
| sphere | 10 | 100 | 0.000e+00 | 100000 | 0.0158 | OK |
| sphere | 50 | 100 | 0.000e+00 | 500000 | 0.2527 | OK |
| sphere | 100 | 100 | 0.000e+00 | 1000000 | 0.8658 | OK |
| rosenbrock | 10 | 100 | 5.849e+00 | 100000 | 0.0181 | OK |
| rosenbrock | 50 | 100 | 4.118e+01 | 500000 | 0.3174 | OK |
| rosenbrock | 100 | 100 | 8.961e+01 | 1000000 | 1.1602 | OK |
| griewank | 10 | 100 | 0.000e+00 | 100000 | 0.0361 | OK |
| griewank | 50 | 100 | 0.000e+00 | 500000 | 0.7540 | OK |
| griewank | 100 | 100 | 0.000e+00 | 1000000 | 2.8960 | OK |
| rastrigin | 10 | 100 | 0.000e+00 | 100000 | 0.0261 | OK |
| rastrigin | 50 | 100 | 1.729e+02 | 500000 | 0.4821 | OK |
| rastrigin | 100 | 100 | 5.336e+02 | 1000000 | 1.8375 | OK |
```
