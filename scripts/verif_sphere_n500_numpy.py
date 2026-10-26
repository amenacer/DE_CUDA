#!/usr/bin/env python3
"""verif_sphere_n500_numpy.py -- controle independant sur Sphere, N = 500 (relecture, point 2a).

Qin et al. (GECCO 2012) publient un taux de succes de 1 (erreur < 1e-8) sur Sphere a N = 500
pour D = 10, 50, 100. Ce script reecrit DE/rand/1/bin generationnel (F = 0,5, CR = 0,3,
budget 10^4 D, bornes ramenees au bord) en NumPy, independamment du code C++/CUDA, et le lance
sur le domaine CEC [-100, 100] puis sur [-5.12, 5.12] (domaine du PSO du cours).

  python3 scripts/verif_sphere_n500_numpy.py     (quelques minutes)

Resultats obtenus (5 graines, moyenne / max) :
  D = 10  : [-100,100] 8.1e-6 / 1.1e-5   [-5.12,5.12] 2.1e-8 / 3.0e-8
  D = 50  : [-100,100] 2.6e-4 / 2.9e-4   [-5.12,5.12] 7.1e-7 / 8.2e-7
  D = 100 : [-100,100] 0.097  / 0.113    [-5.12,5.12] 2.6e-4 / 3.0e-4
Campagne (cuda_v2, 10 runs) : 9.6e-6, 2.4e-4, 0.098. Aucun cas n'atteint 1e-8.
"""
import numpy as np, sys
def draw(rng,N):
    i=np.arange(N); r=np.empty((N,3),int)
    for k in range(3):
        c=rng.integers(0,N,N)
        while True:
            bad=(c==i)|np.any(r[:,:k]==c[:,None],axis=1)
            if not bad.any(): break
            c[bad]=rng.integers(0,N,bad.sum())
        r[:,k]=c
    return r
def de(D,N,lo,hi,seed,F=0.5,CR=0.3,mode='gen'):
    rng=np.random.default_rng(seed); o=rng.uniform(.8*lo,.8*hi,D)
    X=rng.uniform(lo,hi,(N,D)); f=((X-o)**2).sum(1); fe=N
    while fe<10000*D:
        r=draw(rng,N); V=X[r[:,0]]+F*(X[r[:,1]]-X[r[:,2]])
        m=rng.random((N,D))<CR; m[np.arange(N),rng.integers(0,D,N)]=True
        U=np.clip(np.where(m,V,X),lo,hi); fu=((U-o)**2).sum(1); fe+=N
        b=fu<=f; X[b]=U[b]; f[b]=fu[b]
    return f.min()
if __name__ == "__main__":
  for D in [10,50,100]:
    for lo,hi in [(-100,100),(-5.12,5.12)]:
        v=[de(D,500,lo,hi,s) for s in range(1,6)]
        print(D,hi,np.mean(v),np.max(v),flush=True)
