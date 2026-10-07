# Diagramme de séquence du PSO original (tâche 1.4)

Source : `docs/uml/sequence_pso.mmd` · Image pour l'article : `docs/uml/sequence_pso.png`.
Temps par itération mesurés à la tâche 1.3 (GPU T4, Levy 3D, N = 512, 30 000 itérations).

```mermaid
sequenceDiagram
    autonumber
    participant M as main (CPU)
    participant P as cuda_pso (CPU)
    participant G as GPU
    M->>M: init positions, vitesses, pBests, gBest (rand)
    M->>P: cuda_pso(...)
    P->>G: cudaMalloc x4 (devPos, devVel, devPBest, devGBest)
    P->>G: cudaMemcpy H->D x4
    loop MAX_ITER = 30 000 iterations
        P->>G: kernelUpdateParticle<<<48,32>>>(r1, r2)
        Note right of G: 4.8 us/iter (3.8 %)
        P->>G: kernelUpdatePBest<<<48,32>>>
        Note right of G: 37 us/iter (29.5 %)
        G-->>P: cudaMemcpy pBests D->H (6 Ko)
        Note right of G: 11.3 us/iter (9.0 %)
        P->>P: boucle CPU : meilleur pBest -> gBest (1024 evaluations)
        Note right of P: 32.6 us/iter (26.0 %)
        P->>G: cudaMemcpy gBest H->D (12 octets)
        Note right of G: 8.6 us/iter (6.8 %)
    end
    G-->>P: cudaMemcpy D->H x4
    P->>G: cudaFree x4
    P-->>M: retour
    M->>M: affiche temps et f(gBest)
```

## Lecture

- Les flèches 3, 4, 10 et 11 ne sont exécutées **qu'une fois** : allocation, copie initiale, copie finale, libération.
- Dans la boucle, **3 flèches sur 5 traversent le bus PCIe ou restent sur le CPU** (7, 8, 9) : c'est 42 % du temps mesuré, sans aucun calcul GPU.
- La flèche 8 est **séquentielle** : le CPU évalue 1024 fois la fonction pendant que le GPU attend.
- Ce que le DE GPU doit supprimer : les flèches 7, 8 et 9. Toute la boucle reste sur le GPU, et le meilleur individu est obtenu par une réduction parallèle.
