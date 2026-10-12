#!/usr/bin/env python3
"""Etape 5 (issue #14) : campagne experimentale complete.

4 fonctions x 3 dimensions x 3 populations x 10 graines, pour
    sde      (DE sequentiel, src/seq/sde.cpp)
    cuda_v1  (src/cuda/de_cuda_v1.cu)
    cuda_v2  (src/cuda/de_cuda_v2.cu)
    gpu_jde  (src/cuda/de_cuda_variantes.cu --strategie jde), optionnel avec --jde
Tous les programmes ont la meme ligne de commande :  prog <fonction> <D> <N> <seed> <csv>
et ecrivent dans le meme CSV :  algo,func,D,N,seed,best_error,time_s,FE

Usage (depuis la racine du depot, sur Colab avec un GPU T4) :
    python3 scripts/run_all.py                    # tout, sde + v1 + v2
    python3 scripts/run_all.py --jde              # + jDE
    python3 scripts/run_all.py --quick            # 1 graine, D=10, N=50 : verifie que tout tourne
    python3 scripts/run_all.py --algos sde        # seulement le sequentiel (CPU, sans nvcc)

Reprise : si le CSV existe deja, les runs deja faits sont sautes (on peut relancer
apres une coupure Colab). Les runs sont entrelaces (meme configuration, algos
l'un apres l'autre) pour que la derive de la machine touche tous les algos pareil.
"""
import argparse
import csv
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEQ_DIR = ROOT / "src" / "seq"
CUDA_DIR = ROOT / "src" / "cuda"

FUNCS = ["sphere", "rosenbrock", "griewank", "rastrigin"]
DIMS = [10, 50, 100]
POPS = [50, 100, 500]
SEEDS = list(range(1, 11))

EXE = ".exe" if sys.platform.startswith("win") else ""

# nom ecrit dans le CSV -> (dossier, source, binaire, arguments supplementaires, besoin de nvcc)
PROGRAMMES = {
    "sde":      (SEQ_DIR,  "sde.cpp",              "sde",               [], False),
    "cuda_v1":  (CUDA_DIR, "de_cuda_v1.cu",        "de_cuda_v1",        [], True),
    "cuda_v2":  (CUDA_DIR, "de_cuda_v2.cu",        "de_cuda_v2",        [], True),
    "gpu_jde":  (CUDA_DIR, "de_cuda_variantes.cu", "de_cuda_variantes", ["--strategie", "jde"], True),
}

CSV_HEADER = ["algo", "func", "D", "N", "seed", "best_error", "time_s", "FE"]
SANTE_RE = re.compile(r"sante=(\w+)")


def compiler(algo):
    dossier, src, binaire, _, gpu = PROGRAMMES[algo]
    out = dossier / (binaire + EXE)
    if gpu:
        cmd = ["nvcc", "-O3", "-arch=sm_75", "-o", str(out), str(dossier / src)]
    else:
        cmd = ["g++", "-std=c++14", "-O2", "-Wall", "-o", str(out), str(dossier / src)]
    print(f"[compilation] {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, cwd=dossier, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout + p.stderr)
        sys.exit(f"Echec de compilation de {src}")
    if (p.stdout + p.stderr).strip():
        print("  avertissements :", (p.stdout + p.stderr).strip())
    return out


def deja_faits(csv_path):
    faits = set()
    if csv_path.exists():
        with open(csv_path, newline="") as fp:
            for r in csv.DictReader(fp):
                faits.add((r["algo"], r["func"], int(r["D"]), int(r["N"]), int(r["seed"])))
    return faits


def ecrire_setup(path, algos):
    """Materiel et versions : a joindre a l'article (README : results/setup.txt)."""
    def cmd(c):
        try:
            return subprocess.run(c, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return "indisponible"
    lignes = [
        f"date : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"systeme : {platform.platform()}",
        f"cpu : {platform.processor() or cmd(['lscpu'])}",
        f"python : {platform.python_version()}",
    ]
    if shutil.which("g++"):
        lignes.append("g++ : " + cmd(["g++", "--version"]).splitlines()[0])
    if shutil.which("nvcc"):
        lignes.append("nvcc : " + cmd(["nvcc", "--version"]).splitlines()[-1])
    if shutil.which("nvidia-smi"):
        lignes.append("gpu : " + cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                      "--format=csv,noheader"]))
    lignes.append("algos : " + ", ".join(algos))
    path.write_text("\n".join(lignes) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Campagne complete DE_CUDA (etape 5)")
    ap.add_argument("--out", default=str(ROOT / "results" / "campagne.csv"))
    ap.add_argument("--algos", nargs="+", choices=list(PROGRAMMES),
                    help="algos a lancer (defaut : sde cuda_v1 cuda_v2)")
    ap.add_argument("--jde", action="store_true", help="ajoute jDE aux algos par defaut")
    ap.add_argument("--skip-compile", action="store_true", help="les binaires existent deja")
    ap.add_argument("--quick", action="store_true", help="test rapide : 1 graine, D=10, N=50")
    ap.add_argument("--seeds", type=int, default=len(SEEDS), help="nombre de graines (defaut 10)")
    a = ap.parse_args()

    algos = a.algos or (["sde", "cuda_v1", "cuda_v2"] + (["gpu_jde"] if a.jde else []))
    if a.quick:
        dims, pops, seeds = [10], [50], [1]
    else:
        dims, pops, seeds = DIMS, POPS, SEEDS[:a.seeds]

    if any(PROGRAMMES[x][4] for x in algos) and not shutil.which("nvcc") and not a.skip_compile:
        sys.exit("nvcc introuvable : lancez la campagne GPU sur Colab (runtime GPU T4), "
                 "ou utilisez --algos sde pour le sequentiel seul.")
    if "sde" in algos and not shutil.which("g++") and not a.skip_compile:
        sys.exit("g++ introuvable : installez-le, ou lancez la campagne sur Colab.")

    binaires = {}
    for x in algos:
        d, _, b, _, _ = PROGRAMMES[x]
        binaires[x] = compiler(x) if not a.skip_compile else d / (b + EXE)
        if not binaires[x].exists():
            sys.exit(f"Binaire absent : {binaires[x]}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ecrire_setup(out.with_name("setup.txt"), algos)
    faits = deja_faits(out)

    # echauffement GPU : un premier run jete (horloges, cache du driver), hors CSV
    for x in algos:
        if PROGRAMMES[x][4]:
            subprocess.run([str(binaires[x]), "sphere", "10", "50", "1"] + PROGRAMMES[x][3],
                           capture_output=True, text=True)

    total = len(FUNCS) * len(dims) * len(pops) * len(seeds) * len(algos)
    restants = total - sum(1 for k in faits
                           if k[0] in algos and k[1] in FUNCS and k[2] in dims and k[3] in pops
                           and k[4] in seeds)
    print(f"\n{total} runs prevus, {total - restants} deja faits, {restants} a lancer -> {out}\n",
          flush=True)

    anomalies, fait, t0 = [], 0, time.time()
    for f in FUNCS:
        for D in dims:
            for N in pops:
                for seed in seeds:
                    for x in algos:
                        if (x, f, D, N, seed) in faits:
                            continue
                        _, _, _, extra, _ = PROGRAMMES[x]
                        cmd = [str(binaires[x]), f, str(D), str(N), str(seed), str(out)] + extra
                        p = subprocess.run(cmd, capture_output=True, text=True)
                        m = SANTE_RE.search(p.stdout)
                        ok = p.returncode == 0 and m is not None and m.group(1) == "OK"
                        if not ok:
                            anomalies.append((x, f, D, N, seed, p.returncode,
                                              (p.stdout + p.stderr).strip()[-300:]))
                            print(f"  ANOMALIE {x} {f} D={D} N={N} seed={seed} "
                                  f"(code {p.returncode})", flush=True)
                        fait += 1
                    # progression a chaque graine complete
                if restants:
                    ecoule = time.time() - t0
                    reste = ecoule / max(fait, 1) * (restants - fait)
                    print(f"  {f:10s} D={D:3d} N={N:3d}  {fait}/{restants} runs, "
                          f"~{reste / 60:.1f} min restantes", flush=True)

    print(f"\nTermine : {fait} runs en {(time.time() - t0) / 60:.1f} min, "
          f"{len(anomalies)} anomalie(s).")
    if anomalies:
        print("Les lignes en anomalie sont quand meme dans le CSV : a verifier avant analyse.")
        for an in anomalies[:20]:
            print("  ", an)
        return 2

    # controle final : chaque configuration a bien toutes ses graines et FE <= 10^4 x D
    faits = deja_faits(out)
    manque = [(x, f, D, N, s) for x in algos for f in FUNCS for D in dims for N in pops
              for s in seeds if (x, f, D, N, s) not in faits]
    if manque:
        print(f"ATTENTION : {len(manque)} run(s) manquant(s) dans le CSV, relancez le script.")
        return 2
    with open(out, newline="") as fp:
        for r in csv.DictReader(fp):
            if int(r["FE"]) > 10000 * int(r["D"]):
                print("ATTENTION : budget depasse :", r)
                return 2
    print("Controle final OK : toutes les configurations sont completes, budget respecte.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
