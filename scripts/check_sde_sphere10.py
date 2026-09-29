#!/usr/bin/env python3
"""Issue #8 (2.7-2.9) : critere de validation -- 10 runs de src/seq/sde sur
Sphere, D=10, N=50, en verifiant best_error < 1e-8 a chaque run.

Usage : python3 scripts/check_sde_sphere10.py
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEQ_DIR = ROOT / "src" / "seq"
SRC = SEQ_DIR / "sde.cpp"
BIN = SEQ_DIR / "sde"
CSV = ROOT / "results" / "sde_sphere10_check.csv"

D, N, NB_RUNS, THRESHOLD = 10, 50, 10, 1e-8

RESULT_RE = re.compile(r"best_error=([\d.eE+-]+).*FE=(\d+).*sante=(\w+)")


def main():
    subprocess.run(
        ["g++", "-std=c++14", "-O2", "-Wall", "-o", str(BIN), str(SRC)],
        cwd=SEQ_DIR, check=True,
    )
    CSV.parent.mkdir(exist_ok=True)
    if CSV.exists():
        CSV.unlink()

    ok = True
    for seed in range(1, NB_RUNS + 1):
        out = subprocess.run(
            [str(BIN), "sphere", str(D), str(N), str(seed), str(CSV)],
            cwd=SEQ_DIR, check=True, capture_output=True, text=True,
        ).stdout.strip()
        m = RESULT_RE.search(out)
        if not m:
            print(f"[seed {seed}] sortie inattendue : {out}")
            ok = False
            continue
        err, fe, sante = float(m[1]), int(m[2]), m[3]
        good = err < THRESHOLD and sante == "OK"
        ok &= good
        print(f"[seed {seed:2d}] best_error={err:.3e}  FE={fe}  sante={sante}  "
              f"{'OK' if good else 'ECHEC'}")

    print(f"\n{'=== TOUS LES RUNS PASSENT (erreur < 1e-8) ===' if ok else '=== ECHEC ==='}")
    print(f"Resultats bruts : {CSV.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
