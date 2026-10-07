#!/usr/bin/env python3
"""Issue #6 (2.3) : verifie les 4 fonctions de src/common/benchmarks.h contre
une reimplementation NumPy independante.

Compile et execute src/seq/test_vs_numpy.cpp, qui imprime 5 points aleatoires
par fonction (decalage o, point x, valeur C++). Ce script recalcule chaque
valeur avec NumPy, a partir des formules mathematiques (pas du code C++), et
verifie un ecart relatif < 1e-12.

Usage : python3 scripts/verify_benchmarks_numpy.py
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEQ_DIR = ROOT / "src" / "seq"
SRC = SEQ_DIR / "test_vs_numpy.cpp"
BIN = SEQ_DIR / "test_vs_numpy"
TOLERANCE = 1e-12


def numpy_sphere(x, o):
    z = x - o
    return float(np.sum(z ** 2))


def numpy_rosenbrock(x, o):
    z = x - o + 1.0
    a = z[:-1] ** 2 - z[1:]
    b = z[:-1] - 1.0
    return float(np.sum(100.0 * a ** 2 + b ** 2))


def numpy_griewank(x, o):
    z = x - o
    idx = np.arange(1, len(z) + 1)
    return float(np.sum(z ** 2) / 4000.0 - np.prod(np.cos(z / np.sqrt(idx))) + 1.0)


def numpy_rastrigin(x, o):
    z = x - o
    return float(np.sum(z ** 2 - 10.0 * np.cos(2.0 * np.pi * z) + 10.0))


NUMPY_IMPL = {
    "sphere": numpy_sphere,
    "rosenbrock": numpy_rosenbrock,
    "griewank": numpy_griewank,
    "rastrigin": numpy_rastrigin,
}

LINE_RE = re.compile(r"^PT (\w+) (\d+) (\d+) (.+)$")


def build_and_run():
    subprocess.run(
        ["g++", "-std=c++14", "-O2", "-Wall", "-o", str(BIN), str(SRC)],
        cwd=SEQ_DIR, check=True,
    )
    out = subprocess.run([str(BIN)], cwd=SEQ_DIR, check=True,
                          capture_output=True, text=True).stdout
    return out


def main():
    output = build_and_run()

    ok = True
    n_points = 0
    for line in output.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        func, d_str, idx, rest = m.groups()
        d = int(d_str)
        values = [float(v) for v in rest.split()]
        o = np.array(values[:d])
        x = np.array(values[d:2 * d])
        cpp_value = values[2 * d]

        numpy_value = NUMPY_IMPL[func](x, o)
        rel_err = abs(cpp_value - numpy_value) / max(1.0, abs(numpy_value))
        good = rel_err < TOLERANCE
        ok &= good
        n_points += 1
        status = "OK" if good else "ECHEC"
        print(f"  {func:10s} D={d:3d} pt={idx}  C++={cpp_value:.15g}"
              f"  NumPy={numpy_value:.15g}  ecart_relatif={rel_err:.1e}  {status}")

    print(f"\n{n_points} points verifies (4 fonctions x 5 points).")
    print("=== TOUS LES TESTS PASSENT ===" if ok else "=== ECHEC ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
