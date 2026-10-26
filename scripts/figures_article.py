"""Figures en anglais de l'article (Fig. 4 et Fig. 5).

Entrees : results/temps_v1_v2.csv (etape 4, Rastrigin) et results/campagne.csv
Sorties : figures/time_v1_v2_en.pdf et figures/speedup_campaign_en.pdf

Fig. 4 : temps moyen de v1 et v2 en fonction de N (Rastrigin, echelles log).
Fig. 5 : speedup par rapport au DE sequentiel, ratio des temps moyens regroupes sur les
         quatre fonctions (meme definition que la Table V de l'article).

Usage : python scripts/figures_article.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
FIG = RACINE / "figures"
DIMS = [10, 50, 100]
POPS = [50, 100, 500]
STYLE = {"cuda_v1": ("Version 1", "#1f77b4", "o"),
         "cuda_v2": ("Version 2", "#ff7f0e", "s"),
         "gpu_jde": ("jDE", "#2ca02c", "^")}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "legend.fontsize": 8})


def axes_communs(ax, d):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(POPS); ax.set_xticklabels([str(p) for p in POPS])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(f"$D={d}$"); ax.set_xlabel("population size $N$")
    ax.grid(True, which="major", alpha=0.3)


def fig_temps_v1_v2():
    df = pd.read_csv(RACINE / "results" / "temps_v1_v2.csv")
    m = df.groupby(["algo", "D", "N"])["time_s"].mean()
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for ax, d in zip(axs, DIMS):
        for algo in ["cuda_v1", "cuda_v2"]:
            nom, c, mk = STYLE[algo]
            ax.plot(POPS, [m[(algo, d, n)] for n in POPS], marker=mk, color=c, label=nom)
        axes_communs(ax, d)
    axs[0].set_ylabel("mean time (s)")
    axs[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "time_v1_v2_en.pdf")
    plt.close(fig)


def fig_speedup():
    df = pd.read_csv(RACINE / "results" / "campagne.csv")
    m = df.groupby(["D", "N", "algo"])["time_s"].mean().unstack()
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), sharey=True)
    for ax, d in zip(axs, DIMS):
        for algo in ["cuda_v1", "cuda_v2", "gpu_jde"]:
            nom, c, mk = STYLE[algo]
            ax.plot(POPS, [m.loc[(d, n), "sde"] / m.loc[(d, n), algo] for n in POPS],
                    marker=mk, color=c, label=nom)
        ax.axhline(1.0, color="grey", lw=0.8, ls="--")
        axes_communs(ax, d)
    axs[0].set_ylabel("speedup over sequential DE")
    axs[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "speedup_campaign_en.pdf")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_temps_v1_v2()
    fig_speedup()
