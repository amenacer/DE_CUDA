#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyse.py -- analyse des résultats de la campagne DE séquentiel / DE CUDA.

Entrées
-------
--results      CSV : une ligne par run
                 colonnes obligatoires : algo,function,dim,pop,run,time_s
                 + soit `error` (= best_fitness - f_bias), soit `best_fitness`
                 colonne optionnelle : seed
--convergence  CSV (optionnel) : algo,function,dim,pop,run,evals,best_fitness

Sorties (dans --outdir)
-----------------------
tables/quality_D<dim>.{md,tex,csv}   erreur moyenne ± écart-type, meilleur en gras, symbole de test
tables/time_speedup.{md,tex,csv}     temps moyens et speedup par rapport à l'algorithme de référence
tables/stats_pairwise.csv            test de chaque algo contre la référence (p brut, p Holm, effet)
tables/stats_kruskal.csv             Kruskal-Wallis entre tous les algos, par configuration
tables/stats_summary.md              bilan gagne / égalité / perd
figures/*.{png,pdf}                  convergence, boîtes à moustaches, temps, speedup

Exemples
--------
  python analyse.py --results fake_data/results_FAKE.csv \
                    --convergence fake_data/convergence_FAKE.csv --outdir out
  python analyse.py --results results.csv --paired          # runs appariés par graine
  python analyse.py --results results.csv --reference DE_seq --alpha 0.05
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # pas d'écran nécessaire
import matplotlib.pyplot as plt
from scipy import stats

# --------------------------------------------------------------------------- constantes
BIAS = {"sphere": -450.0, "rastrigin": -330.0, "rosenbrock": 390.0, "griewank": -180.0}
TITLES = {
    "sphere": "Shifted Sphere",
    "rastrigin": "Shifted Rastrigin",
    "rosenbrock": "Shifted Rosenbrock",
    "griewank": "Shifted Griewank",
}
EPS = 1e-12  # plancher pour l'échelle log (une erreur peut valoir 0 ou être négative d'epsilon en float)
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 8})


# --------------------------------------------------------------------------- utilitaires
def ordered_functions(df):
    pref = ["sphere", "rastrigin", "rosenbrock", "griewank"]
    present = list(dict.fromkeys(df["function"]))
    return [f for f in pref if f in present] + sorted(f for f in present if f not in pref)


def sci(x, d=2):
    return f"{x:.{d}e}"


def sci_tex(x, d=2):
    if x == 0:
        return "0"
    m, e = f"{x:.{d}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def tex_table(header, rows, caption, label):
    cols = "l" * 2 + "c" * (len(header) - 2)
    header = [h.replace("_", "\\_") for h in header]  # DE_seq -> DE\_seq (sinon erreur LaTeX)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\scriptsize",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def save_fig(fig, outdir, name):
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, name + ".png"), dpi=200)
    fig.savefig(os.path.join(figdir, name + ".pdf"))  # vectoriel pour l'article
    plt.close(fig)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# --------------------------------------------------------------------------- chargement
def load_results(path):
    df = pd.read_csv(path)
    need = ["algo", "function", "dim", "pop", "run", "time_s"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"Colonnes manquantes dans {path} : {miss}")
    if "error" not in df.columns:
        if "best_fitness" not in df.columns:
            sys.exit("Il faut une colonne `error` ou `best_fitness`.")
        bias = df["function"].map(BIAS)
        if bias.isna().any():
            warnings.warn("Fonction inconnue dans BIAS : biais supposé nul pour ces lignes.")
        df["error"] = df["best_fitness"] - bias.fillna(0.0)
    df["error_plot"] = df["error"].clip(lower=EPS)
    return df


def load_convergence(path):
    cv = pd.read_csv(path)
    need = ["algo", "function", "dim", "pop", "run", "evals", "best_fitness"]
    miss = [c for c in need if c not in cv.columns]
    if miss:
        sys.exit(f"Colonnes manquantes dans {path} : {miss}")
    cv["error"] = (cv["best_fitness"] - cv["function"].map(BIAS).fillna(0.0)).clip(lower=EPS)
    return cv


# --------------------------------------------------------------------------- statistiques
def holm(p):
    """Correction de Holm-Bonferroni : contrôle le risque d'au moins un faux positif."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, (n - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def run_stats(df, algos, ref, alpha, paired, scope):
    rows, kw_rows = [], []
    for (func, dim, pop), g in df.groupby(["function", "dim", "pop"], sort=True):
        data = {a: g[g["algo"] == a].sort_values("run") for a in algos}
        a_df = data[ref]
        if len(a_df) == 0:
            continue
        for b in algos:
            if b == ref or len(data[b]) == 0:
                continue
            b_df = data[b]
            if paired:
                m = a_df[["run", "error"]].merge(b_df[["run", "error"]], on="run", suffixes=("_a", "_b"))
                a, bb = m["error_a"].to_numpy(), m["error_b"].to_numpy()
                test = "wilcoxon_signed_rank"
                if len(a) == 0 or np.all(a == bb):
                    stat, p = np.nan, 1.0
                else:
                    r = stats.wilcoxon(a, bb)
                    stat, p = float(r.statistic), float(r.pvalue)
            else:
                a, bb = a_df["error"].to_numpy(), b_df["error"].to_numpy()
                test = "mannwhitney_u"
                if np.ptp(np.concatenate([a, bb])) == 0:
                    stat, p = np.nan, 1.0
                else:
                    r = stats.mannwhitneyu(a, bb, alternative="two-sided")
                    stat, p = float(r.statistic), float(r.pvalue)
            # taille d'effet : probabilité qu'un run de b batte (erreur plus faible) un run de la référence
            p_better = float((bb[:, None] < a[None, :]).mean() + 0.5 * (bb[:, None] == a[None, :]).mean())
            rows.append(
                dict(function=func, dim=dim, pop=pop, ref=ref, algo=b, test=test,
                     n_ref=len(a), n_algo=len(bb),
                     mean_ref=a.mean() if len(a) else np.nan, mean_algo=bb.mean() if len(bb) else np.nan,
                     median_ref=np.median(a) if len(a) else np.nan,
                     median_algo=np.median(bb) if len(bb) else np.nan,
                     statistic=stat, p_raw=p, prob_algo_better=p_better)
            )
        # Kruskal-Wallis : tous les algos ensemble
        groups = [data[a]["error"].to_numpy() for a in algos if len(data[a]) > 0]
        if len(groups) >= 3:
            if np.ptp(np.concatenate(groups)) == 0:
                h, p = np.nan, 1.0
            else:
                r = stats.kruskal(*groups)
                h, p = float(r.statistic), float(r.pvalue)
            kw_rows.append(dict(function=func, dim=dim, pop=pop, n_algos=len(groups), H=h, p=p))

    pw = pd.DataFrame(rows)
    kw = pd.DataFrame(kw_rows)
    if pw.empty:
        return pw, kw
    if scope == "global":
        pw["p_holm"] = holm(pw["p_raw"].to_numpy())
    else:  # une famille = les comparaisons contre la référence pour une même (fonction, dim, pop)
        pw["p_holm"] = np.nan
        for _, idx in pw.groupby(["function", "dim", "pop"]).groups.items():
            pw.loc[idx, "p_holm"] = holm(pw.loc[idx, "p_raw"].to_numpy())

    def decide(r):
        if r["p_holm"] < alpha:
            return "+" if r["median_algo"] < r["median_ref"] else "-"
        return "="

    pw["decision"] = pw.apply(decide, axis=1)
    return pw, kw


# --------------------------------------------------------------------------- tableaux
def quality_tables(df, algos, ref, pw, outdir):
    sym = {}
    if not pw.empty:
        for r in pw.itertuples():
            sym[(r.function, r.dim, r.pop, r.algo)] = r.decision
    tex_sym = {"+": "$+$", "-": "$-$", "=": "$\\approx$"}
    funcs = ordered_functions(df)
    agg = df.groupby(["function", "dim", "pop", "algo"])["error"].agg(["mean", "std"]).reset_index()
    for dim in sorted(df["dim"].unique()):
        header = ["Fonction", "Pop"] + algos
        rows_md, rows_tex, rows_csv = [], [], []
        for f in funcs:
            for pop in sorted(df["pop"].unique()):
                sub = agg[(agg.function == f) & (agg["dim"] == dim) & (agg["pop"] == pop)].set_index("algo")
                if sub.empty:
                    continue
                best = sub["mean"].idxmin()
                md, tx, cs = [TITLES.get(f, f), str(pop)], [TITLES.get(f, f), str(pop)], [f, pop]
                for a in algos:
                    if a not in sub.index:
                        md.append("-"); tx.append("-"); cs.append("")
                        continue
                    m, s = sub.loc[a, "mean"], sub.loc[a, "std"]
                    s = 0.0 if np.isnan(s) else s
                    s_md = f"{sci(m)} ± {sci(s)}"
                    s_tx = f"${sci_tex(m)}\\pm{sci_tex(s)}$"
                    if a != ref and (f, dim, pop, a) in sym:
                        d = sym[(f, dim, pop, a)]
                        s_md += f" ({d})"
                        s_tx += f" {tex_sym[d]}"
                    if a == best:
                        s_md, s_tx = f"**{s_md}**", f"\\textbf{{{s_tx}}}"
                    md.append(s_md); tx.append(s_tx); cs.append(f"{m:.6e};{s:.6e}")
                rows_md.append(md); rows_tex.append(tx); rows_csv.append(cs)
        base = os.path.join(outdir, "tables", f"quality_D{dim}")
        note = (f"\nErreur = f(x) - f*, moyenne ± écart-type (ddof=1) sur les runs. Gras = meilleure moyenne. "
                f"Symbole (test contre {ref}) : + significativement meilleur, - significativement pire, "
                f"= pas de différence significative.\n")
        write(base + ".md", f"### Qualité des solutions, Dim = {dim}\n\n" + md_table(header, rows_md) + note)
        write(base + ".tex", tex_table(
            header, rows_tex,
            f"Erreur moyenne $\\pm$ écart-type, Dim = {dim}. Symboles : test contre {ref.replace('_', chr(92) + '_')}.",
            f"tab:quality_d{dim}"))
        pd.DataFrame(rows_csv, columns=["function", "pop"] + [f"{a}(mean;std)" for a in algos]).to_csv(
            base + ".csv", index=False)


def time_tables(df, algos, ref, outdir):
    agg = df.groupby(["dim", "pop", "algo"])["time_s"].agg(["mean", "std"]).reset_index()
    others = [a for a in algos if a != ref]
    header = ["Dim", "Pop", f"{ref} (s)"] + [f"{a} (s) [speedup]" for a in others]
    rows_md, rows_tex, rows_csv = [], [], []
    for dim in sorted(df["dim"].unique()):
        for pop in sorted(df["pop"].unique()):
            sub = agg[(agg["dim"] == dim) & (agg["pop"] == pop)].set_index("algo")
            if ref not in sub.index:
                continue
            tref = sub.loc[ref, "mean"]
            md = [str(dim), str(pop), f"{tref:.3f}"]
            tx = [str(dim), str(pop), f"{tref:.3f}"]
            cs = [dim, pop, tref]
            for a in others:
                if a not in sub.index:
                    md.append("-"); tx.append("-"); cs += ["", ""]
                    continue
                t = sub.loc[a, "mean"]
                sp = tref / t
                md.append(f"{t:.3f} [x{sp:.1f}]")
                tx.append(f"{t:.3f} [$\\times${sp:.1f}]")
                cs += [t, sp]
            rows_md.append(md); rows_tex.append(tx); rows_csv.append(cs)
    csv_cols = ["dim", "pop", f"T_{ref}"]
    for a in others:
        csv_cols += [f"T_{a}", f"speedup_{a}"]
    base = os.path.join(outdir, "tables", "time_speedup")
    note = (f"\nTemps moyens (s) sur toutes les fonctions et tous les runs. Speedup = T_{ref} / T_algo "
            f"(un speedup < 1 signifie que l'algorithme est plus lent que la référence).\n")
    write(base + ".md", "### Temps d'exécution et speedup\n\n" + md_table(header, rows_md) + note)
    write(base + ".tex", tex_table(header, rows_tex, "Temps d'exécution moyen (s) et speedup.", "tab:time"))
    pd.DataFrame(rows_csv, columns=csv_cols).to_csv(base + ".csv", index=False)
    return agg


def stats_summary(pw, kw, ref, alpha, paired, scope, outdir):
    if pw.empty:
        return
    tests = "Wilcoxon (rang signé, runs appariés)" if paired else "Mann-Whitney U (échantillons indépendants)"
    lines = [
        "### Bilan des tests statistiques\n",
        f"- Référence : `{ref}` ; test : {tests} ; seuil alpha = {alpha}",
        f"- Correction de Holm, famille = " + ("tous les tests" if scope == "global"
                                                else "les comparaisons d'une même (fonction, dim, pop)"),
        "- `+` : l'algorithme est significativement meilleur que la référence ; `-` significativement pire ; "
        "`=` différence non significative\n",
        "#### Par algorithme et dimension\n",
    ]
    cnt = pw.groupby(["algo", "dim", "decision"]).size().unstack(fill_value=0)
    for c in ["+", "=", "-"]:
        if c not in cnt.columns:
            cnt[c] = 0
    rows = [[a, str(d), str(r["+"]), str(r["="]), str(r["-"])] for (a, d), r in cnt.iterrows()]
    lines.append(md_table(["Algorithme", "Dim", "+", "=", "-"], rows))
    if not kw.empty:
        n_sig = int((kw["p"] < alpha).sum())
        lines.append(f"\nKruskal-Wallis (tous algorithmes) : {n_sig} configurations sur {len(kw)} "
                     f"avec p < {alpha} (p brut, sans correction).\n")
    write(os.path.join(outdir, "tables", "stats_summary.md"), "\n".join(lines))


# --------------------------------------------------------------------------- figures
def style_of(algos):
    return {a: (f"C{i % 10}", MARKERS[i % len(MARKERS)]) for i, a in enumerate(algos)}


def fig_convergence(cv, algos, funcs, dim, pop, outdir):
    sty = style_of(algos)
    sub = cv[(cv["dim"] == dim) & (cv["pop"] == pop)]
    if sub.empty:
        return
    n = len(funcs)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.7 * nrows), squeeze=False)
    for ax, f in zip(axes.ravel(), funcs):
        for a in algos:
            s = sub[(sub["function"] == f) & (sub["algo"] == a)]
            if s.empty:
                continue
            q = s.groupby("evals")["error"].quantile([0.25, 0.5, 0.75]).unstack()
            ax.plot(q.index, q[0.5], color=sty[a][0], label=a)
            ax.fill_between(q.index, q[0.25], q[0.75], color=sty[a][0], alpha=0.15)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(TITLES.get(f, f)); ax.set_xlabel("Évaluations de la fonction"); ax.set_ylabel("Erreur f(x) - f*")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend()
    fig.suptitle(f"Convergence (médiane et intervalle interquartile), Dim = {dim}, Pop = {pop}", fontsize=9)
    save_fig(fig, outdir, f"convergence_D{dim}_P{pop}")


def fig_boxplot(df, algos, funcs, dim, pop, outdir):
    sub = df[(df["dim"] == dim) & (df["pop"] == pop)]
    if sub.empty:
        return
    n = len(funcs)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.7 * nrows), squeeze=False)
    for ax, f in zip(axes.ravel(), funcs):
        data = [sub[(sub["function"] == f) & (sub["algo"] == a)]["error_plot"].to_numpy() for a in algos]
        try:
            ax.boxplot(data, tick_labels=[a.replace("DE_", "") for a in algos], showfliers=True)
        except TypeError:  # matplotlib < 3.9 (ex. Anaconda) : l'argument s'appelait "labels"
            ax.boxplot(data, labels=[a.replace("DE_", "") for a in algos], showfliers=True)
        ax.set_yscale("log"); ax.set_title(TITLES.get(f, f)); ax.set_ylabel("Erreur finale")
        ax.tick_params(axis="x", labelrotation=20)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Distribution de l'erreur finale, Dim = {dim}, Pop = {pop}", fontsize=9)
    save_fig(fig, outdir, f"boxplot_D{dim}_P{pop}")


def fig_time(agg, algos, ref, outdir):
    sty = style_of(algos)
    dims = sorted(agg["dim"].unique())
    pops = sorted(agg["pop"].unique())

    # temps en fonction de Pop, une case par Dim
    fig, axes = plt.subplots(1, len(dims), figsize=(3.0 * len(dims), 2.8), squeeze=False)
    for ax, d in zip(axes[0], dims):
        for a in algos:
            s = agg[(agg["dim"] == d) & (agg["algo"] == a)].sort_values("pop")
            ax.errorbar(s["pop"], s["mean"], yerr=s["std"].fillna(0), color=sty[a][0], marker=sty[a][1],
                        capsize=2, label=a)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(pops); ax.set_xticklabels(pops)
        ax.set_title(f"Dim = {d}"); ax.set_xlabel("Taille de population"); ax.set_ylabel("Temps (s)")
    axes[0][0].legend()
    save_fig(fig, outdir, "time_vs_pop")

    # temps en fonction de Dim, une case par Pop
    fig, axes = plt.subplots(1, len(pops), figsize=(3.0 * len(pops), 2.8), squeeze=False)
    for ax, p in zip(axes[0], pops):
        for a in algos:
            s = agg[(agg["pop"] == p) & (agg["algo"] == a)].sort_values("dim")
            ax.errorbar(s["dim"], s["mean"], yerr=s["std"].fillna(0), color=sty[a][0], marker=sty[a][1],
                        capsize=2, label=a)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(dims); ax.set_xticklabels(dims)
        ax.set_title(f"Pop = {p}"); ax.set_xlabel("Dimension"); ax.set_ylabel("Temps (s)")
    axes[0][0].legend()
    save_fig(fig, outdir, "time_vs_dim")

    # speedup
    others = [a for a in algos if a != ref]
    if not others:
        return
    fig, axes = plt.subplots(1, len(dims), figsize=(3.0 * len(dims), 2.8), squeeze=False)
    for ax, d in zip(axes[0], dims):
        tref = agg[(agg["dim"] == d) & (agg["algo"] == ref)].set_index("pop")["mean"]
        for a in others:
            s = agg[(agg["dim"] == d) & (agg["algo"] == a)].set_index("pop")["mean"]
            sp = (tref / s).dropna().sort_index()
            ax.plot(sp.index, sp.values, color=sty[a][0], marker=sty[a][1], label=a)
        ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8)  # en dessous : plus lent que la référence
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(pops); ax.set_xticklabels(pops)
        ax.set_title(f"Dim = {d}"); ax.set_xlabel("Taille de population"); ax.set_ylabel(f"Speedup vs {ref}")
    axes[0][0].legend()
    save_fig(fig, outdir, "speedup")


# --------------------------------------------------------------------------- main
def middle(values):
    v = sorted(values)
    return v[len(v) // 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--convergence")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--reference", help="algorithme de référence (défaut : DE_seq s'il existe, sinon le premier)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--paired", action="store_true", help="Wilcoxon signé (runs appariés par numéro de run)")
    ap.add_argument("--holm-scope", choices=["config", "global"], default="config")
    ap.add_argument("--conv-dim", type=int)
    ap.add_argument("--conv-pop", type=int)
    ap.add_argument("--all-configs", action="store_true", help="courbes/boîtes pour toutes les (dim, pop)")
    args = ap.parse_args()

    df = load_results(args.results)
    algos = list(dict.fromkeys(df["algo"]))
    ref = args.reference or ("DE_seq" if "DE_seq" in algos else algos[0])
    if ref not in algos:
        sys.exit(f"Référence inconnue : {ref}. Algorithmes présents : {algos}")
    funcs = ordered_functions(df)

    # contrôle de santé des données : 10 runs attendus par configuration
    counts = df.groupby(["algo", "function", "dim", "pop"]).size()
    if counts.nunique() > 1:
        warnings.warn("Nombre de runs différent selon les configurations :\n" + str(counts[counts != counts.max()]))

    print(f"Algorithmes : {algos} | référence : {ref} | fonctions : {funcs}")
    pw, kw = run_stats(df, algos, ref, args.alpha, args.paired, args.holm_scope)
    tdir = os.path.join(args.outdir, "tables")
    os.makedirs(tdir, exist_ok=True)
    if not pw.empty:
        pw.to_csv(os.path.join(tdir, "stats_pairwise.csv"), index=False)
    if not kw.empty:
        kw.to_csv(os.path.join(tdir, "stats_kruskal.csv"), index=False)
    quality_tables(df, algos, ref, pw, args.outdir)
    agg = time_tables(df, algos, ref, args.outdir)
    stats_summary(pw, kw, ref, args.alpha, args.paired, args.holm_scope, args.outdir)

    fig_time(agg, algos, ref, args.outdir)
    dims, pops = sorted(df["dim"].unique()), sorted(df["pop"].unique())
    configs = [(d, p) for d in dims for p in pops] if args.all_configs else [
        (args.conv_dim or middle(dims), args.conv_pop or middle(pops))]
    cv = load_convergence(args.convergence) if args.convergence else None
    for d, p in configs:
        fig_boxplot(df, algos, funcs, d, p, args.outdir)
        if cv is not None:
            fig_convergence(cv, algos, funcs, d, p, args.outdir)

    print(f"Terminé. Résultats dans : {args.outdir}/tables et {args.outdir}/figures")


if __name__ == "__main__":
    main()
