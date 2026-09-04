"""Stage A figures: load curves for both channels, Delta trajectories,
and the pointwise P2 y=x plot.

Usage: python scripts/make_figures.py [--model-tag Qwen3-0.6B-Base]
       [--test-seeds 1 2 ...]
Writes figures/<model-tag>_*.png
"""
import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import analysis                  # noqa: E402

plt.rcParams.update({"grid.linestyle": ":", "grid.linewidth": 0.6,
                     "axes.grid": True, "grid.alpha": 0.55,
                     "font.size": 9, "figure.dpi": 150})
COL = {"low": "#1f77b4", "mid": "#2ca02c", "high": "#d62728"}


def logit(a):
    a = np.clip(np.asarray(a, float), 1e-6, 1 - 1e-6)
    return np.log(a) - np.log1p(-a)


def fig_load_curves(tag, hf, lf, out):
    """Both channels vs ln(N-1), one panel per class, ordered by
    measured Delta at N=8 (ascending: most confusable first)."""
    dbar = hf[hf.n == 8].groupby("rho")["Delta"].median()
    order = list(dbar.sort_values().index)
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.1), sharey=True)
    for ax, rho in zip(axes, order):
        g = hf[hf.rho == rho]
        gm = g.groupby("n")["A_att"].median()
        q1 = g.groupby("n")["A_att"].quantile(0.25)
        q3 = g.groupby("n")["A_att"].quantile(0.75)
        x = np.log(gm.index.values - 1.0)
        ax.plot(x, logit(gm.values), "o-", color=COL[rho],
                label=r"$A_{\rm att}$ (median D3 head)")
        ax.fill_between(x, logit(q1.values), logit(q3.values),
                        color=COL[rho], alpha=0.15, lw=0)
        b = lf[lf.rho == rho].groupby("n")[["A_logit", "A_corr"]].mean()
        xb = np.log(b.index.values - 1.0)
        ax.plot(xb, logit(b["A_logit"].values), "s--", color="0.35",
                mfc="none", label=r"$A_{\rm logit}$ raw")
        ax.plot(xb, logit(b["A_corr"].values), "^-", color="0.1",
                ms=4, label=r"$A_{\rm logit}$ corrected (D1b)")
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_title(f"{analysis.CLASS_LABEL[rho]} "
                     f"($\\bar\\Delta_{{N=8}}$ = {dbar[rho]:.1f})")
        ax.set_xlabel(r"$\ln(N-1)$")
    axes[0].set_ylabel(r"$\mathrm{logit}\,A$")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{tag}: load curves, both channels", y=1.02)
    fig.tight_layout()
    fig.savefig(out / f"{tag}_load_curves.png", bbox_inches="tight")
    plt.close(fig)


def fig_p2_pointwise(tag, hf, out):
    """y = x: measured logit A_att vs ln(N-1) - Delta + sigma^2/2."""
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    sub = hf[np.isfinite(hf.logit_A_att)]
    for rho, g in sub.groupby("rho"):
        ax.plot(g["pred_logit_A"], g["logit_A_att"], "o", ms=3,
                alpha=0.45, color=COL[rho],
                label=analysis.CLASS_LABEL[rho])
    lo = min(sub["pred_logit_A"].min(), sub["logit_A_att"].min()) - 0.5
    hi = max(sub["pred_logit_A"].max(), sub["logit_A_att"].max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "-", color="0.3", lw=1, label=r"$y=x$")
    ax.set_xlabel(r"$\ln(N-1) - \Delta + \sigma^2/2$  (measured QK)")
    ax.set_ylabel(r"$\mathrm{logit}\,A_{\rm att}$  (measured mass)")
    ax.set_title(f"{tag}: pointwise P2 (D3 heads, all cells)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"{tag}_p2_pointwise.png", bbox_inches="tight")
    plt.close(fig)


def fig_delta(tag, hf, out):
    """Delta(N) and sigma^2(N) at D3 heads."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for key, ax in zip(["Delta", "sigma2"], axes):
        for rho, g in hf.groupby("rho"):
            gm = g.groupby("n")[key].median()
            ax.plot(np.log(gm.index.values - 1.0), gm.values, "o-",
                    color=COL[rho], label=analysis.CLASS_LABEL[rho])
        ax.set_xlabel(r"$\ln(N-1)$")
        ax.set_ylabel({"Delta": r"$\Delta$",
                       "sigma2": r"$\sigma^2$"}[key])
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{tag}: QK statistics vs load (median D3 head)", y=1.02)
    fig.tight_layout()
    fig.savefig(out / f"{tag}_qk_stats.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default="Qwen3-0.6B-Base")
    ap.add_argument("--sel-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--test-seeds", type=int, nargs="+", default=None)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    base = pathlib.Path(__file__).resolve().parents[1]
    out = base / "figures"
    out.mkdir(exist_ok=True)

    meta, cells = analysis.load_model_grid(args.model_tag)
    if args.test_seeds is None:
        seeds = sorted({k[2] for k in cells}) 
        test_seeds = [s for s in seeds if s not in args.sel_seeds]
    else:
        test_seeds = args.test_seeds
    heads, _ = analysis.select_heads(cells, sel_seeds=tuple(args.sel_seeds),
                                     k=args.k)
    hf = analysis.head_frame(cells, heads, test_seeds=set(test_seeds))
    hf["pred_logit_A"] = (np.log(hf["n"] - 1.0) - hf["Delta"]
                          + hf["sigma2"] / 2)
    hf["logit_A_att"] = logit(hf["A_att"])
    lf = analysis.logit_frame(cells, seeds=set(test_seeds))

    fig_load_curves(args.model_tag, hf, lf, out)
    fig_p2_pointwise(args.model_tag, hf, out)
    fig_delta(args.model_tag, hf, out)
    print("test seeds:", test_seeds)
    print("wrote 3 figures to", out)


if __name__ == "__main__":
    main()
