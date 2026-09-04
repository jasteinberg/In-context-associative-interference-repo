"""The P2 crossover figure: ln N* (measured from A_att curves) vs
Delta - sigma^2/2 (measured QK, interpolated at the crossover), one
point per (head, class, model size). Pre-registered: "Every (head, size,
rho) is a point on a y=x plot; scatter off the diagonal is the
finding."

Usage: python scripts/fig_p2_crossover.py
       [--models Qwen3-0.6B-Base Qwen3-1.7B-Base Qwen3-4B-Base]
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
MARK = {"Qwen3-0.6B-Base": "o", "Qwen3-1.7B-Base": "s",
        "Qwen3-4B-Base": "^", "Qwen3-8B-Base": "D"}
COL = {"low": "#1f77b4", "mid": "#2ca02c", "high": "#d62728"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["Qwen3-0.6B-Base", "Qwen3-1.7B-Base",
                             "Qwen3-4B-Base"])
    ap.add_argument("--sel-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    out = pathlib.Path(__file__).resolve().parents[1] / "figures"

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    all_pts = []
    for tag in args.models:
        meta, cells = analysis.load_model_grid(tag)
        seeds = sorted({k[2] for k in cells})
        test = [s for s in seeds if s not in args.sel_seeds]
        heads, _ = analysis.select_heads(
            cells, sel_seeds=tuple(args.sel_seeds), k=args.k)
        pf = analysis.per_head_crossovers(cells, heads, test)
        pf["model"] = tag
        all_pts.append(pf)
        for rho, g in pf[~pf.censored].groupby("rho"):
            ax.plot(g["pred"], g["ln_nstar"], MARK[tag], ms=5, mew=0.8,
                    mfc="none", color=COL[rho], alpha=0.85)
        gc = pf[pf.censored]
        if len(gc):
            ax.plot(gc["pred"], gc["ln_nstar"], "v", ms=4, mew=0.7,
                    mfc="none", color="0.6", alpha=0.8)
        print(f"{tag}: {len(pf)} (head, rho) pairs reach A=1/2 of "
              f"{args.k * 3}; {int(pf.censored.sum())} left-censored; "
              f"test seeds {test[0]}..{test[-1]}")

    import pandas as pd
    pts = pd.concat(all_pts)
    meas = pts[~pts.censored]
    lo = min(pts["pred"].min(), pts["ln_nstar"].min()) - 0.4
    hi = max(pts["pred"].max(), pts["ln_nstar"].max()) + 0.4
    ax.plot([lo, hi], [lo, hi], "-", color="0.3", lw=1)
    res = meas["ln_nstar"] - meas["pred"]
    print(f"pooled residual (uncensored only): median {res.median():+.2f}, "
          f"IQR [{res.quantile(.25):+.2f}, {res.quantile(.75):+.2f}], "
          f"n = {len(meas)} (+{int(pts.censored.sum())} censored)")
    for rho, c in COL.items():
        ax.plot([], [], "o", color=c, mfc="none",
                label=analysis.CLASS_LABEL[rho])
    for tag, m in MARK.items():
        if tag in args.models:
            ax.plot([], [], m, color="0.4", mfc="none",
                    label=tag.replace("Qwen3-", "").replace("-Base", ""))
    ax.plot([], [], "v", color="0.6", mfc="none",
            label="left-censored (bound)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlabel(r"$\Delta - \sigma^2/2$ at the crossover (measured QK)")
    ax.set_ylabel(r"$\ln N^*$ (measured $A_{\rm att}$ crossover)")
    ax.set_title("P2: parameter-free crossover identity")
    fig.tight_layout()
    fig.savefig(out / "p2_crossover_ladder.png", bbox_inches="tight")
    pts.to_csv(out / "p2_crossover_points.csv", index=False)
    print("wrote", out / "p2_crossover_ladder.png")


if __name__ == "__main__":
    main()
