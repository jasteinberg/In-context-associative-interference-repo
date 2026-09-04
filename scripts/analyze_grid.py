"""First-look analysis of a model's fixed-T grid: D3 head set, P1
slopes, pointwise P2 residuals, and both behavioural curves.

Usage: python scripts/analyze_grid.py [--model-tag Qwen3-0.6B-Base]
       [--sel-seeds 0] [--test-seeds 1 2] [--k 8]
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import analysis     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default="Qwen3-0.6B-Base")
    ap.add_argument("--sel-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--test-seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    pd.set_option("display.width", 200)

    meta, cells = analysis.load_model_grid(args.model_tag)
    heads, share = analysis.select_heads(
        cells, sel_seeds=tuple(args.sel_seeds), k=args.k)
    print(f"== {args.model_tag}  (beta = {meta['scaling']:.4f}) ==")
    print("D3 head set (by cued share at N=8, selection seeds "
          f"{args.sel_seeds}):")
    for (l, h) in heads:
        print(f"  L{l}H{h}: mean a_1 = {share[l, h]:.3f}")

    hf = analysis.head_frame(cells, heads, test_seeds=set(args.test_seeds))
    hf["lnNm1"] = np.log(hf["n"] - 1.0)
    hf["pred_logit_A"] = hf["lnNm1"] - hf["Delta"] + hf["sigma2"] / 2
    hf["resid"] = hf["logit_A_att"] - hf["pred_logit_A"]

    # P1: slope of logit A_att in ln(N-1), per (head, rho); windowed
    # to the unsaturated regime A in (0.1, 0.9), >= 3 N points
    print("\nP1 slopes (logit A_att vs ln(N-1), A in (0.1, 0.9); "
          "predict 1):")
    sub = hf[np.isfinite(hf.logit_A_att)]
    win = sub[(sub.A_att > 0.1) & (sub.A_att < 0.9)]
    slopes = []
    for (l, h), g in win.groupby(["layer", "head"]):
        parts = []
        for rho, gg in sorted(g.groupby("rho")):
            gm = gg.groupby("n")["logit_A_att"].median()
            if len(gm) < 3:
                parts.append(f"{rho}: --  ")
                continue
            b, a = np.polyfit(np.log(gm.index.values - 1.0), gm.values, 1)
            slopes.append(b)
            parts.append(f"{rho}: {b:+.2f}")
        print(f"  L{l}H{h}:  " + "   ".join(parts))
    if slopes:
        print(f"  pooled: median {np.median(slopes):+.2f}, "
              f"IQR [{np.percentile(slopes, 25):+.2f}, "
              f"{np.percentile(slopes, 75):+.2f}], n = {len(slopes)}")

    # P1 corrected (the restated P1): slope of
    # logit A + Delta - sigma^2/2 vs ln(N-1); predict 1 exactly
    cslopes = []
    win = win.assign(corr=win.logit_A_att + win.Delta - win.sigma2 / 2)
    for (l, h, rho), g in win.groupby(["layer", "head", "rho"]):
        gm = g.groupby("n")["corr"].median()
        if len(gm) < 3:
            continue
        b, a = np.polyfit(np.log(gm.index.values - 1.0), gm.values, 1)
        cslopes.append(b)
    if cslopes:
        print(f"P1 corrected slope (logit A + Delta - s2/2 vs ln(N-1); "
              f"predict 1): median {np.median(cslopes):+.2f}, "
              f"IQR [{np.percentile(cslopes, 25):+.2f}, "
              f"{np.percentile(cslopes, 75):+.2f}], n = {len(cslopes)}")

    print("\nPointwise P2 residual logit A - [ln(N-1) - Delta + s2/2] "
          "(predict 0), by N (median over heads/rho/seeds):")
    r = sub.groupby("n")["resid"].agg(["median", "mean", "std", "count"])
    print(r.round(2).to_string())

    print("\nA_att at D3 heads, median over heads/seeds, by (rho, N):")
    piv = sub.pivot_table(index="rho", columns="n", values="A_att",
                          aggfunc="median")
    print(piv.round(3).to_string())

    print("\nDelta at D3 heads, median, by (rho, N):")
    piv = sub.pivot_table(index="rho", columns="n", values="Delta",
                          aggfunc="median")
    print(piv.round(2).to_string())

    lf = analysis.logit_frame(cells, seeds=set(args.test_seeds))
    print("\nBehavioural channel, mean over seeds, by (rho, N):")
    for col in ["A_logit", "A_corr", "leakage"]:
        if col not in lf:
            continue
        piv = lf.pivot_table(index="rho", columns="n", values=col,
                             aggfunc="mean")
        print(f"-- {col} --")
        print(piv.round(3).to_string())

    print("\nCrossovers ln N* (x = ln(N-1) at A = 1/2):")
    for rho, g in lf.groupby("rho"):
        gm = g.groupby("n")[["A_logit"] +
                            (["A_corr"] if "A_corr" in g else [])].mean()
        ns = gm.index.values
        for col in gm.columns:
            y = np.log(gm[col]) - np.log1p(-gm[col])
            print(f"  {rho:4s} {col}: ln N* = "
                  f"{analysis.crossover(ns, y.values):.2f}")
    att_x = {}
    for rho, g in sub.groupby("rho"):
        gm = g.groupby("n")["logit_A_att"].median()
        att_x[rho] = analysis.crossover(gm.index.values, gm.values)
        print(f"  {rho:4s} A_att (median D3 head): ln N* = {att_x[rho]:.2f}")


if __name__ == "__main__":
    main()
