"""Control 1a: the true-value channel, un-renormalized.

D1 reports A_logit = 1 - p(v_1)/sum_i p(v_i), so the "true-value share"
1 - A_logit is that same number and carries no independent information.
What the renormalization removes is the off-set mass. This script reads
the completed grids and reports the true-value channel *before* that
division, so the write-up can say what D1 hides:

  p1_raw   = p(v_1)                      raw next-token probability
  share    = sum_i p(v_i) = 1 - leakage  in-set mass
  acc      = 1[argmax_v p(v) = v_1]      vocab-wide accuracy
  A_logit  = 1 - p1_raw/share            (raw)   as already reported
  A_corr   = D1b prior-corrected                  as already reported

Under the lemma p_1 = [1 + (N-1)e^{-beta*Delta + beta^2 sigma^2/2}]^{-1},
so for A -> 1 we expect ln p1_raw ~ ln N* - ln(N-1): slope -1 against
ln(N-1), the same law read on the un-renormalized quantity. Divergence
between that slope and the logit-channel slope is leakage structure,
not retrieval.

Written by Claude against Julia's spec (see the LLM-usage accounting in the application).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis import (CLASS_LABEL, crossover, load_model_grid,  # noqa: E402
                      logit_frame)

ANALYSIS = Path(__file__).resolve().parents[1] / "runs" / "analysis"
MODELS = ["Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-4B-Base"]
SEEDS = tuple(range(10))
NS = [2, 4, 8, 16, 32, 64, 128, 256]
RHOS = ["low", "mid", "high"]


def true_value_frame(cells, seeds=SEEDS):
    """One row per (n, rho, seed): the un-renormalized true-value channel."""
    rows = []
    for (n, rho, seed, cue), z in cells.items():
        if cue != "cued" or seed not in seeds:
            continue
        p = z["p_values"]
        share = float(p.sum())
        rows.append({
            "n": n, "rho": rho, "seed": seed,
            "p1_raw": float(p[0]),
            "share": share,
            "leakage": float(z["leakage"]),
            "acc": bool(z["top1_is_cued_value"]),
            "A_logit": float(z["A_logit"]),
        })
    return pd.DataFrame(rows)


def windowed_slope(x, y, lo=0.1, hi=0.9, mask=None):
    """OLS slope of y on x over the rows where mask holds (default all
    finite). Returns (slope, n_points)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    if ok.sum() < 3:
        return np.nan, int(ok.sum())
    return float(np.polyfit(x[ok], y[ok], 1)[0]), int(ok.sum())


def report(model_tag):
    meta, cells = load_model_grid(model_tag)
    tv = true_value_frame(cells)
    lg = logit_frame(cells, SEEDS)
    out = [f"\n{'=' * 70}",
           "CONTROL 1a -- true-value channel, un-renormalized",
           f"model: {model_tag}   seeds: {len(SEEDS)}   "
           f"dtype: {meta.get('dtype', '?')}",
           "p1_raw = p(v_1); share = sum_i p(v_i) = 1 - leakage;",
           "acc = P[argmax over vocab == v_1]; A_logit as reported (D1).",
           "=" * 70]
    for rho in RHOS:
        g = (tv[tv.rho == rho].groupby("n")
             .agg(p1_raw=("p1_raw", "median"), share=("share", "median"),
                  acc=("acc", "mean"), A_logit=("A_logit", "median"))
             .reindex(NS))
        out.append(f"\n  rho={rho} ({CLASS_LABEL[rho]})")
        out.append(f"    {'N':>5} {'p1_raw':>9} {'share':>8} {'acc':>6} "
                   f"{'A_logit':>8}")
        for n, r in g.iterrows():
            out.append(f"    {n:>5} {r.p1_raw:>9.4f} {r.share:>8.4f} "
                       f"{r.acc:>6.2f} {r.A_logit:>8.4f}")

        lx = np.log(g.index.values.astype(float) - 1.0)
        s_p1, k1 = windowed_slope(lx, np.log(g.p1_raw.values))
        win = (g.A_logit.values > 0.1) & (g.A_logit.values < 0.9)
        s_win, k2 = windowed_slope(lx, np.log(g.p1_raw.values), mask=win)
        acc_star, acc_cens = acc_crossover(g.index.values, g.acc.values)
        out.append(f"    slope d ln p1_raw / d ln(N-1): {s_p1:+.3f} "
                   f"(n={k1});  windowed A in (0.1,0.9): {s_win:+.3f} "
                   f"(n={k2});  expected -1")
        flag = "  [right-censored: still >= 1/2 at N=256]" if acc_cens \
            else ""
        out.append(f"    accuracy crossover ln N* (acc = 1/2): "
                   f"{acc_star:.3f}{flag}")
    # corrected-channel crossovers for side-by-side comparison
    out.append("\n  D1/D1b logit-channel crossovers (for comparison)")
    for rho in RHOS:
        g = lg[lg.rho == rho].groupby("n").median(numeric_only=True).reindex(NS)
        for col in ("A_logit", "A_corr"):
            if col not in g:
                continue
            a = g[col].values.clip(1e-9, 1 - 1e-9)
            ln_star, cens = crossover(g.index.values,
                                      np.log(a) - np.log1p(-a),
                                      return_censored=True)
            flag = " (censored)" if cens else ""
            out.append(f"    rho={rho:<5} {col:<8} ln N* = "
                       f"{ln_star:.3f}{flag}")
    return "\n".join(out)


def main():
    for m in MODELS:
        txt = report(m)
        print(txt)
        f = ANALYSIS / f"{m}.txt"
        with f.open("a") as fh:
            fh.write(txt + "\n")
        print(f"\n[appended to {f}]")



def acc_crossover(ns, acc):
    """ln N* where accuracy falls through 1/2, interpolating in ln(N-1).

    Accuracy is a DECREASING curve, so this is the mirror of
    analysis.crossover(): find the first N at which acc drops below 1/2
    and interpolate. Returns (ln N*, censored). Censored=True means the
    curve never reaches 1/2 within the grid -- the value is the right
    edge and is a bound, not a measurement. If accuracy is already
    below 1/2 at the smallest N there is no onset to locate and the
    left edge is returned, also flagged.
    """
    x = np.log(np.asarray(ns, float) - 1.0)
    y = np.asarray(acc, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return np.nan, True
    if y[0] < 0.5:                      # already below at the left edge
        return float(x[0]), True
    if y.min() >= 0.5:                  # never crosses within the grid
        return float(x[-1]), True
    j = int(np.argmax(y < 0.5))
    x0, x1, y0, y1 = x[j - 1], x[j], y[j - 1], y[j]
    return float(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)), False


if __name__ == "__main__":
    main()
