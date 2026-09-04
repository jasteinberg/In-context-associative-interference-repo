"""P2 residual decomposition: cumulant truncation vs everything else.

In the saved (beta-absorbed) units, item i's score is S_i (span-LSE)
and its mass is proportional to exp(S_i), so A_att = 1/2 iff
    exp(S_1) = sum_{i>=2} exp(S_i) = exp(Sbar) (N-1) exp(G),
    G = ln < exp(eta) >_emp,  eta_i = S_i - Sbar  (i >= 2).
Hence EXACTLY, per cell:  ln(N-1) = Delta - G  at the crossover.
G is the empirical cumulant generating function at t = 1:
    G = k2/2 + k3/6 + k4/24 + ...
P2 keeps k2 only, so the ladder residual  r = ln N*_meas - pred  is
    r = sigma^2/2 - G   (+ interpolation/median noise).
Predictions compared at the measured crossover x*:
    gauss : Delta - sigma^2/2              (P2 as plotted)
    k3    : Delta - sigma^2/2 - k3/6        (one more cumulant)
    exact : Delta - G                        (all cumulants; residual
                                              must vanish up to the
                                              median/interp noise)
    gaussN: Delta - E_MC[G | Gaussian eta, sigma^2, N-1 samples]
            (finite-sample Gaussian null: the annealed e^{sigma^2/2}
             overestimates a max-dominated sum -- the REM reading)
Sink/filler mass cannot enter: A_att is a ratio over item spans.

Written by Claude against Julia's spec (see the LLM-usage accounting in the application).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import analysis  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "runs" / "analysis"
RNG = np.random.default_rng(0)


_GN_CACHE = {}


def gauss_finite_G(sigma2, m, draws=4000):
    """E[ ln (1/m) sum_{j<=m} exp(eta_j) ] for eta ~ N(0, sigma2), by MC.
    Cached on (round(sigma2, 2), m)."""
    key = (round(float(sigma2), 2), int(m))
    if key not in _GN_CACHE:
        s = np.sqrt(max(sigma2, 0.0))
        eta = RNG.standard_normal((draws, m)) * s
        _GN_CACHE[key] = float(np.mean(logsumexp(eta, axis=1) - np.log(m)))
    return _GN_CACHE[key]


def cell_moments(S):
    """S: (n,) item scores, S[0] cued. Returns Delta, sigma2, G, k3, GN."""
    oth = S[1:].astype(np.float64)
    m = len(oth)
    eta = oth - oth.mean()
    delta = float(S[0] - oth.mean())
    s2 = float(eta.var(ddof=1)) if m > 1 else 0.0
    G = float(logsumexp(eta) - np.log(m))
    k3 = float(np.mean(eta ** 3)) if m > 2 else 0.0
    return delta, s2, G, k3, gauss_finite_G(s2, m)


def head_frame_ext(cells, heads, test_seeds):
    rows = []
    for (n, rho, seed, cue), z in cells.items():
        if cue != "cued" or seed not in test_seeds:
            continue
        S_all = z["S_items"]
        for (l, h) in heads:
            d, s2, G, k3, GN = cell_moments(S_all[l, h])
            rows.append({"n": n, "rho": rho, "seed": seed, "layer": l,
                         "head": h, "A_att": float(z["A_att"][l, h]),
                         "Delta": d, "sigma2": s2, "G": G, "k3": k3,
                         "GN": GN,
                         "sink_mass": float(z["sink_mass"][l, h]),
                         "filler_mass": float(z["filler_mass"][l, h])})
    return pd.DataFrame(rows)


PREDS = {"gauss": lambda g: g["Delta"] - g["sigma2"] / 2,
         "k3": lambda g: g["Delta"] - g["sigma2"] / 2 - g["k3"] / 6,
         "exact": lambda g: g["Delta"] - g["G"],
         "gaussN": lambda g: g["Delta"] - g["GN"]}


def ladder(hf):
    """One row per (head, rho): x* and every prediction at x*."""
    rows = []
    cols = ["A_att", "Delta", "sigma2", "G", "k3", "GN",
            "sink_mass", "filler_mass"]
    for (l, h, rho), g in hf.groupby(["layer", "head", "rho"]):
        gm = g.groupby("n")[cols].median()
        ns = gm.index.values
        x = np.log(ns - 1.0)
        a = np.clip(gm["A_att"].values, 1e-6, 1 - 1e-6)
        xs, cens = analysis.crossover(ns, np.log(a) - np.log1p(-a),
                                      return_censored=True)
        if not np.isfinite(xs):
            continue
        rec = {"layer": l, "head": h, "rho": rho, "ln_nstar": xs,
               "censored": bool(cens)}
        for k, f in PREDS.items():
            rec["res_" + k] = xs - float(np.interp(xs, x, f(gm).values))
        rec["G-s2/2"] = float(np.interp(xs, x, (gm["G"] - gm["sigma2"] / 2).values))
        rec["k3/6"] = float(np.interp(xs, x, (gm["k3"] / 6).values))
        rec["GN-s2/2"] = float(np.interp(xs, x, (gm["GN"] - gm["sigma2"] / 2).values))
        rec["s2"] = float(np.interp(xs, x, gm["sigma2"].values))
        # REM freezing line in beta-absorbed units: sigma^2 = 2 ln(N-1)
        rec["freeze"] = rec["s2"] / (2 * xs) if xs > 0 else np.nan
        rec["sink"] = float(np.interp(xs, x, gm["sink_mass"].values))
        rec["filler"] = float(np.interp(xs, x, gm["filler_mass"].values))
        rows.append(rec)
    return pd.DataFrame(rows)


def q(s):
    s = s.dropna()
    return f"{s.median():+.3f} [{s.quantile(.25):+.3f},{s.quantile(.75):+.3f}] n={len(s)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-4B-Base"])
    ap.add_argument("--sel-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for tag in args.models:
        _, cells = analysis.load_model_grid(tag)
        seeds = sorted({k[2] for k in cells})
        test = {s for s in seeds if s not in args.sel_seeds}
        heads, _ = analysis.select_heads(cells, tuple(args.sel_seeds),
                                         k=args.k)
        hf = head_frame_ext(cells, heads, test)
        lad = ladder(hf)
        u = lad[~lad.censored]
        lines = [f"# P2 residual decomposition -- {tag}",
                 f"heads {heads}; test seeds {sorted(test)}; "
                 f"{len(u)} uncensored / {len(lad)} points",
                 "residual = ln N*_meas - pred  (median [IQR], uncensored)"]
        for k in PREDS:
            lines.append(f"  {k:7s} {q(u['res_' + k])}")
        lines.append("per class (gauss / exact):")
        for rho, g in u.groupby("rho"):
            lines.append(f"  {rho:5s} {q(g['res_gauss'])}  /  {q(g['res_exact'])}")
        lines.append("truncation terms at x* (median [IQR]):")
        for c in ["G-s2/2", "k3/6", "GN-s2/2", "s2", "freeze", "sink",
                  "filler"]:
            lines.append(f"  {c:8s} {q(u[c])}")
        # attribution: r_gauss = (s2/2 - G) + noise; split G - s2/2 into
        # finite-N Gaussian part and the non-Gaussian remainder
        lines.append("attribution of r_gauss (medians):")
        lines.append(f"  finite-N Gaussian part  -(GN - s2/2) = {-u['GN-s2/2'].median():+.3f}")
        lines.append(f"  non-Gaussian part       -(G - GN)    = {-(u['G-s2/2'] - u['GN-s2/2']).median():+.3f}")
        lines.append(f"  of which k3/6           -k3/6        = {-u['k3/6'].median():+.3f}")
        lines.append(f"  median/interp noise      res_exact   = {u['res_exact'].median():+.3f}")
        txt = "\n".join(lines) + "\n"
        print(txt)
        (OUT / f"p2_residual_{tag}.txt").write_text(txt)
        lad.to_csv(OUT / f"p2_residual_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
