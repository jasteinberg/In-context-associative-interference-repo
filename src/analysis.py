"""Analysis for the fixed-T grid: head selection (D3), the two
channels' load curves, and the pointwise P2 test.

What each object tests (be precise in the write-up):
- Attention channel: a_i is mathematically softmax(S_i), and Delta,
  sigma^2 are moments of the same S_i, so
      logit A_att(N) = ln(N-1) - Delta(N) + sigma^2(N)/2
  is a *cumulant truncation* statement: two moments suffice
  (Gaussianity of the score ensemble) and the interference sum is
  extensive in N (eta iid across items -- the claim rho stresses).
  It is not an independent measurement of behaviour.
- Logit channel: behaviour. Raw A_logit, and the D1b prior-corrected
  version q_i \\propto p_cued(v_i) / p_null(v_i) from the null twin.
  Agreement of its crossover with the attention channel's is the
  independent test.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

GRID = pathlib.Path(__file__).resolve().parents[1] / "runs" / "grid"

# Signed off 23 Aug: measured Delta is the axis; the lexical classes
# are the sampling device and are named by content, never by the
# intended (and unrealized) rho ordering.
CLASS_LABEL = {"low": "low (mixed classes)", "mid": "mid (concrete nouns)",
               "high": "high (person names)"}


def load_model_grid(model_tag):
    """Return (meta, dict[(n, rho, seed, cue)] -> npz)."""
    d = GRID / model_tag
    meta = json.loads((d / "meta.json").read_text())
    cells = {}
    for f in sorted(d.glob("N*.npz")):
        z = np.load(f, allow_pickle=False)
        key = (int(z["n"]), str(z["rho"]), int(z["seed"]), str(z["cue"]))
        cells[key] = z
    return meta, cells


def select_heads(cells, sel_seeds=(0,), n_sel=8, k=8, rhos=("low", "mid", "high")):
    """D3: rank heads by cued attention share a_1 (absolute mass) at
    N ~= n_sel on the selection seed(s), averaged over rho cells;
    return the frozen top-k as [(layer, head), ...]."""
    acc = None
    m = 0
    for rho in rhos:
        for s in sel_seeds:
            z = cells[(n_sel, rho, s, "cued")]
            acc = z["cued_share"] if acc is None else acc + z["cued_share"]
            m += 1
    mean_share = acc / m
    flat = np.argsort(mean_share, axis=None)[::-1][:k]
    return [tuple(map(int, np.unravel_index(i, mean_share.shape)))
            for i in flat], mean_share


def head_frame(cells, heads, test_seeds, cues=("cued",)):
    """Tidy frame: one row per (n, rho, seed, cue, layer, head)."""
    rows = []
    for (n, rho, seed, cue), z in cells.items():
        if seed not in test_seeds or cue not in cues:
            continue
        for (l, h) in heads:
            A = float(z["A_att"][l, h])
            rows.append({
                "n": n, "rho": rho, "seed": seed, "cue": cue,
                "layer": l, "head": h,
                "A_att": A,
                "logit_A_att": float(np.log(A) - np.log1p(-A))
                if 0 < A < 1 else np.nan,
                "Delta": float(z["Delta"][l, h]),
                "sigma2": float(z["sigma2"][l, h]),
                "cued_share": float(z["cued_share"][l, h]),
                "item_mass": float(z["item_mass"][l, h]),
                "sink_mass": float(z["sink_mass"][l, h]),
            })
    return pd.DataFrame(rows)


def logit_frame(cells, seeds):
    """Behavioural channel per (n, rho, seed): raw and prior-corrected.

    Corrected: q_i \\propto p_cued(v_i) / p_null(v_i) over the value set
    of the shared context; A_corr = 1 - q_1 / sum_i q_i.
    """
    rows = []
    for (n, rho, seed, cue), z in cells.items():
        if cue != "cued" or seed not in seeds:
            continue
        p_c = z["p_values"]
        rec = {"n": n, "rho": rho, "seed": seed,
               "A_logit": float(z["A_logit"]),
               "leakage": float(z["leakage"]),
               "top1_is_cued": bool(z["top1_is_cued_value"])}
        znull = cells.get((n, rho, seed, "absent"))
        if znull is not None:
            p_n = znull["p_values"]
            q = p_c / np.clip(p_n, 1e-30, None)
            rec["A_corr"] = float(1.0 - q[0] / q.sum())
            rec["null_leakage"] = float(znull["leakage"])
        rows.append(rec)
    return pd.DataFrame(rows)


def crossover(ns, logit_a, return_censored=False):
    """ln N* by linear interpolation of logit A across 0 in ln(N-1).

    Left-censored (curve already above 0 at the smallest N): the grid
    edge is returned with censored=True -- it is a bound, not a
    measurement. No crossing at all returns nan.
    """
    x = np.log(np.asarray(ns, float) - 1.0)
    y = np.asarray(logit_a, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    out = (np.nan, False)
    if len(x) >= 2 and y.max() >= 0:
        if y[0] >= 0:
            out = (float(x[0]), True)
        else:
            j = int(np.argmax(y >= 0))
            x0, x1, y0, y1 = x[j - 1], x[j], y[j - 1], y[j]
            out = (float(x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)), False)
    return out if return_censored else out[0]


def per_head_crossovers(cells, heads, test_seeds, rhos=("low", "mid", "high")):
    """The P2 crossover objects: for each (layer, head, rho), the
    measured crossover x* = ln(N-1) at A_att = 1/2 (median over test
    seeds), Delta - sigma^2/2 interpolated at x*, and a left-censoring
    flag (crossover at the grid edge is a bound, not a measurement).
    One row per (head, rho) that reaches A = 1/2 anywhere in range."""
    rows = []
    hf = head_frame(cells, heads, test_seeds=set(test_seeds))
    for (l, h, rho), g in hf.groupby(["layer", "head", "rho"]):
        gm = g.groupby("n")[["A_att", "Delta", "sigma2"]].median()
        ns = gm.index.values
        x = np.log(ns - 1.0)
        a = np.clip(gm["A_att"].values, 1e-6, 1 - 1e-6)
        xstar, cens = crossover(ns, np.log(a) - np.log1p(-a),
                                return_censored=True)
        if not np.isfinite(xstar):
            continue
        pred = np.interp(xstar, x, (gm["Delta"] - gm["sigma2"] / 2).values)
        rows.append({"layer": l, "head": h, "rho": rho,
                     "ln_nstar": xstar, "pred": float(pred),
                     "censored": bool(cens)})
    return pd.DataFrame(rows)
