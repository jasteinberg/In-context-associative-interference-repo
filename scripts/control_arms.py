"""Controls 1b (reverse cue) and 3 (ask-the-model).

analysis.logit_frame is hardcoded to the ("cued", "absent") pair, so
this generalizes it: any (measure, null) cue pair sharing a context.

  1b  ("reverse", "revnull") -- cue is the VALUE, target set is the
      entity set. Tests whether the interference law is symmetric
      under exchanging which side of the pair is the key.
  3   ("ask", "asknull")     -- same context, question appended.
      `high` class only (the base prompt must be a well-formed
      sentence for a natural-language question to be well-posed).

Reported against the forward pair ("cued", "absent") as the baseline.

Written by Claude against Julia's spec (see the LLM-usage accounting in the application).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis import CLASS_LABEL, crossover, load_model_grid  # noqa: E402

ANALYSIS = Path(__file__).resolve().parents[1] / "runs" / "analysis"
MODELS = ["Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-4B-Base"]
SEEDS = tuple(range(10))
NS = [2, 4, 8, 16, 32, 64, 128, 256]
PAIRS = [("cued", "absent"), ("reverse", "revnull"), ("ask", "asknull")]


def pair_frame(cells, measure, null, seeds=SEEDS, rhos=None):
    """One row per (n, rho, seed) for a (measure, null) cue pair.

    Mirrors analysis.logit_frame exactly, but for an arbitrary pair.
    A_corr = 1 - q_1/sum_i q_i with q_i = p_measure(t_i)/p_null(t_i),
    the Holtzman PMI_DC correction over whichever target set the arm
    normalizes on (values for forward/ask, entities for reverse).
    """
    rows = []
    for (n, rho, seed, cue), z in cells.items():
        if cue != measure or seed not in seeds:
            continue
        if rhos is not None and rho not in rhos:
            continue
        p_m = z["p_values"]
        rec = {"n": n, "rho": rho, "seed": seed,
               "A_logit": float(z["A_logit"]),
               "leakage": float(z["leakage"]),
               "top1_is_target": bool(z["top1_is_cued_value"])}
        zn = cells.get((n, rho, seed, null))
        if zn is not None:
            q = p_m / np.clip(zn["p_values"], 1e-30, None)
            rec["A_corr"] = float(1.0 - q[0] / q.sum())
            rec["null_leakage"] = float(zn["leakage"])
        rows.append(rec)
    return pd.DataFrame(rows)


def ln_star(g, col):
    """ln N* and censoring flag from a per-N median frame."""
    if col not in g:
        return np.nan, True
    a = np.clip(g[col].values.astype(float), 1e-9, 1 - 1e-9)
    return crossover(g.index.values, np.log(a) - np.log1p(-a),
                     return_censored=True)


def report(model_tag):
    _, cells = load_model_grid(model_tag)
    have = {c for (_, _, _, c) in cells}
    out = [f"\n{'=' * 74}",
           "CONTROLS 1b (reverse cue) and 3 (ask-the-model)",
           f"model: {model_tag}   seeds: {len(SEEDS)}",
           "A_corr = D1b prior correction against the arm's own null "
           "twin.",
           "1b normalizes over the ENTITY set; forward/ask over the "
           "VALUE set.",
           "=" * 74]

    for measure, null in PAIRS:
        if measure not in have:
            out.append(f"\n  [{measure}: no cells on disk, skipped]")
            continue
        rhos = ["high"] if measure == "ask" else ["low", "mid", "high"]
        if null not in have:
            out.append(f"\n  [{measure}: null twin {null!r} missing -- "
                       f"raw only, no D1b correction]")
        for rho in rhos:
            f = pair_frame(cells, measure, null, rhos=[rho])
            if f.empty:
                continue
            g = f.groupby("n").median(numeric_only=True).reindex(NS)
            acc = f.groupby("n")["top1_is_target"].mean().reindex(NS)
            out.append(f"\n  arm={measure:<8} rho={rho:<5} "
                       f"({CLASS_LABEL[rho]})")
            cols = [c for c in ("A_logit", "A_corr") if c in g]
            hdr = "".join(f"{c:>9}" for c in cols)
            out.append(f"    {'N':>5}{hdr} {'leak':>7} {'acc':>6}")
            for n in NS:
                vals = "".join(f"{g.loc[n, c]:>9.4f}" for c in cols)
                out.append(f"    {n:>5}{vals} {g.loc[n, 'leakage']:>7.3f} "
                           f"{acc.loc[n]:>6.2f}")
            for c in cols:
                s, cens = ln_star(g, c)
                flag = " (censored)" if cens else ""
                out.append(f"    ln N* [{c}] = {s:.3f}{flag}")
    return "\n".join(out)


def main():
    for m in MODELS:
        txt = report(m)
        print(txt)
        with (ANALYSIS / f"{m}.txt").open("a") as fh:
            fh.write(txt + "\n")


if __name__ == "__main__":
    main()
