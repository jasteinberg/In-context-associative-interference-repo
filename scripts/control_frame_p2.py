"""Does the P2 identity hold within each cue frame?

The ask arm does not degrade behaviourally with load. Two readings:

  (a) the frame ESCAPES the law -- retrieval is doing something else,
      and P2 describes only document continuation;
  (b) the frame MOVES ALONG the law -- the QA framing raises the key
      contrast Delta, pushing ln N* beyond the grid, and the same
      zero-parameter identity predicts each arm's own crossover.

(b) is the strong result: one equation, frame entering as a measured
parameter rather than an escape hatch.

Protocol mirrors P2 exactly: heads selected WITHIN each arm on a
held-out selection seed (0) by cued attention share at N=8, evaluated
on test seeds 1-9, `high` class only. x* = ln(N-1) where A_att = 1/2,
prediction = Delta - sigma^2/2 interpolated at x*.

Written by Claude against Julia's spec (see the LLM-usage accounting in the application).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis import crossover, head_frame, load_model_grid  # noqa: E402

ANALYSIS = Path(__file__).resolve().parents[1] / "runs" / "analysis"
MODELS = ["Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-4B-Base"]
ARMS = ["cued", "relation", "repeat", "repeatphrase", "ask"]
SEL_SEED, TEST_SEEDS = 0, tuple(range(1, 10))


def select_heads_arm(cells, cue, n_sel=8, k=8, rho="high"):
    """Top-k heads by cued attention share at N=n_sel, on the selection
    seed, WITHIN the given arm. Same rule as analysis.select_heads but
    arm-parameterized -- the QA frame need not recruit the same heads."""
    key = (n_sel, rho, SEL_SEED, cue)
    if key not in cells:
        return None
    ms = cells[key]["cued_share"]
    flat = np.argsort(ms, axis=None)[::-1][:k]
    return [tuple(map(int, np.unravel_index(i, ms.shape))) for i in flat]


def arm_crossovers(cells, cue, heads):
    """P2 rows for one arm: measured x*, predicted Delta - sigma^2/2."""
    hf = head_frame(cells, heads, test_seeds=set(TEST_SEEDS), cues=(cue,))
    hf = hf[hf.rho == "high"]
    rows = []
    for (l, h), g in hf.groupby(["layer", "head"]):
        gm = g.groupby("n")[["A_att", "Delta", "sigma2"]].median()
        ns = gm.index.values
        x = np.log(ns - 1.0)
        a = np.clip(gm["A_att"].values, 1e-6, 1 - 1e-6)
        xstar, cens = crossover(ns, np.log(a) - np.log1p(-a),
                                return_censored=True)
        if not np.isfinite(xstar):
            continue
        pred = float(np.interp(xstar, x,
                               (gm["Delta"] - gm["sigma2"] / 2).values))
        rows.append({"layer": l, "head": h, "ln_nstar": xstar,
                     "pred": pred, "censored": bool(cens),
                     "Delta_at_8": float(gm["Delta"].get(8, np.nan)),
                     "sigma2_at_8": float(gm["sigma2"].get(8, np.nan))})
    return pd.DataFrame(rows)


def report(model_tag):
    _, cells = load_model_grid(model_tag)
    have = {c for (_, _, _, c) in cells}
    out = [f"\n{'=' * 74}",
           "P2 IDENTITY WITHIN EACH CUE FRAME (high class)",
           f"model: {model_tag}   select seed {SEL_SEED}, test seeds "
           f"{TEST_SEEDS[0]}-{TEST_SEEDS[-1]}",
           "x* = ln(N-1) at A_att = 1/2;  pred = Delta - sigma^2/2 at x*",
           "residual = x* - pred  (P2 predicts 0)",
           "=" * 74,
           f"  {'arm':13s}{'n_head':>7}{'cens':>6}{'med x*':>9}"
           f"{'med pred':>10}{'residual':>10}{'Delta@8':>9}"]
    for cue in ARMS:
        if cue not in have:
            out.append(f"  {cue:13s}   [no cells]")
            continue
        heads = select_heads_arm(cells, cue)
        df = arm_crossovers(cells, cue, heads)
        if df.empty:
            out.append(f"  {cue:13s}   [no head reaches A_att = 1/2 "
                       f"within N <= 256 -- crossover is a bound]")
            continue
        un = df[~df.censored]
        n_un, n_c = len(un), int(df.censored.sum())
        if n_un == 0:
            out.append(f"  {cue:13s}{len(df):>7}{n_c:>6}   "
                       f"[all censored]")
            continue
        res = un.ln_nstar - un.pred
        out.append(f"  {cue:13s}{n_un:>7}{n_c:>6}"
                   f"{un.ln_nstar.median():>9.3f}{un.pred.median():>10.3f}"
                   f"{res.median():>+10.3f}{df.Delta_at_8.median():>9.3f}")
    return "\n".join(out)


def main():
    for m in MODELS:
        txt = report(m)
        print(txt)
        with (ANALYSIS / f"{m}.txt").open("a") as fh:
            fh.write(txt + "\n")


if __name__ == "__main__":
    main()
