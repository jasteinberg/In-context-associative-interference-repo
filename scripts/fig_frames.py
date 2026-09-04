"""Write-up figures E2 and E3.

E2  figures/p2_frames.png : per-head measured x* = ln(N*-1) vs
    predicted Delta - sigma^2/2, one panel per model, five cue formats as
    marker shapes (heads selected WITHIN each arm on seed 0, test
    seeds 1-9, `high` class, uncensored heads filled / censored open).
    The y = x line is the zero-parameter identity.

E3  figures/channel_gap.png : per model, A_att at that arm's heads
    (median over test seeds) and top-1 accuracy vs N, for the
    completion (`cued`) and question (`ask`) cue formats. Attention
    disperses under both; behaviour fails only under completion.

Written by Claude against Julia's spec (see the LLM-usage accounting in the application).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))
from analysis import head_frame, load_model_grid  # noqa: E402
from control_frame_p2 import (ARMS, MODELS, TEST_SEEDS,  # noqa: E402
                              arm_crossovers, select_heads_arm)

FIG = HERE / "figures"
MARK = {"cued": "o", "ask": "s", "relation": "^", "repeat": "v",
        "repeatphrase": "D"}
LABEL = {"cued": "completion", "ask": "question", "relation": "relation",
         "repeat": "repeat", "repeatphrase": "repeat phrase"}
NS = [2, 4, 8, 16, 32, 64, 128, 256]


def fig_e2(grids):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True, sharey=True)
    for ax, m in zip(axes, MODELS):
        cells = grids[m]
        for arm in ARMS:
            heads = select_heads_arm(cells, arm)
            if heads is None:
                continue
            df = arm_crossovers(cells, arm, heads)
            if df.empty:
                continue
            for cens, sub in df.groupby("censored"):
                ax.scatter(sub.pred, sub.ln_nstar, marker=MARK[arm], s=48,
                           facecolors="none" if cens else "C0",
                           edgecolors="C0", linewidths=1.2,
                           label=LABEL[arm] if not cens else None)
        lim = (0, 6)
        ax.plot(lim, lim, "k--", lw=1, label="ln N* = Δ − σ²/2")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(m.replace("-Base", ""))
        ax.set_xlabel("predicted  Δ − σ²/2")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("measured  ln(N* − 1)")
    h, l = axes[0].get_legend_handles_labels()
    # legend from all panels (some arms censored in some models)
    seen = {}
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(ll, hh)
    axes[-1].legend(seen.values(), seen.keys(), fontsize=8, loc="lower right")
    fig.suptitle("P2 within each cue format: per-head crossover vs zero-parameter "
                 "prediction (high class)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "p2_frames.png", dpi=160)
    plt.close(fig)


def arm_curves(cells, arm):
    """Median A_att over the arm's heads and test seeds, and mean
    top-1 accuracy over all 10 seeds, per N (high class)."""
    heads = select_heads_arm(cells, arm)
    hf = head_frame(cells, heads, test_seeds=set(TEST_SEEDS), cues=(arm,))
    hf = hf[hf.rho == "high"]
    a_att = hf.groupby("n")["A_att"].median().reindex(NS)
    acc = {}
    for n in NS:
        v = [bool(cells[(n, "high", s, arm)]["top1_is_cued_value"])
             for s in range(10) if (n, "high", s, arm) in cells]
        acc[n] = np.mean(v) if v else np.nan
    return a_att.values, np.array([acc[n] for n in NS])


def fig_e3(grids):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharey=True)
    for ax, m in zip(axes, MODELS):
        cells = grids[m]
        for arm, c in (("cued", "C0"), ("ask", "C3")):
            a_att, acc = arm_curves(cells, arm)
            ax.plot(NS, a_att, "-", color=c, marker=MARK[arm],
                    label=f"{LABEL[arm]}: A_att (retrieval heads)")
            ax.plot(NS, acc, "--", color=c, marker=MARK[arm], mfc="none",
                    label=f"{LABEL[arm]}: top-1 accuracy")
        ax.axhline(0.5, color="k", lw=0.8, alpha=0.5)
        ax.set_xscale("log", base=2); ax.set_xticks(NS)
        ax.set_xticklabels(NS); ax.set_xlabel("N (items in context)")
        ax.set_title(m.replace("-Base", "")); ax.grid(alpha=0.3)
    axes[0].set_ylabel("mass / accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Channel gap: attention disperses under both cue formats; "
                 "behaviour fails only under completion (high class, 10 seeds)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "channel_gap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    grids = {m: load_model_grid(m)[1] for m in MODELS}
    fig_e2(grids)
    fig_e3(grids)
    print("wrote", FIG / "p2_frames.png", FIG / "channel_gap.png")
