"""Control 2 capture: residual stream at the answer position.

One forward per (n, rho, seed) on the CUED arm, keeping:
  resid      (n_layers+1, d_model) at answer_pos -- probe site
  v1_id      correct target (the cued item's value)
  comp_lg    competitor by LOGIT mass: argmax_{i>0} p(v_i).
             NOTE this is defined by the model's own output, so a probe
             recovering it at the final layer has learned W_U, not a
             representation. Layer-resolved onset is the real question.
  comp_att   competitor by MECHANISM: argmax_{i>0} of the cued-item
             attention share averaged over the frozen retrieval heads
             (D3 selection, analysis.select_heads). Non-circular.
  A_logit    ridge target
  value_ids  the in-context value set (for rank-based readouts)

Seeds 0-29: the trained probes need more samples than d_model, and 10
seeds x 8 N x 3 rho = 240 is far too few. 30 seeds gives 720 per model.

Written by Claude against Julia's spec (see the LLM-usage accounting in the application).
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import capture  # noqa: E402
import channels  # noqa: E402
import items  # noqa: E402
from analysis import load_model_grid, select_heads  # noqa: E402

N_MAX = 256
NS = [2, 4, 8, 16, 32, 64, 128, 256]
RHOS = ["low", "mid", "high"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(range(30)))
    ap.add_argument("--ns", type=int, nargs="+", default=NS)
    ap.add_argument("--rhos", nargs="+", default=RHOS)
    args = ap.parse_args()

    base = pathlib.Path(__file__).resolve().parents[1]
    tag = args.model.split("/")[-1]
    out = base / "runs" / "probe" / f"{tag}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"{out} exists, skipping", flush=True)
        return

    # frozen retrieval heads: reuse the D3 selection from the completed grid
    _, cells = load_model_grid(tag)
    heads, _ = select_heads(cells)
    print(f"{tag}: retrieval heads {heads}", flush=True)

    runner = capture.Runner(args.model, dtype=getattr(torch, args.dtype))
    pools = items.build_pools(runner.tokenizer, n_max=N_MAX)

    R, meta = [], []
    t0 = time.time()
    for rho in args.rhos:
        for n in args.ns:
            for seed in args.seeds:
                p = items.build_prompt(pools, n=n, rho=rho, seed=seed,
                                       n_max=N_MAX, cue="cued")
                cap = runner.forward(p.input_ids, want_resid=True)
                scores = runner.answer_scores(cap)
                rows = runner.attention_rows(scores)
                att = channels.attention_channel(scores, rows, p)
                lg = channels.logit_channel(cap.logits, p)

                pv = lg["p_values"].numpy()
                comp_lg = int(np.argmax(pv[1:]) + 1) if n > 1 else 0
                a = att["a_items"].numpy()          # (L, H, n)
                a_sel = np.mean([a[l, h] for l, h in heads], axis=0)
                comp_att = int(np.argmax(a_sel[1:]) + 1) if n > 1 else 0

                R.append(cap.resid.numpy().astype(np.float32))
                meta.append((n, RHOS.index(rho), seed,
                             p.value_ids[0],
                             p.value_ids[comp_lg],
                             p.value_ids[comp_att],
                             lg["A_logit"], lg["leakage"],
                             int(lg["top1_is_cued_value"])))
    m = np.array([x[:6] + x[8:] for x in meta], dtype=np.int64)
    np.savez_compressed(
        out,
        resid=np.stack(R),                     # (cells, L+1, d_model)
        n=m[:, 0], rho=m[:, 1], seed=m[:, 2],
        v1_id=m[:, 3], comp_lg_id=m[:, 4], comp_att_id=m[:, 5],
        top1_is_v1=m[:, 6],
        A_logit=np.array([x[6] for x in meta], dtype=np.float32),
        leakage=np.array([x[7] for x in meta], dtype=np.float32),
        rho_labels=np.array(RHOS),
        heads=np.array(heads),
    )
    print(f"{tag}: {len(R)} cells in {time.time()-t0:.0f}s -> {out} "
          f"({out.stat().st_size/1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
