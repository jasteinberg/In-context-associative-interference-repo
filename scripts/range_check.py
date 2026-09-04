"""Range check -- the gate (pre-registered kill criteria).

Measure A at N = 2 and N = 256 only, before the grid: if those two are
not separable, nothing between them will be. Runs both channels at
every (layer, head), saves per-prompt arrays to runs/range_check/, and
prints a summary keyed to the gate decision.

Usage:
  python scripts/range_check.py [--model Qwen/Qwen3-0.6B-Base]
      [--dtype float32] [--seeds 0 1 2] [--ns 2 256] [--validate]
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import capture      # noqa: E402
import channels     # noqa: E402
import items        # noqa: E402

N_MAX = 256


def run_cell(runner, pools, n, rho, seed, out_dir):
    p = items.build_prompt(pools, n=n, rho=rho, seed=seed, n_max=N_MAX)
    t0 = time.time()
    cap = runner.forward(p.input_ids)
    scores = runner.answer_scores(cap)
    rows = runner.attention_rows(scores)
    att = channels.attention_channel(scores, rows, p)
    logit = channels.logit_channel(cap.logits, p)
    dt = time.time() - t0

    tag = f"N{n}_{rho}_s{seed}"
    np.savez_compressed(
        out_dir / f"{tag}.npz",
        **{k: v.numpy() for k, v in att.items()},
        n=n, seed=seed,
    )
    rec = {
        "n": n, "rho": rho, "seed": seed, "sec": round(dt, 2),
        **{k: v for k, v in logit.items() if k != "p_values"},
        "cued_entity": p.entity_words[0], "cued_value": p.value_words[0],
    }
    # summary over heads: best head by cued mass, and its A_att
    cs = att["cued_share"]
    l, h = np.unravel_index(int(cs.argmax()), cs.shape)
    rec["best_head"] = [int(l), int(h)]
    rec["best_cued_share"] = float(cs[l, h])
    rec["best_A_att"] = float(att["A_att"][l, h])
    rec["best_Delta"] = float(att["Delta"][l, h])
    rec["best_sigma2"] = float(att["sigma2"][l, h])
    rec["best_sink_mass"] = float(att["sink_mass"][l, h])
    rec["best_filler_mass"] = float(att["filler_mass"][l, h])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ns", type=int, nargs="+", default=[2, 256])
    ap.add_argument("--rhos", nargs="+", default=["low", "mid", "high"])
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    base = pathlib.Path(__file__).resolve().parents[1]
    out_dir = base / "runs" / "range_check" / args.model.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = capture.Runner(args.model, dtype=getattr(torch, args.dtype))
    pools = items.build_pools(runner.tokenizer, n_max=N_MAX)
    if args.validate:
        pv = items.build_prompt(pools, n=8, rho="high", seed=0, n_max=32)
        atol = 1e-4 if args.dtype == "float32" else 5e-2
        print("validate:", runner.validate(pv.input_ids, atol=atol))

    recs = []
    for rho in args.rhos:
        for n in args.ns:
            for seed in args.seeds:
                rec = run_cell(runner, pools, n, rho, seed, out_dir)
                recs.append(rec)
                print(
                    f"N={n:3d} rho={rho:4s} s={seed} "
                    f"A_logit={rec['A_logit']:.4f} L={rec['leakage']:.4f} "
                    f"top1={'Y' if rec['top1_is_cued_value'] else 'N'} "
                    f"head=L{rec['best_head'][0]}H{rec['best_head'][1]} "
                    f"a1={rec['best_cued_share']:.3f} "
                    f"A_att={rec['best_A_att']:.4f} "
                    f"Dl={rec['best_Delta']:.2f}", flush=True,
                )
    meta = {"model": args.model, "dtype": args.dtype, "n_max": N_MAX,
            "scaling": runner.scaling, "records": recs}
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()
