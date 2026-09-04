"""Main fixed-T grid (write-up, Setup; D1b null twin).

For every cell (N, rho, seed) runs two forwards on the identical
context -- cue="cued" and cue="absent" -- and saves per-(layer, head)
arrays for every head, so D3 head selection happens in analysis, not
at runtime. Resumable: a cell whose npz exists is skipped.

Usage:
  python scripts/run_grid.py --model Qwen/Qwen3-0.6B-Base --dtype float32
      [--seeds 0 1 2] [--ns 2 4 8 16 32 64 128 256]
      [--rhos low mid high] [--validate]
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


def run_prompt(runner, pools, n, rho, seed, cue):
    p = items.build_prompt(pools, n=n, rho=rho, seed=seed, n_max=N_MAX,
                           cue=cue)
    cap = runner.forward(p.input_ids)
    scores = runner.answer_scores(cap)
    rows = runner.attention_rows(scores)
    att = channels.attention_channel(scores, rows, p)
    logit = channels.logit_channel(cap.logits, p)
    return p, att, logit


def save_cell(out_dir, n, rho, seed, cue, p, att, logit):
    tag = f"N{n}_{rho}_s{seed}_{cue}"
    np.savez_compressed(
        out_dir / f"{tag}.npz",
        **{k: v.numpy() for k, v in att.items()},
        p_values=logit["p_values"].numpy(),
        A_logit=logit["A_logit"], leakage=logit["leakage"],
        top1_id=logit["top1_id"],
        top1_is_cued_value=logit["top1_is_cued_value"],
        n=n, seed=seed, rho=rho, cue=cue,
        cue_entity=p.cue_entity_word,
        entity_words=np.array(p.entity_words),
        value_words=np.array(p.value_words),
    )
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ns", type=int, nargs="+",
                    default=[2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--rhos", nargs="+", default=["low", "mid", "high"])
    ap.add_argument("--cues", nargs="+", default=["cued", "absent"])
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    base = pathlib.Path(__file__).resolve().parents[1]
    out_dir = base / "runs" / "grid" / args.model.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = capture.Runner(args.model, dtype=getattr(torch, args.dtype))
    pools = items.build_pools(runner.tokenizer, n_max=N_MAX)
    if args.validate:
        pv = items.build_prompt(pools, n=8, rho="high", seed=0, n_max=32)
        atol = 1e-4 if args.dtype == "float32" else 5e-2
        print("validate:", runner.validate(pv.input_ids, atol=atol), flush=True)

    meta = {"model": args.model, "dtype": args.dtype, "n_max": N_MAX,
            "scaling": runner.scaling, "head_dim": runner.head_dim,
            "n_layers": runner.n_layers, "n_heads": runner.n_heads}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    todo = [(n, rho, seed, cue)
            for rho in args.rhos for n in args.ns
            for seed in args.seeds for cue in args.cues]
    t0, done = time.time(), 0
    for n, rho, seed, cue in todo:
        tag = f"N{n}_{rho}_s{seed}_{cue}"
        if (out_dir / f"{tag}.npz").exists():
            continue
        p, att, logit = run_prompt(runner, pools, n, rho, seed, cue)
        save_cell(out_dir, n, rho, seed, cue, p, att, logit)
        done += 1
        if cue == "cued":
            print(f"{tag:24s} A_logit={logit['A_logit']:.4f} "
                  f"L={logit['leakage']:.3f} "
                  f"top1={'Y' if logit['top1_is_cued_value'] else 'N'}",
                  flush=True)
    print(f"ran {done}/{len(todo)} cells in {time.time()-t0:.0f}s "
          f"-> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
