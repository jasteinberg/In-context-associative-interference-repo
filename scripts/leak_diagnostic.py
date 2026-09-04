"""Leakage decomposition diagnostic (leakage decomposition, 23 Aug).

Splits the answer-position mass into: in-context values, off-context
surnames (rest of the value pool), generic function words, in-context
entities, and remainder. Hypothesis: total surname-pool mass is
monotone in N (format prior strengthening); the leakage *fall* at
N = 256 is pool exhaustion -- 256 of ~298 surnames are in-set, so
surname-shaped mass lands in-set by construction.

Usage: python scripts/leak_diagnostic.py [--model Qwen/Qwen3-0.6B-Base]
"""
import argparse
import pathlib
import sys

import torch

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import capture   # noqa: E402
import items     # noqa: E402

FUNC_WORDS = [" the", " a", " an", " to", " of", " in", " not", " also",
              " very", " one", ",", ".", " and", " that", " it", " called"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ns", type=int, nargs="+",
                    default=[2, 4, 8, 16, 32, 64, 128, 256])
    args = ap.parse_args()

    runner = capture.Runner(args.model, dtype=getattr(torch, args.dtype))
    pools = items.build_pools(runner.tokenizer, n_max=256)
    pool_ids = torch.tensor(sorted(pools.values.values()))
    func_ids = []
    for w in FUNC_WORDS:
        ids = runner.tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            func_ids.append(ids[0])
    func_ids = torch.tensor(func_ids)

    print(f"{'N':>4} {'rho':4s} | {'in-set':>7} {'off-pool':>8} "
          f"{'pool tot':>8} | {'func':>6} {'entity':>7} {'rest':>6}")
    agg = {}
    for rho in ["low", "mid", "high"]:
        for n in args.ns:
            row = torch.zeros(5)
            for seed in args.seeds:
                p = items.build_prompt(pools, n=n, rho=rho, seed=seed,
                                       n_max=256)
                cap = runner.forward(p.input_ids)
                pr = torch.softmax(cap.logits, -1)
                inset = pr[torch.tensor(p.value_ids)].sum()
                pool = pr[pool_ids].sum()
                func = pr[func_ids].sum()
                ent = pr[torch.tensor(p.entity_ids)].sum()
                row += torch.tensor([inset, pool - inset, pool, func, ent])
            row /= len(args.seeds)
            rest = 1.0 - row[2] - row[3] - row[4]
            agg[(n, rho)] = row
            print(f"{n:>4} {rho:4s} | {row[0]:7.3f} {row[1]:8.3f} "
                  f"{row[2]:8.3f} | {row[3]:6.3f} {row[4]:7.3f} "
                  f"{rest:6.3f}", flush=True)
    print("\npool total monotone in N per class?")
    for rho in ["low", "mid", "high"]:
        seq = [float(agg[(n, rho)][2]) for n in args.ns]
        mono = all(b >= a - 0.02 for a, b in zip(seq, seq[1:]))
        print(f"  {rho}: {['%.2f' % v for v in seq]}  "
              f"{'monotone(±0.02)' if mono else 'NOT monotone'}")


if __name__ == "__main__":
    main()
