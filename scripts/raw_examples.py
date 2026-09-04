"""Raw examples for the write-up: randomly selected, not cherry-picked,
placed right after the executive summary.

Draws five (N, seed, arm) cells with a FIXED rng seed (auditable),
rebuilds the exact prompt, runs Qwen3-0.6B-Base, and prints: the
prompt with the cued item and the top-competitor item bracketed, the
cue tail, the model's greedy next token, and the top-3 next tokens.
Prompts are 1027 tokens, so the middle is elided in the print;
full text goes to runs/analysis/raw_examples.md.

Written by Claude against Julia's spec (see the LLM-usage accounting in the application).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
import items  # noqa: E402
from capture import Runner  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B-Base"
OUT = HERE / "runs" / "analysis" / "raw_examples.md"
RNG_SEED = 2026
DRAWS = [("cued", 8), ("cued", 64), ("cued", 256), ("ask", 256), ("cued", 32)]


def main():
    rng = random.Random(RNG_SEED)
    runner = Runner(MODEL, dtype=torch.float32)
    tok = runner.tokenizer
    pools = items.build_pools(tok, n_max=256)
    lines = [f"# Raw examples — {MODEL}", "",
             f"Selection: arms/N fixed as {DRAWS}; seed per draw from "
             f"random.Random({RNG_SEED}) over 0-9; class `high`. Greedy "
             "next token after the cue tail. [cued]=cued item, "
             "[top competitor]=the in-context item whose value receives "
             "the most probability among non-cued values.", ""]
    for arm, n in DRAWS:
        seed = rng.randrange(10)
        p = items.build_prompt(pools, n=n, rho="high", seed=seed,
                               n_max=256, cue=arm)
        cap = runner.forward(p.input_ids)
        probs = torch.softmax(cap.logits, -1).numpy()
        tid = np.array(p.target_ids)
        pv = probs[tid]
        comp = int(np.argmax(np.where(np.arange(len(tid)) == 0, -1, pv)))
        top3 = np.argsort(probs)[::-1][:3]
        greedy = tok.decode([int(top3[0])])
        # render context with markers
        pieces = []
        for i, (s, e) in enumerate(p.item_spans):
            txt = tok.decode(p.input_ids[s:e])
            if i == 0:
                txt = f"[cued]{txt}[/cued]"
            elif i == comp:
                txt = f"[top competitor]{txt}[/top competitor]"
            pieces.append((s, txt))
        for s, e in p.filler_spans:
            pieces.append((s, tok.decode(p.input_ids[s:e])))
        pieces.sort()
        ctx = "".join(t for _, t in pieces)
        tail = tok.decode(p.input_ids[p.cue_span[0]:p.cue_span[1]])
        correct = tok.decode([int(tid[0])])
        lines += [f"## {arm}, N={n}, seed={seed}", "",
                  f"Correct value: `{correct}`;  model greedy: `{greedy}` "
                  f"({'CORRECT' if int(top3[0]) == int(tid[0]) else 'WRONG'});  "
                  f"p(correct)={pv[0]:.3f}, p(top competitor)={pv[comp]:.3f}, "
                  f"A_logit={1 - pv[0] / pv.sum():.3f}, leakage={1 - pv.sum():.3f}",
                  f"Top-3 next tokens: " + ", ".join(
                      f"`{tok.decode([int(t)])}` {probs[t]:.3f}" for t in top3),
                  "", "Context (items and filler in their actual order; "
                  "1027 tokens):", "", "```", ctx, "```",
                  f"Cue tail: `{tail}` → model continues with `{greedy}`", ""]
    OUT.write_text("\n".join(lines))
    print("\n".join(l for l in lines if not l.startswith(("```", " ")) and len(l) < 400))


if __name__ == "__main__":
    main()
