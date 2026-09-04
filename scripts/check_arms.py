"""Regression + arm check for the items.py cue-arm extension.

Assertions rather than output to be read: the script exits nonzero on the first
failure, so it can gate a run instead of relying on someone noticing a line.

(1) cued/absent prompts must be byte-identical to the pre-change builder, or
    every existing npz becomes irreproducible.
(2) each arm must have the index arithmetic and the target set it claims. The
    expectations in ARMS are the values observed and reviewed on 26 Aug. A
    change to build_prompt that moves any of them is a decision to be made
    deliberately, not a detail to be discovered later in an analysis.
"""
import sys

sys.path.insert(0, "src")
from transformers import AutoTokenizer
from items import build_pools, build_prompt, W


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        sys.exit(1)


tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
pools = build_pools(tok, n_max=256)

T_BASE = 1 + 256 * W + 2                 # EOS sink + slot grid + " is" + value

# arm -> (which set the answer is drawn from, whether the cue entity appears
# in the context). The null twins are exactly the arms with cue_in_context False.
ARMS = {"cued":    ("values",   True),
        "absent":  ("values",   False),
        "reverse": ("entities", True),
        "revnull": ("entities", False),
        "ask":     ("values",   True)}

print("=== (1) regression: cued/absent unchanged ===")
bad = []
for n in [2, 4, 8, 16, 32, 64, 128, 256]:
    for rho in ["low", "mid", "high"]:
        for s in range(10):
            for cue in ["cued", "absent"]:
                p = build_prompt(pools, n=n, rho=rho, seed=s, cue=cue)
                ok = (len(p.input_ids) == T_BASE
                      and p.answer_pos == 1026
                      and p.cue_span == (1025, 1027)
                      and p.target_ids == p.value_ids
                      and p.input_ids[p.answer_pos] == pools.is_id
                      and all(e - s_ == W for s_, e in p.item_spans)
                      and len(p.filler_spans) == 256 - n)
                if not ok:
                    bad.append((n, rho, s, cue))
check("cued/absent prompts unchanged over all 480 cells", not bad,
      f"{len(bad)} failures, first at {bad[0] if bad else '-'}")

print("\n=== (2) arms, N=8 high seed=0 ===")
for cue, (want_target, want_in_ctx) in ARMS.items():
    p = build_prompt(pools, n=8, rho="high", seed=0, cue=cue)
    T = len(p.input_ids)
    tgt = "values" if p.target_ids == p.value_ids else "entities"
    in_ctx = p.cue_entity_word in (p.entity_words + p.value_words)
    print(f"{cue:8s} T={T:5d} answer_pos={p.answer_pos:5d} cue_span={p.cue_span} "
          f"target={tgt:8s} cue_in_context={in_ctx}")
    print(f"         tail={tok.decode(p.input_ids[p.cue_span[0]:])!r}")
    check(f"{cue}: answer drawn from the {want_target}", tgt == want_target,
          f"got {tgt}, target[0]={tok.decode([p.target_ids[0]])!r}")
    check(f"{cue}: cue entity in context is {want_in_ctx}", in_ctx == want_in_ctx)
    check(f"{cue}: answer_pos is the final position", p.answer_pos == T - 1,
          f"answer_pos={p.answer_pos}, T={T}")
    check(f"{cue}: cue span runs to the end of the prompt", p.cue_span[1] == T,
          f"cue_span={p.cue_span}, T={T}")
    # only the ask arm changes the prompt length, by the question frame it adds
    check(f"{cue}: prompt length", T > T_BASE if cue == "ask" else T == T_BASE,
          f"T={T}, T_base={T_BASE}")

print("\nall arm checks passed")
