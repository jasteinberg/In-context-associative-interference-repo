"""The two measurement channels (write-up, D1) and QK statistics (D3).

Span reduction: item i's per-head score is S_i = logsumexp of the head's
post-scaling answer-position scores over the item's W-token span. Since
the attention mass on a span is proportional to exp(logsumexp of its
scores), this is the exact reduction of a span to the one-key-per-item
object in the derivation: A_att computed from {S_i} equals the mass
ratio 1 - a_1 / sum_i a_i.

Delta and sigma^2 are moments of {S_i}: Delta = S_cued - mean(S_others),
sigma^2 = var(S_others). Scores are post-scaling, so beta is already
inside S and the P2 identity reads ln N* = Delta - sigma^2/2 in these
units; Runner.scaling records beta for the cross-family comparison.
"""
from __future__ import annotations

import torch


def span_lse(scores_ht, spans):
    """scores_ht: (H, T); spans: list of (start, end).
    Returns (H, n_spans) logsumexp per span."""
    return torch.stack(
        [torch.logsumexp(scores_ht[:, a:b], dim=-1) for a, b in spans], dim=-1
    )


def span_mass(rows_ht, spans):
    """rows_ht: (H, T) attention probabilities. Returns (H, n_spans)."""
    return torch.stack([rows_ht[:, a:b].sum(dim=-1) for a, b in spans], dim=-1)


def attention_channel(scores_lht, rows_lht, prompt):
    """Per (layer, head): A_att, Delta, sigma^2, and the mass ledger.

    Returns dict of (L, H) tensors. Mass ledger entries sum to 1 with
    'other' absorbing cue self-attention and any residual positions.
    """
    L, H, T = scores_lht.shape
    out = {}
    S = torch.stack(
        [span_lse(scores_lht[l], prompt.item_spans) for l in range(L)]
    )                                                  # (L, H, n)
    a_items = torch.stack(
        [span_mass(rows_lht[l], prompt.item_spans) for l in range(L)]
    )                                                  # (L, H, n)
    item_total = a_items.sum(-1)
    out["A_att"] = 1.0 - a_items[..., 0] / item_total.clamp_min(1e-30)
    out["cued_share"] = a_items[..., 0]                # a_1 (absolute mass)
    out["item_mass"] = item_total
    if prompt.filler_spans:
        f = torch.stack(
            [span_mass(rows_lht[l], prompt.filler_spans) for l in range(L)]
        )
        out["filler_mass"] = f.sum(-1)
    else:
        out["filler_mass"] = torch.zeros(L, H)
    out["sink_mass"] = rows_lht[..., 0]
    a, b = prompt.cue_span
    out["cue_mass"] = rows_lht[..., a:b].sum(-1)

    others = S[..., 1:]                                # non-cued items
    out["Delta"] = S[..., 0] - others.mean(-1)
    out["sigma2"] = (
        others.var(-1, unbiased=True) if others.shape[-1] > 1
        else torch.zeros(L, H)
    )
    out["S_cued"] = S[..., 0]
    out["S_items"] = S                                 # (L, H, n) for Stage B
    out["a_items"] = a_items
    return out


def logit_channel(logits, prompt):
    """Behavioural channel at the answer position (D1).

    A_logit = 1 - p(t_1) / sum_i p(t_i) over the in-context TARGET set;
    leakage L = 1 - sum_i p(t_i) reported separately, never folded in.

    The target set is prompt.target_ids, index 0 = the correct target.
    Forward arms ("cued"/"absent"/"ask") set it to the value set, so
    this is unchanged from the original D1 definition. Reverse arms
    ("reverse"/"revnull") set it to the entity set, since the cue is a
    value and the retrieved object is an entity.
    """
    p = torch.softmax(logits, dim=-1)
    tgt = list(prompt.target_ids)
    p_vals = p[torch.tensor(tgt)]
    total = p_vals.sum()
    top_id = int(p.argmax())
    return {
        "A_logit": float(1.0 - p_vals[0] / total.clamp_min(1e-30)),
        "leakage": float(1.0 - total),
        "p_cued_value": float(p_vals[0]),
        "top1_is_cued_value": top_id == tgt[0],
        "top1_id": top_id,
        "p_values": p_vals,          # (n,) over the target set, item order
    }
