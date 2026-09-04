"""Control 2: is the correct value still linearly decodable after
behaviour fails?  Three readouts per layer on the control-2 captures
(runs/probe/<model>.npz: resid at answer_pos, 30 seeds x 8 N x 3 rho).

1. Logit lens (zero parameters): p^(l) = softmax over the in-context
   value set of RMSNorm_f(h_l) W_U;  A^(l) = 1 - p_1 / sum_i p_i and
   top-1-is-v1.  Final layer must reproduce the saved A_logit.
2. Diagonal-bilinear pairwise probe (d parameters, logistic, no
   intercept):  P[v1 beats comp] = sigma( w . (h_l * (e_v1 - e_comp)) ),
   e = input embeddings.  Antisymmetric augmentation (x, 1), (-x, 0).
   Train seeds 0-19, test 20-29; accuracy per (N, layer), separately
   for comp_lg (model-defined, near-circular at the last layer) and
   comp_att (retrieval-head-defined, non-circular).  NOTE all three
   Qwen3 sizes tie embeddings, so e_v = W_U[v]; the probe is a learned
   diagonal reweighting of the model's own readout.
3. Ridge h_l -> A_logit (failure readability): test R^2 per layer.

Written by Claude against Julia's spec (see the LLM-usage accounting in the application).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from safetensors import safe_open
from sklearn.linear_model import LogisticRegression, Ridge
from transformers import AutoTokenizer

BASE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
import items  # noqa: E402

RHOS = ["low", "mid", "high"]


def load_readout(model):
    """(W_E, norm_w, eps) from the HF cache without instantiating the model."""
    d = snapshot_download(model, local_files_only=True)
    cfg = json.load(open(os.path.join(d, "config.json")))
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
    else:
        wm = {k: "model.safetensors" for k in
              ["model.embed_tokens.weight", "model.norm.weight"]}
    def get(name):
        with safe_open(os.path.join(d, wm[name]), "pt") as f:
            return f.get_tensor(name).float().numpy()
    assert cfg.get("tie_word_embeddings", False)
    return get("model.embed_tokens.weight"), get("model.norm.weight"), cfg["rms_norm_eps"]


def rmsnorm(h, w, eps):
    return h / np.sqrt(np.mean(h * h, axis=-1, keepdims=True) + eps) * w


def value_sets(model, n, rho, seed):
    tok = AutoTokenizer.from_pretrained(model)
    pools = items.build_pools(tok, n_max=256)
    return [items.build_prompt(pools, n=int(a), rho=RHOS[int(b)],
                               seed=int(c), n_max=256, cue="cued").value_ids
            for a, b, c in zip(n, rho, seed)]


def logit_lens(R, W, nw, eps, vsets, v1):
    """Per (cell, layer): A^(l), top1_is_v1."""
    C, L, _ = R.shape
    A = np.zeros((C, L)); top = np.zeros((C, L), bool)
    for c in range(C):
        Wv = W[vsets[c]]                              # (n, d)
        # transformers 4.57: hidden_states[-1] is ALREADY post-final-norm
        # (verified: raw h[-1] @ W_U reproduces the logits to 1e-5);
        # indices 0..L-1 are pre-norm. capture.py's docstring ("output
        # of block l-1, pre-layernorm") is wrong at the last index.
        hn = np.concatenate([rmsnorm(R[c, :-1], nw, eps), R[c, -1:]])
        z = hn @ Wv.T                                 # (L, n)
        z -= z.max(-1, keepdims=True); p = np.exp(z); p /= p.sum(-1, keepdims=True)
        assert vsets[c][0] == v1[c]
        A[c] = 1 - p[:, 0]; top[c] = p.argmax(-1) == 0
    return A, top


def bilinear_probe(R, W, v1, comp, train, test, ns, NCOL, C_reg=1.0):
    """Test accuracy per (N, layer) of P[v1 > comp] with a diagonal
    bilinear form. Cells where comp == v1 (n == 1 only) are excluded."""
    L = R.shape[1]
    keep = comp != v1
    D = W[v1] - W[comp]                               # (C, d)
    acc = np.full((len(ns), L), np.nan)
    for l in range(L):
        X = R[:, l, :] * D
        s = X[train & keep].std() + 1e-12
        Xtr = np.concatenate([X[train & keep], -X[train & keep]]) / s
        ytr = np.r_[np.ones((train & keep).sum()), np.zeros((train & keep).sum())]
        clf = LogisticRegression(fit_intercept=False, C=C_reg, max_iter=2000)
        clf.fit(Xtr, ytr)
        pred = clf.decision_function(X / s) > 0
        for j, n in enumerate(ns):
            m = test & keep & (NCOL == n)
            if m.any():
                acc[j, l] = pred[m].mean()
    return acc


def ridge_r2(R, y, train, test, alpha=None):
    """Test R^2 per layer of a ridge h_l -> y."""
    L = R.shape[1]; out = np.zeros(L)
    for l in range(L):
        X = R[:, l, :]; mu, sd = X[train].mean(0), X[train].std() + 1e-12
        Xs = (X - mu) / sd
        a = alpha or X.shape[1]                       # alpha ~ d: strong shrinkage
        rg = Ridge(alpha=a).fit(Xs[train], y[train])
        yp = rg.predict(Xs[test])
        out[l] = 1 - ((y[test] - yp) ** 2).sum() / ((y[test] - y[test].mean()) ** 2).sum()
    return out


def fmt_row(label, vals):
    return f"{label:>6s} " + " ".join(f"{v:5.2f}" if np.isfinite(v) else "  nan" for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base"])
    ap.add_argument("--train-seeds", type=int, default=20)
    ap.add_argument("--layer-stride", type=int, default=1)
    args = ap.parse_args()
    out = BASE / "runs" / "analysis"
    for model in args.models:
        tag = model.split("/")[-1]
        z = np.load(BASE / "runs" / "probe" / f"{tag}.npz")
        R = z["resid"][:, ::args.layer_stride, :].astype(np.float32)
        ncol, v1 = z["n"], z["v1_id"]
        ns = sorted(set(ncol.tolist()))
        train = z["seed"] < args.train_seeds; test = ~train
        W, nw, eps = load_readout(model)
        vsets = value_sets(model, ncol, z["rho"], z["seed"])
        A, top = logit_lens(R, W, nw, eps, vsets, v1)
        sane = np.abs(A[:, -1] - z["A_logit"]).max()
        lines = [f"# Control 2 probes -- {tag}  (layers 0..{R.shape[1]-1}, "
                 f"stride {args.layer_stride}; train seeds <{args.train_seeds})",
                 f"logit-lens final layer vs saved A_logit: max |diff| = {sane:.2e}",
                 "", "## Logit lens: median A^(l) per N (rows) x layer (cols)"]
        lines.append("   N   " + " ".join(f"{l:5d}" for l in range(R.shape[1])))
        for n in ns:
            lines.append(fmt_row(str(n), np.median(A[ncol == n], 0)))
        lines += ["", "## Logit lens: fraction top-1 = v1 per N x layer"]
        for n in ns:
            lines.append(fmt_row(str(n), top[ncol == n].mean(0)))
        for name, comp in [("comp_lg", z["comp_lg_id"]), ("comp_att", z["comp_att_id"])]:
            acc = bilinear_probe(R, W, v1, comp, train, test, ns, ncol)
            lines += ["", f"## Bilinear probe, {name}: test accuracy P[v1 > comp] per N x layer"]
            for j, n in enumerate(ns):
                lines.append(fmt_row(str(n), acc[j]))
            lines.append(fmt_row("mean", np.nanmean(acc, 0)))
        r2 = ridge_r2(R, z["A_logit"].astype(np.float64), train, test)
        # within-N: A_logit minus its per-N mean, so the probe cannot
        # score by reading N itself off the residual
        y = z["A_logit"].astype(np.float64)
        yw = y - np.array([y[ncol == n].mean() for n in ncol])
        r2w = ridge_r2(R, yw, train, test)
        lines += ["", "## Ridge h_l -> A_logit: test R^2 per layer",
                  fmt_row("R2", r2), "## ... within-N (per-N mean removed)",
                  fmt_row("R2w", r2w)]
        txt = "\n".join(lines) + "\n"
        print(txt, flush=True)
        (out / f"probe_{tag}.txt").write_text(txt)
        np.savez_compressed(out / f"probe_{tag}_curves.npz", A=A, top=top, r2=r2, n=ncol)


if __name__ == "__main__":
    main()
