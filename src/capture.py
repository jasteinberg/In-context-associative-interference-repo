"""Model runner with per-head QK capture for the in-context load experiment.

Captures, per layer: q at the answer position and k at every position,
both post-QK-norm and post-RoPE -- exactly the tensors whose inner
products the softmax sees (write-up, D3). Captured via forward
pre-hooks on each attention module, recomputing q/k with the module's
own projections; validate() checks the reconstructed answer-position
attention row against HF's eager `output_attentions` to ~bf16 tolerance.

Avoids materializing L x H x T x T attention: scores at the answer
position are rebuilt from the captured q, k. Logits are restricted to
the final position via `logits_to_keep=1`.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


class Capture:
    def __init__(self):
        self.q_last = []   # per layer: (Hq, d) float32 cpu
        self.k_all = []    # per layer: (Hkv, T, d) float32 cpu
        self.logits = None  # (vocab,) float32 cpu, answer position


class Runner:
    def __init__(self, model_name, device="mps", dtype=torch.bfloat16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, attn_implementation="eager"
        ).to(device).eval()
        self.device = device
        cfg = self.model.config
        lt = getattr(cfg, "layer_types", None)
        if lt is not None:
            assert all(t == "full_attention" for t in lt), lt
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv
        attn0 = self.model.model.layers[0].self_attn
        self.head_dim = attn0.head_dim
        self.scaling = attn0.scaling  # = head_dim ** -0.5, the beta in P1/P2

    def _hook(self, cap):
        def fn(module, args, kwargs):
            h = kwargs.get("hidden_states", args[0] if args else None)
            cos, sin = kwargs["position_embeddings"]
            shp = (*h.shape[:-1], -1, module.head_dim)
            q = module.q_norm(module.q_proj(h).view(shp)).transpose(1, 2)
            k = module.k_norm(module.k_proj(h).view(shp)).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            cap.q_last.append(q[0, :, -1, :].float().cpu())
            cap.k_all.append(k[0].float().cpu())
        return fn

    @torch.inference_mode()
    def forward(self, input_ids, want_resid=False):
        """One forward pass with the QK pre-hooks registered.

        want_resid=True additionally keeps the canonical residual
        stream at the answer position: cap.resid is
        (n_layers+1, d_model) float32, index 0 = embeddings and index
        l = the output of block l-1 (pre-layernorm), final position
        only. Default False leaves behaviour byte-identical.
        """
        cap = Capture()
        handles = [
            layer.self_attn.register_forward_pre_hook(
                self._hook(cap), with_kwargs=True
            )
            for layer in self.model.model.layers
        ]
        try:
            ids = torch.tensor([input_ids], device=self.device)
            out = self.model(ids, logits_to_keep=1,
                             output_hidden_states=want_resid)
        finally:
            for h in handles:
                h.remove()
        cap.logits = out.logits[0, -1].float().cpu()
        if want_resid:
            cap.resid = torch.stack(
                [h[0, -1, :].float().cpu() for h in out.hidden_states])
        return cap

    def answer_scores(self, cap):
        """(L, Hq, T) post-scaling scores at the answer position: the
        logits whose softmax is the answer-position attention row."""
        rows = []
        for l in range(self.n_layers):
            q = cap.q_last[l]                       # (Hq, d)
            k = cap.k_all[l]                        # (Hkv, T, d)
            k_rep = k.repeat_interleave(self.n_rep, dim=0)  # (Hq, T, d)
            s = torch.einsum("hd,htd->ht", q, k_rep) * self.scaling
            rows.append(s)
        return torch.stack(rows)                    # (L, Hq, T)

    @staticmethod
    def attention_rows(scores):
        """Softmax over positions (answer is last, so all are visible)."""
        return torch.softmax(scores, dim=-1)

    @torch.inference_mode()
    def validate(self, input_ids, atol=1e-4):
        """Check reconstructed answer-row attention against HF eager
        output_attentions. Returns max abs deviation over layers/heads.

        Exact (~2e-6) in float32. In bf16 on MPS the model's own matmul
        accumulates at reduced precision, giving ~1e-2 row deviations
        (measured 23 Aug, uniform across layers); reconstruction
        is the fp32-accumulated product of the same bf16 q, k, so for
        bf16 models pass atol~5e-2 and prefer fp32 where memory allows."""
        cap = self.forward(input_ids)
        rows = self.attention_rows(self.answer_scores(cap))
        ids = torch.tensor([input_ids], device=self.device)
        out = self.model(ids, output_attentions=True, logits_to_keep=1)
        worst = 0.0
        for l in range(self.n_layers):
            ref = out.attentions[l][0, :, -1, :].float().cpu()
            dev = (rows[l] - ref).abs().max().item()
            worst = max(worst, dev)
        assert worst < atol, f"attention reconstruction off by {worst}"
        return worst
