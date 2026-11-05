# Copyright (c) 2026, OpenEuroLLM. All rights reserved.
"""Megatron core attention with page-sparse attention (page_sparse.py), for ``SelfAttention`` when
``page_sparse_selector`` is set: the drop-in replacement of the TE core attention module (same call, same
output layout), for packed batches (THD, document by document) with or without context parallelism, and
unpacked ``[s, b, h, d]`` batches without it.

Context parallelism: our GQA keys / values are small (2 heads x 128), so every rank all-gathers them (the
all-gather's backward is a reduce-scatter: each key gets the gradients of every rank's queries) and attends its
own queries -- Megatron's chunks r and 2 cp - 1 - r of every document -- against the document's full keys.

``indexer`` selector: a learned :class:`page_sparse.PageIndexer` on the attention input (``set_hidden``),
trained by KL to the dense attention's page mass (``page_sparse_indexer_loss_coeff``; attached to the output
like the MoE aux losses); ``page_sparse_dense_forward`` keeps the LM on dense attention meanwhile (the warm-up).
"""

from __future__ import annotations

import torch

from megatron.core.transformer.experimental_attention_variant import page_sparse as ps
from megatron.core.transformer.module import MegatronModule


class PageSparseCoreAttention(MegatronModule):
    def __init__(self, config, layer_number, attn_mask_type=None, attention_type=None, cp_comm_type=None,
                 softmax_scale=None, pg_collection=None, **kwargs):
        super().__init__(config=config)
        self.layer_number = layer_number
        self.cfg = ps.PageSparseConfig(
            selector=config.page_sparse_selector, page=config.page_sparse_page, topk=config.page_sparse_topk,
            window=config.page_sparse_window, sink_pages=config.page_sparse_sink_pages,
            compress=config.page_sparse_compress,
        )
        cp = pg_collection.cp if pg_collection is not None else None
        self.cp_group = cp if cp is not None and cp.size() > 1 else None
        kv = config.kv_channels or config.hidden_size // config.num_attention_heads
        self.scale = softmax_scale if softmax_scale is not None else kv**-0.5
        self.indexer = None
        if self.cfg.selector == "indexer":
            self.indexer = ps.PageIndexer(config.hidden_size, config.page_sparse_indexer_heads,
                                          config.page_sparse_indexer_dim, self.cfg.page)
        self.flex = config.page_sparse_backend == "flex" and self.cfg.selector in ps.PAGE_SELECTORS
        self.q_block = config.page_sparse_flex_q_block
        self.loss_coeff = float(config.page_sparse_indexer_loss_coeff or 0.0)
        self.dense_forward = bool(config.page_sparse_dense_forward)
        self._hidden = None

    def set_hidden(self, hidden_states: torch.Tensor) -> None:
        """The attention's input ``[s, b, h]`` / ``[t, 1, h]`` (the indexer's input), set before each call."""
        self._hidden = hidden_states

    # ------------------------------------------------------------------ one document / sequence
    def _attend(self, q, k, v, q_offset, xq, kpages):
        """q [1, hq, sq, d] at positions q_offset.., k / v [1, g, L, d]; xq [1, sq, h] and kpages [1, P, dim]
        for the indexer. Returns (out [1, hq, sq, d], indexer loss or None)."""
        if self.cfg.selector == "dense":
            return ps.dense_attention(q, k, v, q_offset, self.scale), None
        scores = loss = None
        if self.indexer is not None:
            scores = self.indexer(xq, kpages)
            if self.training and self.loss_coeff > 0:
                oracle = ps.PageSparseConfig(selector="oracle", page=self.cfg.page, window=self.cfg.window,
                                             sink_pages=self.cfg.sink_pages)
                with torch.no_grad():
                    target = ps.page_scores(q.detach(), k.detach(), oracle, q_offset, self.scale)
                cand = ps.candidate_mask(q.shape[2], scores.shape[-1], q_offset, self.cfg, q.device)
                loss = ps.indexer_kl_loss(scores, target, cand) * q.shape[2]  # summed over the queries
        if self.dense_forward:
            return ps.dense_attention(q, k, v, q_offset, self.scale), loss
        sel_scores = None if scores is None else scores.detach()  # no gradient through the selection
        if self.flex and q.is_cuda:
            return ps.flex_page_sparse_attention(q, k, v, self.cfg, q_offset, self.scale, sel_scores,
                                                 self.q_block), loss
        return ps.page_sparse_attention(q, k, v, self.cfg, q_offset, self.scale, sel_scores), loss

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None, attention_bias=None,
                packed_seq_params=None, **kwargs):
        thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        if attention_bias is not None:
            raise NotImplementedError("page_sparse_selector: attention_bias")
        out, losses = (self._forward_thd if thd else self._forward_sbhd)(query, key, value, packed_seq_params)
        if losses:
            from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

            n = sum(x[1] for x in losses)
            loss = sum(x[0] for x in losses) / max(n, 1)
            out = MoEAuxLossAutoScaler.apply(out, loss * self.loss_coeff)
        self._hidden = None
        return out

    def _forward_sbhd(self, query, key, value, packed_seq_params):
        if self.cp_group is not None:
            raise NotImplementedError("page_sparse_selector: unpacked batches under context parallelism")
        s, b, hq, d = query.shape
        outs, losses = [], []
        for i in range(b):
            q = query[:, i].permute(1, 0, 2).unsqueeze(0)
            k = key[:, i].permute(1, 0, 2).unsqueeze(0)
            v = value[:, i].permute(1, 0, 2).unsqueeze(0)
            xq = kp = None
            if self.indexer is not None:
                x = self._hidden[:, i].unsqueeze(0)
                xq, kp = x, self.indexer.page_keys(x)
            o, loss = self._attend(q, k, v, 0, xq, kp)
            outs.append(o[0].permute(1, 0, 2).reshape(s, hq * d))
            if loss is not None:
                losses.append((loss, s))
        return torch.stack(outs, 1), losses

    def _forward_thd(self, query, key, value, packed_seq_params):
        """query [t, hq, d], key / value [t, g, d] (this rank's THD tokens) -> [t, hq * d]."""
        cu = packed_seq_params.cu_seqlens_q_padded
        if cu is None:
            cu = packed_seq_params.cu_seqlens_q
        t, hq, d = query.shape
        x = None if self.indexer is None else self._hidden[:, 0]  # [t, h]
        if self.cp_group is not None:
            from megatron.core.ssm import fla_cp

            st = fla_cp.thd_stream(cu, self.cp_group)
            kfull, vfull = st.full(key), st.full(value)  # [total, g, d], in token order
            kidx = None if x is None else st.full(self.indexer.wk(x))  # the index keys of every token
            cp, rank, bounds = st.cp, st.rank, st.cu_cpu.tolist()
        else:
            kfull, vfull = key, value
            kidx = None if x is None else self.indexer.wk(x)
            cp, rank, bounds = 1, 0, cu.to(torch.long).cpu().tolist()
        pieces, losses, local = [], [], 0
        for a, b in zip(bounds[:-1], bounds[1:]):
            L = b - a
            if L == 0:
                continue
            kd = kfull[a:b].permute(1, 0, 2).unsqueeze(0)  # [1, g, L, d]
            vd = vfull[a:b].permute(1, 0, 2).unsqueeze(0)
            kp = None
            if kidx is not None:
                kp = ps.page_pool(kidx[a:b].unsqueeze(0).unsqueeze(0), self.cfg.page, "mean")[:, 0]
            c = L // (2 * cp)
            # this rank's query chunks of the document: (global offset, local start, length)
            spans = [(0, local, L)] if cp == 1 else [(rank * c, local, c), ((2 * cp - 1 - rank) * c, local + c, c)]
            for off, lo, n in spans:
                q = query[lo:lo + n].permute(1, 0, 2).unsqueeze(0)  # [1, hq, n, d]
                end = off + n  # the chunk's queries see the document's keys up to its end
                xq = None if x is None else x[lo:lo + n].unsqueeze(0)
                kpc = None if kp is None else kp[:, : -(-end // self.cfg.page)]
                o, loss = self._attend(q, kd[:, :, :end], vd[:, :, :end], off, xq, kpc)
                pieces.append(o[0].permute(1, 0, 2).reshape(n, hq * d))
                if loss is not None:
                    losses.append((loss, n))
            local += L // cp
        if local < t:  # padding after the last document
            pieces.append(query.new_zeros(t - local, hq * d))
        return torch.cat(pieces, 0), losses
