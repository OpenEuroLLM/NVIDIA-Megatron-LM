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

``nsa`` / ``csa2`` / ``lsa``: the trained variants of sparse_variants.py (NSA's compression + selection + window
branches with gates; CSA2's learned m:1 main-KV compression, lightning indexer on the compressed keys -- KL
warm-up like ``indexer`` -- and raw SWA; LSA's clustered key index). ``page_sparse_hier_*``: coarse-to-fine
selection for the page selectors. ``page_sparse_per_group``: one selection per KV group.
``page_sparse_share_source``: the attention layer of that layer number selects, the other attention layers reuse
its selection (one index shared across layers, V4.1-style): the source clears and fills a per-forward store, the
others read it -- valid while the source runs before them on the same micro-batch (forward, and full recompute
without pipeline parallelism).
"""

from __future__ import annotations

import torch

from megatron.core.transformer.experimental_attention_variant import page_sparse as ps
from megatron.core.transformer.experimental_attention_variant import sparse_variants as sv
from megatron.core.transformer.module import MegatronModule


# the shared selection of the current micro-batch: (document start, query offset) -> the source layer's selection
_SHARED_SELECTION: dict = {}


class PageSparseCoreAttention(MegatronModule):
    def __init__(self, config, layer_number, attn_mask_type=None, attention_type=None, cp_comm_type=None,
                 softmax_scale=None, pg_collection=None, **kwargs):
        super().__init__(config=config)
        self.layer_number = layer_number
        self.cfg = ps.PageSparseConfig(
            selector=config.page_sparse_selector, page=config.page_sparse_page, topk=config.page_sparse_topk,
            window=config.page_sparse_window, sink_pages=config.page_sparse_sink_pages,
            compress=config.page_sparse_compress, hier_group=config.page_sparse_hier_group,
            hier_topk=config.page_sparse_hier_topk, select_block=config.page_sparse_select_block,
            select_score=config.page_sparse_select_score, per_group=config.page_sparse_per_group,
        )
        self.share_source = config.page_sparse_share_source  # 0: every layer selects for itself
        cp = pg_collection.cp if pg_collection is not None else None
        self.cp_group = cp if cp is not None and cp.size() > 1 else None
        kv = config.kv_channels or config.hidden_size // config.num_attention_heads
        self.scale = softmax_scale if softmax_scale is not None else kv**-0.5
        self.indexer = self.variant = None
        if self.cfg.selector == "indexer":
            self.indexer = ps.PageIndexer(config.hidden_size, config.page_sparse_indexer_heads,
                                          config.page_sparse_indexer_dim, self.cfg.page)
        elif self.cfg.selector == "nsa":
            self.variant = sv.NSA(config.hidden_size, config.num_attention_heads, kv, sv.NSAConfig(
                block=config.page_sparse_nsa_block, stride=config.page_sparse_nsa_stride, page=self.cfg.page,
                topk=self.cfg.topk, sink_pages=self.cfg.sink_pages,
                select_window=config.page_sparse_nsa_select_window, window=self.cfg.window,
                select_block=config.page_sparse_select_block))
            self.variant.flex = config.page_sparse_backend == "flex"
        elif self.cfg.selector == "csa2":
            self.variant = sv.CSA2(config.hidden_size, config.num_query_groups or config.num_attention_heads, kv,
                                   sv.CSA2Config(compress=self.cfg.compress,
                                                 topk_entries=config.page_sparse_csa2_topk_entries,
                                                 window=self.cfg.window, page=self.cfg.page,
                                                 sink_pages=self.cfg.sink_pages,
                                                 index_heads=config.page_sparse_indexer_heads,
                                                 index_dim=config.page_sparse_indexer_dim,
                                                 select_block=config.page_sparse_select_block))
            self.variant.flex = config.page_sparse_backend == "flex"
        elif self.cfg.selector == "lsa":
            self.variant = sv.LSA(sv.LSAConfig(pool=config.page_sparse_lsa_pool, block=config.page_sparse_lsa_block,
                                               buckets=config.page_sparse_lsa_buckets, topk=self.cfg.topk,
                                               window=self.cfg.window, page=self.cfg.page,
                                               sink_pages=self.cfg.sink_pages,
                                               select_block=config.page_sparse_select_block))
            self.variant.flex = config.page_sparse_backend == "flex"
        self.flex = config.page_sparse_backend == "flex" and self.cfg.selector in ps.PAGE_SELECTORS
        self.q_block = config.page_sparse_flex_q_block
        self.loss_coeff = float(config.page_sparse_indexer_loss_coeff or 0.0)
        self.dense_forward = bool(config.page_sparse_dense_forward)
        self._hidden = None

    def set_hidden(self, hidden_states: torch.Tensor) -> None:
        """The attention's input ``[s, b, h]`` / ``[t, 1, h]`` (the indexer's input), set before each call."""
        self._hidden = hidden_states

    # ------------------------------------------------------------------ one document / sequence
    def _page_select(self, fn, key, *args, **kwargs):
        """A page selector's attention ``fn``, sharing the selection across layers when configured."""
        if not self.share_source:
            return fn(*args, **kwargs)
        if self.layer_number == self.share_source:
            out, sel = fn(*args, return_selection=True, **kwargs)
            _SHARED_SELECTION[key] = sel
            return out
        if key not in _SHARED_SELECTION:
            raise RuntimeError(f"page_sparse_share_source {self.share_source}: no selection for {key} -- the source "
                               "layer must run before this one on the same micro-batch")
        return fn(*args, selection=_SHARED_SELECTION[key], **kwargs)

    def _attend(self, q, k, v, q_offset, xq, kpages, kfeat=None, key=(0, 0)):
        """q [1, hq, sq, d] at positions q_offset.., k / v [1, g, L, d]; xq [1, sq, h] and kpages [1, P, dim]
        for the indexer; kfeat [1, L, F] the key side's per-token features (csa2: the compression weights).
        Returns (out [1, hq, sq, d], indexer loss or None)."""
        if self.cfg.selector == "dense":
            return ps.dense_attention(q, k, v, q_offset, self.scale), None
        if self.cfg.selector == "nsa":
            return self.variant(q, k, v, xq, q_offset, self.scale), None
        if self.cfg.selector == "lsa":
            return self.variant(q, k, v, q_offset, self.scale), None
        if self.cfg.selector == "csa2":
            loss = None
            if self.training and self.loss_coeff > 0:
                loss = self.variant.indexer_loss(q, k, xq, kfeat, q_offset, self.scale) * q.shape[2]
            if self.dense_forward:
                return ps.dense_attention(q, k, v, q_offset, self.scale), loss
            return self.variant(q, k, v, xq, kfeat, q_offset, self.scale), loss
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
            return self._page_select(ps.flex_page_sparse_attention, key, q, k, v, self.cfg, q_offset, self.scale,
                                     sel_scores, self.q_block), loss
        return self._page_select(ps.page_sparse_attention, key, q, k, v, self.cfg, q_offset, self.scale,
                                 sel_scores), loss

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None, attention_bias=None,
                packed_seq_params=None, **kwargs):
        thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        if self.share_source and self.layer_number == self.share_source:
            _SHARED_SELECTION.clear()  # a new micro-batch (or its recompute): the source refills
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

    def _needs_hidden(self) -> bool:
        return self.indexer is not None or self.cfg.selector in ("nsa", "csa2")

    def _forward_sbhd(self, query, key, value, packed_seq_params):
        if self.cp_group is not None:
            raise NotImplementedError("page_sparse_selector: unpacked batches under context parallelism")
        s, b, hq, d = query.shape
        outs, losses = [], []
        for i in range(b):
            q = query[:, i].permute(1, 0, 2).unsqueeze(0)
            k = key[:, i].permute(1, 0, 2).unsqueeze(0)
            v = value[:, i].permute(1, 0, 2).unsqueeze(0)
            xq = kp = kf = None
            if self.indexer is not None:
                x = self._hidden[:, i].unsqueeze(0)
                xq, kp = x, self.indexer.page_keys(x)
            elif self._needs_hidden():
                xq = self._hidden[:, i].unsqueeze(0)
                if self.cfg.selector == "csa2":
                    kf = self.variant.token_features(xq)
            o, loss = self._attend(q, k, v, 0, xq, kp, kf, key=(i, 0))
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
        x = self._hidden[:, 0] if self._needs_hidden() else None  # [t, h]
        # the key side's per-token features: the indexer's index keys, CSA2's compression weights
        feat = None
        if self.indexer is not None:
            feat = self.indexer.wk(x)
        elif self.cfg.selector == "csa2":
            feat = self.variant.token_features(x)
        if self.cp_group is not None:
            from megatron.core.ssm import fla_cp

            st = fla_cp.thd_stream(cu, self.cp_group)
            kfull, vfull = st.full(key), st.full(value)  # [total, g, d], in token order
            ffull = None if feat is None else st.full(feat)  # every token's key-side features
            cp, rank, bounds = st.cp, st.rank, st.cu_cpu.tolist()
        else:
            kfull, vfull, ffull = key, value, feat
            cp, rank, bounds = 1, 0, cu.to(torch.long).cpu().tolist()
        pieces, losses, local = [], [], 0
        for a, b in zip(bounds[:-1], bounds[1:]):
            L = b - a
            if L == 0:
                continue
            kd = kfull[a:b].permute(1, 0, 2).unsqueeze(0)  # [1, g, L, d]
            vd = vfull[a:b].permute(1, 0, 2).unsqueeze(0)
            kp = kf = None
            if self.indexer is not None:
                kp = ps.page_pool(ffull[a:b].unsqueeze(0).unsqueeze(0), self.cfg.page, "mean")[:, 0]
            elif ffull is not None:
                kf = ffull[a:b].unsqueeze(0)  # [1, L, F]
            c = L // (2 * cp)
            # this rank's query chunks of the document: (global offset, local start, length)
            spans = [(0, local, L)] if cp == 1 else [(rank * c, local, c), ((2 * cp - 1 - rank) * c, local + c, c)]
            for off, lo, n in spans:
                q = query[lo:lo + n].permute(1, 0, 2).unsqueeze(0)  # [1, hq, n, d]
                end = off + n  # the chunk's queries see the document's keys up to its end
                xq = None if x is None else x[lo:lo + n].unsqueeze(0)
                kpc = None if kp is None else kp[:, : -(-end // self.cfg.page)]
                kfc = None if kf is None else kf[:, :end]
                o, loss = self._attend(q, kd[:, :, :end], vd[:, :, :end], off, xq, kpc, kfc, key=(a, off))
                pieces.append(o[0].permute(1, 0, 2).reshape(n, hq * d))
                if loss is not None:
                    losses.append((loss, n))
            local += L // cp
        if local < t:  # padding after the last document
            pieces.append(query.new_zeros(t - local, hq * d))
        return torch.cat(pieces, 0), losses
