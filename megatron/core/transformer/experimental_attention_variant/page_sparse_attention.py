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

_BOUNDS: dict = {}


def _host_bounds(cu: torch.Tensor) -> list:
    """``cu`` (a micro-batch's document boundaries) as host ints, cached by tensor identity (fla_cp.thd_stream's
    scheme): every attention layer, its recompute and its backward share one micro-batch's cu, and the copy syncs the
    host (one sync per layer idled the GPU, t19d profile 2026-11-21). The cache holds the tensor, so its id is not
    reused while cached."""
    hit = _BOUNDS.get(id(cu))
    if hit is not None and hit[0] is cu:
        return hit[1]
    bounds = cu.to(torch.long).cpu().tolist()
    _BOUNDS[id(cu)] = (cu, bounds)
    while len(_BOUNDS) > 8:
        _BOUNDS.pop(next(iter(_BOUNDS)))
    return bounds

class IndexerLossLoggingHelper:
    """The selector's indexer KL (``page_sparse_indexer_loss_coeff``: the page indexer or CSA2's lightning
    indexer vs the dense attention mass), averaged over the attention layers, micro-batches and ranks of a step
    and logged as ``indexer_kl``. Recorded once per micro-batch and layer: with grad enabled (the training
    forward, or the recompute of a no-grad checkpointed one)."""

    tracker = {}

    @staticmethod
    def record(loss: torch.Tensor) -> None:
        if not torch.is_grad_enabled():
            return
        t = IndexerLossLoggingHelper.tracker
        if "value" not in t:
            t["value"] = torch.zeros(2, device=loss.device)
        t["value"][0] += loss.detach().float()
        t["value"][1] += 1

    @staticmethod
    def track_metrics(iteration: int, writer=None, wandb_writer=None, total_loss_dict=None,
                      name: str = "indexer_kl") -> None:
        """Reduce over all ranks (a collective: call on every rank when the coefficient is set), log, clear."""
        t = IndexerLossLoggingHelper.tracker
        if "value" not in t:
            t["value"] = torch.zeros(2, device=torch.cuda.current_device())
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(t["value"])
        value = t["value"][0] / t["value"][1].clamp(min=1)
        if total_loss_dict is not None:
            total_loss_dict[name] = total_loss_dict.get(name, 0.0) + value
        if writer is not None:
            writer.add_scalar(name, value, iteration)
        if wandb_writer is not None:
            wandb_writer.log({name: value}, iteration)
        t["value"].zero_()


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
            block_candidates=config.page_sparse_block_candidates, candidate_block=config.page_sparse_flex_q_block,
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
                                                 select_block=config.page_sparse_select_block,
                                                 entry_bias=config.page_sparse_csa2_entry_bias,
                                                 entry_rms=config.page_sparse_csa2_entry_rms))
            self.variant.flex = config.page_sparse_backend == "flex"
            self.variant.tag = f"L{layer_number} csa2"
        elif self.cfg.selector == "lsa":
            self.variant = sv.LSA(sv.LSAConfig(pool=config.page_sparse_lsa_pool, block=config.page_sparse_lsa_block,
                                               buckets=config.page_sparse_lsa_buckets, topk=self.cfg.topk,
                                               window=self.cfg.window, page=self.cfg.page,
                                               sink_pages=self.cfg.sink_pages,
                                               select_block=config.page_sparse_select_block,
                                               latent=config.page_sparse_lsa_latent,
                                               visibility=config.page_sparse_lsa_visibility,
                                               kl_every=config.page_sparse_lsa_kl_every),
                                  config.num_query_groups or config.num_attention_heads, config.kv_channels)
            self.variant.flex = config.page_sparse_backend == "flex"
        if config.page_sparse_selector_adam:  # the selector's parameters on Adam, not the matrix optimizer
            for mod in (self.indexer, self.variant):
                for param in (mod.parameters() if mod is not None else ()):
                    param.use_adam_with_emerging_optimizers = True
        # rowmask: the packed path's kernel with a bit word per (query tile, page) (rowmask_attention.py) -- per-token
        # choices inside block candidates without flex's per-element masks
        self.flex = config.page_sparse_backend in ("flex", "rowmask") and self.cfg.selector in ps.PAGE_SELECTORS
        self.packed_kernel = "rowmask" if config.page_sparse_backend == "rowmask" else "flex"
        self.token = config.page_sparse_backend == "token"  # per-token selection, NSA kernel (token_sparse.py)
        self.q_block = config.page_sparse_flex_q_block
        self.packed_flex = config.page_sparse_packed_flex  # one flex call for every document of a packed batch
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

    def _indexer_scores(self, q, k, q_offset, xq, kpages, drop: int = 0):
        """The indexer's page scores (detached: no gradient through the selection) and its KL warm-up loss
        (summed over the queries after the ``drop`` prepended ones), or (None, None) for the other page
        selectors."""
        if self.indexer is None:
            return None, None
        scores = self.indexer(xq, kpages)
        loss = None
        if self.training and self.loss_coeff > 0:
            oracle = ps.PageSparseConfig(selector="oracle", page=self.cfg.page, window=self.cfg.window,
                                         sink_pages=self.cfg.sink_pages)
            qo = q[:, :, drop:]
            with torch.no_grad():
                target = ps.page_scores(qo.detach(), k.detach(), oracle, q_offset + drop, self.scale)
            cand = ps.candidate_mask(qo.shape[2], scores.shape[-1], q_offset + drop, self.cfg, q.device)
            loss = ps.indexer_kl_loss(scores[:, drop:], target, cand) * qo.shape[2]
        return scores.detach(), loss

    def _packed_flex(self, spans, keys):
        """Every span (document or CP chunk) of the micro-batch through ONE flex call (page selectors), with
        the shared selection across layers: the source stores each span's, the others reuse them."""
        selections = None
        if self.share_source and self.layer_number != self.share_source:
            missing = [k for k in keys if k not in _SHARED_SELECTION]
            if missing:
                raise RuntimeError(f"page_sparse_share_source {self.share_source}: no selection for {missing[0]} "
                                   "-- the source layer must run before this one on the same micro-batch")
            selections = [_SHARED_SELECTION[k] for k in keys]
        source = bool(self.share_source) and self.layer_number == self.share_source
        if source:  # the selections are returned only for the layers that reuse them
            outs, sels = ps.flex_packed_page_attention(spans, self.cfg, self.scale, self.q_block, selections,
                                                       return_selection=True, kernel=self.packed_kernel)
        else:
            outs = ps.flex_packed_page_attention(spans, self.cfg, self.scale, self.q_block, selections,
                                                 kernel=self.packed_kernel)
        if source:
            for k, sel in zip(keys, sels):
                _SHARED_SELECTION[k] = sel
        return outs

    def _attend(self, q, k, v, q_offset, xq, kpages, kfeat=None, key=(0, 0), drop: int = 0):
        """q [1, hq, sq, d] at positions q_offset.., k / v [1, g, L, d]; xq [1, sq, h] and kpages [1, P, dim]
        for the indexer; kfeat [1, L, F] the key side's per-token features (csa2: the compression weights).
        The first ``drop`` queries are prepended block heads (CP, :meth:`_cp_block_heads`): they select, but the
        caller drops their outputs and the losses leave them out.
        Returns (out [1, hq, sq, d], indexer loss or None)."""
        if self.cfg.selector == "dense":
            return ps.dense_attention(q, k, v, q_offset, self.scale), None
        if self.cfg.selector == "nsa":
            return self.variant(q, k, v, xq, q_offset, self.scale), None
        qo, xo = q[:, :, drop:], (None if xq is None else xq[:, drop:])  # the loss rows
        if self.cfg.selector == "lsa":
            loss = None
            if self.training and self.loss_coeff > 0 and self.variant.proj_k is not None:  # D-LSA distillation
                loss = self.variant.indexer_loss(qo, k, q_offset + drop, self.scale) * qo.shape[2]
            if self.dense_forward:
                return ps.dense_attention(q, k, v, q_offset, self.scale), loss
            return self.variant(q, k, v, q_offset, self.scale), loss
        if self.cfg.selector == "csa2":
            loss = None
            if self.training and self.loss_coeff > 0:
                loss = self.variant.indexer_loss(qo, k, xo, kfeat, q_offset + drop, self.scale) * qo.shape[2]
            if self.dense_forward:
                return ps.dense_attention(q, k, v, q_offset, self.scale), loss
            return self.variant(q, k, v, xq, kfeat, q_offset, self.scale), loss
        sel_scores, loss = self._indexer_scores(q, k, q_offset, xq, kpages, drop)
        if self.dense_forward:
            return ps.dense_attention(q, k, v, q_offset, self.scale), loss
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
        if sv.NAN_DEBUG:
            for name, t in (("query", query), ("key", key), ("value", value), ("hidden", self._hidden)):
                sv.nan_check(f"L{self.layer_number}", name, t)
        out, losses = (self._forward_thd if thd else self._forward_sbhd)(query, key, value, packed_seq_params)
        if sv.NAN_DEBUG:
            sv.nan_check(f"L{self.layer_number}", "attention out", out)
            for i, (lv, _) in enumerate(losses):
                sv.nan_check(f"L{self.layer_number}", f"indexer loss {i}", lv)
        if losses:
            from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

            n = sum(x[1] for x in losses)
            loss = sum(x[0] for x in losses) / max(n, 1)
            IndexerLossLoggingHelper.record(loss)
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

    def _cp_block_heads(self, query, x, bounds, cp, rank):
        """Under CP, the queries a chunk needs from other ranks. A select block (``select_block`` B queries share
        one selection, made by the block's first query) that straddles a chunk boundary has its first query on
        another rank; the chunk then gets the block's earlier queries prepended, so it starts at a multiple of B
        and every selector makes the one-rank selection. Every rank builds the same request list (the zigzag
        layout is known everywhere), fills the rows it holds, and one all-reduce shares them.
        -> {(document start, chunk offset): (q rows [h, hq, d], x rows [h, H] or None)}, detached: their outputs
        are dropped, they only select (and selection carries no gradient)."""
        B = self.cfg.select_block
        if cp == 1 or B <= 1 or self.cfg.selector == "dense":
            return {}
        reqs, buf_rows, loc_rows, n_rows, loc0 = [], [], [], 0, 0
        for a, b in zip(bounds[:-1], bounds[1:]):
            L = b - a
            if L == 0:
                continue
            c = L // (2 * cp)
            for r in range(cp):
                for off in (r * c, (2 * cp - 1 - r) * c):
                    h = off % B
                    if h == 0:
                        continue
                    if r == rank:
                        reqs.append((a, off, n_rows, h))
                    for i, p in enumerate(range(off - h, off)):  # the block's queries before the chunk
                        j = p // c
                        if (j if j < cp else 2 * cp - 1 - j) == rank:
                            buf_rows.append(n_rows + i)
                            loc_rows.append(loc0 + (p - j * c) + (0 if j < cp else c))
                    n_rows += h
            loc0 += L // cp
        if n_rows == 0:
            return {}
        bi = torch.tensor(buf_rows, dtype=torch.long, device=query.device)
        li = torch.tensor(loc_rows, dtype=torch.long, device=query.device)
        qb = query.new_zeros(n_rows, *query.shape[1:])
        qb[bi] = query.detach()[li]
        torch.distributed.all_reduce(qb, group=self.cp_group)  # each row is non-zero on its owner only
        xb = None
        if x is not None:
            xb = x.new_zeros(n_rows, x.shape[-1])
            xb[bi] = x.detach()[li]
            torch.distributed.all_reduce(xb, group=self.cp_group)
        return {(a, off): (qb[s0:s0 + h], None if xb is None else xb[s0:s0 + h]) for a, off, s0, h in reqs}

    def _forward_thd(self, query, key, value, packed_seq_params):
        """query [t, hq, d], key / value [t, g, d] (this rank's THD tokens) -> [t, hq * d]."""
        cu = packed_seq_params.cu_seqlens_q_padded
        if cu is None:
            cu = packed_seq_params.cu_seqlens_q
        t, hq, d = query.shape
        if self.token:  # every query token selects its own pages: one NSA-kernel call for all documents
            if self.cp_group is not None:
                raise NotImplementedError("page_sparse_backend token: context parallelism")
            from megatron.core.transformer.experimental_attention_variant import token_sparse

            out = token_sparse.token_page_attention_thd(query, key, value, cu, self.cfg, self.scale)
            return out.reshape(t, hq * out.shape[-1]), []
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
            cp, rank, bounds = 1, 0, _host_bounds(cu)
        # page selectors under flex: every document's spans in ONE flex call (per-document calls cost a kernel
        # launch per document and layer, forward and backward)
        packed = (self.flex and self.packed_flex and query.is_cuda and self.variant is None
                  and not self.dense_forward and self.cfg.selector in ps.PAGE_SELECTORS)
        flex_spans, flex_keys, flex_rows = [], [], []
        pieces, losses, local = [], [], 0
        heads = self._cp_block_heads(query, x, bounds, cp, rank)
        # every document's keys and this rank's query chunks as ONE split each (in order): a slice per document
        # backpropagates a zero tensor of the whole input per document (O(documents x tokens), 2026-11-20), split's
        # backward is one cat
        docs, q_sizes, at = [], [], 0
        for a, b in zip(bounds[:-1], bounds[1:]):
            L = b - a
            c = L // (2 * cp)
            # this rank's query chunks of the document: (global offset, local start, length)
            spans = [(0, at, L)] if cp == 1 else [(rank * c, at, c), ((2 * cp - 1 - rank) * c, at + c, c)]
            docs.append((a, b, spans))
            q_sizes += [n for _, _, n in spans]
            at += L // cp
        q_sizes.append(t - at)  # padding after the last document
        k_sizes = [b - a for a, b in zip(bounds[:-1], bounds[1:])] + [kfull.shape[0] - bounds[-1]]
        q_parts = iter(query.split(q_sizes, 0))
        x_parts = iter(x.split(q_sizes, 0)) if x is not None else None
        k_parts, v_parts = kfull.split(k_sizes, 0), vfull.split(k_sizes, 0)
        f_parts = ffull.split(k_sizes, 0) if ffull is not None else None
        for di, (a, b, spans) in enumerate(docs):
            L = b - a
            if L == 0:
                for _ in spans:
                    next(q_parts)
                    x_parts is not None and next(x_parts)
                continue
            kd = k_parts[di].permute(1, 0, 2).unsqueeze(0)  # [1, g, L, d]
            vd = v_parts[di].permute(1, 0, 2).unsqueeze(0)
            kp = kf = None
            if self.indexer is not None:
                kp = ps.page_pool(f_parts[di].unsqueeze(0).unsqueeze(0), self.cfg.page, "mean")[:, 0]
            elif f_parts is not None:
                kf = f_parts[di].unsqueeze(0)  # [1, L, F]
            for off, lo, n in spans:
                q = next(q_parts).permute(1, 0, 2).unsqueeze(0)  # [1, hq, n, d]
                end = off + n  # the chunk's queries see the document's keys up to its end
                xq = None if x_parts is None else next(x_parts).unsqueeze(0)
                h = 0  # prepended block heads (a chunk starting inside a select block, CP)
                if (a, off) in heads:
                    qh, xh = heads[(a, off)]
                    h = qh.shape[0]
                    q = torch.cat([qh.permute(1, 0, 2).unsqueeze(0), q], 2)
                    xq = None if xq is None else torch.cat([xh.unsqueeze(0), xq], 1)
                kpc = None if kp is None else kp[:, : -(-end // self.cfg.page)]
                kfc = None if kf is None else kf[:, :end]
                if packed:
                    sel_scores, loss = self._indexer_scores(q, kd[:, :, :end], off - h, xq, kpc, h)
                    flex_spans.append((q, kd[:, :, :end], vd[:, :, :end], off - h, sel_scores))
                    flex_keys.append((a, off))
                    flex_rows.append((len(pieces), n, h))
                    pieces.append(None)  # filled after the one flex call
                else:
                    o, loss = self._attend(q, kd[:, :, :end], vd[:, :, :end], off - h, xq, kpc, kfc, key=(a, off),
                                           drop=h)
                    pieces.append(o[0, :, h:].permute(1, 0, 2))  # [n, hq, d] view: one copy, in the cat
                if loss is not None:
                    losses.append((loss, n))
            local += L // cp
        if flex_spans:
            for (i, n, h), o in zip(flex_rows, self._packed_flex(flex_spans, flex_keys)):
                pieces[i] = o[0, :, h:].permute(1, 0, 2)
        if local < t:  # padding after the last document
            pieces.append(query.new_zeros(t - local, hq, d))
        # the permuted [n, hq, d] views concatenated once (a reshape per piece copied every output twice: +0.5 GB per
        # layer at v4b's 64k, 2026-11-22)
        return torch.cat(pieces, 0).view(t, hq * d), losses
