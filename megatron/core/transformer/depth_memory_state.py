# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""The Linear Depth Memory state as a static workspace with an unwinding backward
(arXiv:2610.18809, Section 3.3 / Appendix F; ``depth_memory_impl: torch``).

The state ``(M, P)`` of one micro-batch is not an autograd tensor: it lives in persistent
buffers that every step updates in place, and the backward runs the steps in reverse,
restoring ``M_{e-1} = M_e - kappa_e u_e^T`` from a small per-step tape (the paper's state
unwinding; no window deferral here, E = 1). Every step (edge) ``e`` is a write followed by the
next read::

    kappa = P kbar / (lambda + kbar' P kbar),  u = vbar - M' kbar,
    M <- M + kappa u',  P <- (P - kappa kbar' P) / lambda,  read = M' q

with ``kbar = s * khat``, ``vbar = s * v`` (``s = sqrt(write weight)``). Edge 0 writes the
embedding; edge ``e`` the output of the ``e``-th sublayer; the last edge's read feeds the LM head.

Why static buffers: CUDA graphs bake every address they touch, and the per-layer graphs replay
without Python, so the state must be the same memory on every micro-batch -- the buffers are
allocated once per token count (outside any capture: the first forward is eager) and kept.
The backward order is fixed (the last edge first), so the window bookkeeping is a function of
the edge alone. One workspace serves one model chunk without pipeline parallelism (a micro-batch's
forward and backward complete before the next one starts).

The same interface (``forward(edge, ...)``, ``backward(edge, read_grad)``) is implemented by the
CuTe kernels of the paper (cute_kernels ``cute_ldm``), which this module's math is the oracle of.
"""

import os

import torch


class TorchDepthMemoryWorkspace:
    """The LDM state of ``total_edges`` steps for ``n`` tokens, FP32, in PyTorch."""

    def __init__(self, total_edges: int, groups: int, key_dim: int, value_dim: int, alpha: float):
        self.total_edges, self.groups, self.key_dim, self.value_dim = (
            total_edges, groups, key_dim, value_dim
        )
        self.alpha = float(alpha)
        self.n = None

    def ensure(self, n: int, device) -> None:
        """Allocate the buffers for ``n`` tokens (no-op when they exist)."""
        if self.n == n and self.m.device == device:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("depth memory: workspace allocation inside a CUDA graph capture")
        L, g, k, d = self.total_edges, self.groups, self.key_dim, self.value_dim
        f32 = dict(dtype=torch.float32, device=device)
        self.m = torch.zeros(n, g, k, d, **f32)
        self.p = torch.zeros(n, g, k, k, **f32)
        self.p0 = (torch.eye(k, **f32) * self.alpha).expand(n, g, k, k).contiguous()
        self.dm = torch.zeros(n, g, k, d, **f32)
        self.dp = torch.zeros(n, g, k, k, **f32)
        # per-edge tape: the pre-write P, the gain, the pseudo-value and the addresses
        self.t_p = torch.zeros(L, n, g, k, k, **f32)
        self.t_kappa = torch.zeros(L, n, g, k, **f32)
        self.t_u = torch.zeros(L, n, g, d, **f32)
        self.t_kbar = torch.zeros(L, n, g, k, **f32)
        self.t_khat = torch.zeros(L, n, g, k, **f32)
        self.t_q = torch.zeros(L, n, g, k, **f32)
        self.t_s = torch.zeros(L, n, g, **f32)
        self.t_lam = torch.zeros(L, g, **f32)
        self.n = n
        self.state_loaded = False  # see load_state / load_grad
        self.grad_loaded = False

    @torch.no_grad()
    def forward(self, edge, khat, query, value, scale, forgetting) -> torch.Tensor:
        """Step ``edge``: write (``khat`` [N,G,K], ``value`` [N,G,D], ``scale`` [N,G],
        ``forgetting`` [G]), then read ``query`` [N,G,K]; returns the read [N,G,D] FP32."""
        if edge == 0 and not self.state_loaded:
            self.m.zero_()
            self.p.copy_(self.p0)
        m, p = self.m, self.p
        khat, query, scale = khat.float(), query.float(), scale.float()
        lam = forgetting.float()
        kbar = khat * scale.unsqueeze(-1)
        vbar = value.float() * scale.unsqueeze(-1)
        pk = torch.einsum("ngij,ngj->ngi", p, kbar)
        kappa = pk / (lam + (kbar * pk).sum(-1)).unsqueeze(-1)
        u = vbar - torch.einsum("ngkd,ngk->ngd", m, kbar)
        self.t_p[edge].copy_(p)
        self.t_kappa[edge].copy_(kappa)
        self.t_u[edge].copy_(u)
        self.t_kbar[edge].copy_(kbar)
        self.t_khat[edge].copy_(khat)
        self.t_q[edge].copy_(query)
        self.t_s[edge].copy_(scale)
        self.t_lam[edge].copy_(lam)
        m.add_(kappa.unsqueeze(-1) * u.unsqueeze(-2))
        kp = torch.einsum("ngi,ngij->ngj", kbar, p)
        p.sub_(kappa.unsqueeze(-1) * kp.unsqueeze(-2)).div_(lam[:, None, None])
        return torch.einsum("ngkd,ngk->ngd", m, query)

    def backward(self, edge, read_grad, slot=None):
        """The reverse of step ``edge`` (called in strictly decreasing edge order): the
        gradients of (khat, query, value, scale, forgetting [G]), and ``M`` unwound to
        ``M_{edge-1}``. ``slot``: the tape entry holding the edge (default: the edge)."""
        if edge == self.total_edges - 1 and not self.grad_loaded:
            self.dm.zero_()
            self.dp.zero_()
        edge = edge if slot is None else slot  # the tape index from here on
        m = self.m  # holds M_edge here
        with torch.no_grad():
            dr = read_grad.float()
            q = self.t_q[edge]
            dq = torch.einsum("ngkd,ngd->ngk", m, dr)
            dm_total = self.dm + q.unsqueeze(-1) * dr.unsqueeze(-2)  # dL/dM_edge
            kappa, u, kbar = self.t_kappa[edge], self.t_u[edge], self.t_kbar[edge]
            du = torch.einsum("ngkd,ngk->ngd", dm_total, kappa)
            dkappa_m = torch.einsum("ngkd,ngd->ngk", dm_total, u)
            m.sub_(kappa.unsqueeze(-1) * u.unsqueeze(-2))  # unwind: M_{edge-1}
            dkbar_m = -torch.einsum("ngkd,ngd->ngk", m, du)
            vbar = u + torch.einsum("ngkd,ngk->ngd", m, kbar)
            self.dm.copy_(dm_total).sub_(kbar.unsqueeze(-1) * du.unsqueeze(-2))  # dL/dM_{edge-1}
        # the gain and the covariance update: a small local graph on [N, G, K, K]
        with torch.enable_grad():
            p_prev = self.t_p[edge].detach().requires_grad_(True)
            kb = kbar.detach().requires_grad_(True)
            lam = self.t_lam[edge].detach().requires_grad_(True)
            pk = torch.einsum("ngij,ngj->ngi", p_prev, kb)
            kap = pk / (lam + (kb * pk).sum(-1)).unsqueeze(-1)
            kp = torch.einsum("ngi,ngij->ngj", kb, p_prev)
            p_new = (p_prev - kap.unsqueeze(-1) * kp.unsqueeze(-2)) / lam[:, None, None]
            d_p_prev, d_kb, d_lam = torch.autograd.grad(
                (kap, p_new), (p_prev, kb, lam), (dkappa_m, self.dp.clone())
            )
        with torch.no_grad():
            self.dp.copy_(d_p_prev)
            dkbar = dkbar_m + d_kb
            s, khat = self.t_s[edge], self.t_khat[edge]
            value = vbar / s.unsqueeze(-1)
            dkhat = dkbar * s.unsqueeze(-1)
            dvalue = du * s.unsqueeze(-1)
            dscale = (dkbar * khat).sum(-1) + (du * value).sum(-1)
        return dkhat, dq, dvalue, dscale, d_lam

    # -- segments (a state carried across recompute chunks / pipeline stages; see the CuTe
    # workspace for the window fold this E = 1 oracle does not need) ----------------------------
    def load_state(self, m: torch.Tensor, p: torch.Tensor) -> None:
        self.m.copy_(m.view_as(self.m))
        self.p.copy_(p.view_as(self.p))
        self.state_loaded = True

    def export_state(self):
        return self.m, self.p

    def load_grad(self, dm: torch.Tensor, dp: torch.Tensor) -> None:
        self.dm.copy_(dm.view_as(self.dm))
        self.dp.copy_(dp.view_as(self.dp))
        self.grad_loaded = True

    def export_grad(self):
        return self.dm, self.dp

    def grad_buffers(self):
        """The buffers the backward starts from (load_grad's targets)."""
        return self.dm, self.dp

    def state_layout(self):
        """(per-token shape, dtype) of the carried state (M, P) and of its gradient (dM, dP)."""
        g, k, d = self.groups, self.key_dim, self.value_dim
        both = [((g, k, d), torch.float32), ((g, k, k), torch.float32)]
        return both, both


class UnwindingDepthMemoryWorkspace(TorchDepthMemoryWorkspace):
    """``depth_memory_unwind``: the state without a tape across edges (see the config field).

    Phases per micro-batch: the forward runs edges 0 .. L-1 in order and updates (M, P) in place
    (only edge 0, the embedding write outside the recompute chunks, keeps its tape). After the last
    edge the workspace is in the backward phase: a ``forward(edge)`` call then is the recompute of
    that edge's layer, run right before its backward, with the state at ``(M_e, P_e)``; it returns
    the same read ``M_e' q`` and recovers the edge's tape from the post-update state:

        kappa = P_e kbar,  c = 1 - kappa . kbar,  u = (vbar - M_e' kbar) / c,
        P_{e-1} = lambda (P_e + kappa kappa' / c)

    (RLS: P_e kbar is the gain; Sherman-Morrison for P). ``backward(edge)`` then runs the
    oracle's backward on that one-edge tape and unwinds M and P to edge e - 1.
    """

    def ensure(self, n: int, device) -> None:
        if self.n == n and self.m.device == device:
            return
        total = self.total_edges
        self.total_edges = 2  # tape slots: 0 the embedding write, 1 the edge being unwound
        try:
            super().ensure(n, device)
        finally:
            self.total_edges = total
        self._phase, self._next = "forward", 0
        g, d = self.groups, self.value_dim
        self.t_vbar = torch.zeros(n, g, d, dtype=torch.float32, device=device)
        self.t_c = torch.zeros(n, g, dtype=torch.float32, device=device)

    @torch.no_grad()
    def forward(self, edge, khat, query, value, scale, forgetting) -> torch.Tensor:
        if self._fused:
            return self._fused_forward(edge, khat, query, value, scale, forgetting)
        if self._phase == "forward" or edge == 0:
            if edge != self._next:
                raise RuntimeError(
                    f"depth_memory_unwind: forward edge {edge}, expected {self._next} (every "
                    "layer must be in a recompute chunk of its own)"
                )
            if edge == 0:
                self.m.zero_()
                self.p.copy_(self.p0)
                self._phase = "forward"
                read = TorchDepthMemoryWorkspace.forward(self, 0, khat, query, value, scale,
                                                         forgetting)
            else:
                read = self._step(khat, query, value, scale, forgetting)
            self._next = edge + 1
            if self._next == self.total_edges:
                self._phase, self._unwind = "backward", self.total_edges - 1
            return read
        if edge != self._unwind:
            raise RuntimeError(
                f"depth_memory_unwind: recompute of edge {edge}, the state is at {self._unwind}"
            )
        return self._recover(khat, query, value, scale, forgetting)

    @property
    def _fused(self) -> bool:
        """The fused Triton passes over M (depth_memory_triton) on CUDA."""
        return self.m.is_cuda and os.environ.get("DEPTH_MEMORY_UNWIND_TRITON", "1") != "0"

    def _step(self, khat, query, value, scale, forgetting):
        """The forward write + read without a tape."""
        m, p = self.m, self.p
        lam = forgetting.float()
        kbar = khat.float() * scale.float().unsqueeze(-1)
        vbar = value.float() * scale.float().unsqueeze(-1)
        if self._fused:
            from megatron.core.transformer.depth_memory_triton import forward_step, p_forward

            kappa = torch.empty_like(kbar)
            p_forward(p, kbar, lam, kappa)  # kappa, and P updated in place
            return forward_step(m, kbar, kappa, vbar, query.float())
        pk = torch.einsum("ngij,ngj->ngi", p, kbar)
        kappa = pk / (lam + (kbar * pk).sum(-1)).unsqueeze(-1)
        u = vbar - torch.einsum("ngkd,ngk->ngd", m, kbar)
        m.add_(kappa.unsqueeze(-1) * u.unsqueeze(-2))
        kp = torch.einsum("ngi,ngij->ngj", kbar, p)
        p.sub_(kappa.unsqueeze(-1) * kp.unsqueeze(-2)).div_(lam[:, None, None])
        return torch.einsum("ngkd,ngk->ngd", m, query.float())

    def _recover(self, khat, query, value, scale, forgetting):
        """The recompute of the edge the state is at: its read and its tape (slot 1)."""
        m, p = self.m, self.p
        khat, query, scale = khat.float(), query.float(), scale.float()
        lam = forgetting.float()
        kbar = khat * scale.unsqueeze(-1)
        vbar = value.float() * scale.unsqueeze(-1)
        if self._fused:
            from megatron.core.transformer.depth_memory_triton import p_recover

            self.t_kbar[1].copy_(kbar)
            self.t_khat[1].copy_(khat)
            self.t_q[1].copy_(query)
            self.t_s[1].copy_(scale)
            self.t_lam[1].copy_(lam)
            self.t_vbar.copy_(vbar)
            p_recover(p, self.t_kbar[1], self.t_lam[1], self.t_kappa[1], self.t_c, self.t_p[1])
            if os.environ.get("DEPTH_MEMORY_UNWIND_RECOMPUTE_READ", "0") == "1":
                return torch.einsum("ngkd,ngk->ngd", m, query)
            # the recomputed layer's output is not consumed by its backward (only the graph):
            # no pass over M here, u is recovered inside the backward's pass
            return torch.empty(m.shape[0], m.shape[1], m.shape[3], device=m.device)
        kappa = torch.einsum("ngij,ngj->ngi", p, kbar)
        c = (1.0 - (kappa * kbar).sum(-1)).unsqueeze(-1)
        u = (vbar - torch.einsum("ngkd,ngk->ngd", m, kbar)) / c
        self.t_p[1].copy_(p + kappa.unsqueeze(-1) * (kappa / c).unsqueeze(-2)).mul_(
            lam[:, None, None]
        )
        self.t_kappa[1].copy_(kappa)
        self.t_u[1].copy_(u)
        self.t_kbar[1].copy_(kbar)
        self.t_khat[1].copy_(khat)
        self.t_q[1].copy_(query)
        self.t_s[1].copy_(scale)
        self.t_lam[1].copy_(lam)
        return torch.einsum("ngkd,ngk->ngd", m, query)

    def backward(self, edge, read_grad):
        if self._fused:
            return self._fused_backward2(edge, read_grad)
        if edge == 0:
            out = super().backward(0, read_grad, slot=0)
            self.p.copy_(self.t_p[0])  # the initial P (M is unwound to zero)
            self._phase, self._next = "forward", 0
            return out
        if self._phase != "backward" or edge != self._unwind:
            raise RuntimeError(f"depth_memory_unwind: backward of edge {edge} out of order")
        out = (self._fused_backward if self._fused else super().backward)(edge, read_grad, slot=1)
        self.p.copy_(self.t_p[1])  # P_{e-1}; M was unwound by the backward
        self._unwind = edge - 1
        return out


    def _fused_backward(self, edge, read_grad, slot):
        """The backward of the edge M is at (slot 1, recovered by _recover): one pass over M / dM
        (u recovered there, M unwound), the covariance's vjp fused (depth_memory_triton)."""
        from megatron.core.transformer.depth_memory_triton import p_backward, unwind_backward_step

        if edge == self.total_edges - 1 and not self.grad_loaded:
            self.dm.zero_()
            self.dp.zero_()
        kappa, kbar, lam = self.t_kappa[slot], self.t_kbar[slot], self.t_lam[slot]
        with torch.no_grad():
            dq, du, dkappa_m, dkbar_m = unwind_backward_step(
                self.m, self.dm, self.t_q[slot], read_grad.float(), kappa, kbar, self.t_vbar,
                self.t_c,
            )
            d_kb, d_lam = p_backward(self.t_p[slot], kbar, lam, dkappa_m, self.dp)
            dkbar = dkbar_m + d_kb
            s, khat = self.t_s[slot], self.t_khat[slot]
            value = self.t_vbar / s.unsqueeze(-1)
            dkhat = dkbar * s.unsqueeze(-1)
            dvalue = du * s.unsqueeze(-1)
            dscale = (dkbar * khat).sum(-1) + (du * value).sum(-1)
        return dkhat, dq, dvalue, dscale, d_lam


    # -- the fused path (depth_memory_triton v2): every edge, edge 0 included ---------------
    def _fused_forward(self, edge, khat, query, value, scale, forgetting):
        from megatron.core.transformer.depth_memory_triton import forward2, p_step2

        raw = dict(khat=khat.float(), q=query.float(), s=scale.float(), lam=forgetting.float(),
                   v=value)
        if self._phase == "forward" or edge == 0:
            if edge != self._next:
                raise RuntimeError(
                    f"depth_memory_unwind: forward edge {edge}, expected {self._next} (every "
                    "layer must be in a recompute chunk of its own)"
                )
            if edge == 0:
                self.m.zero_()
                self.p.copy_(self.p0)
                self._phase = "forward"
                self._edge0 = raw  # not recomputed: its inputs stay for its backward
            kappa = torch.empty_like(raw["khat"])
            p_step2(self.p, raw["khat"], raw["s"], raw["lam"], kappa)
            read = forward2(self.m, raw["khat"], raw["s"], kappa, value, raw["q"], value.dtype)
            self._next = edge + 1
            if self._next == self.total_edges:
                self._phase, self._unwind = "backward", self.total_edges - 1
            return read
        if edge != self._unwind:
            raise RuntimeError(
                f"depth_memory_unwind: recompute of edge {edge}, the state is at {self._unwind}"
            )
        self._recovered = raw
        if os.environ.get("DEPTH_MEMORY_UNWIND_RECOMPUTE_READ", "0") == "1":
            return torch.einsum("ngkd,ngk->ngd", self.m, raw["q"]).to(value.dtype)
        # the recomputed layer's output is not consumed by its backward (only the graph)
        return torch.empty(self.m.shape[0], self.m.shape[1], self.m.shape[3],
                           device=self.m.device, dtype=value.dtype)

    def _fused_backward2(self, edge, read_grad):
        from megatron.core.transformer.depth_memory_triton import backward2, p_backward, p_step2

        if edge == 0:
            raw = self._edge0
        else:
            if self._phase != "backward" or edge != self._unwind:
                raise RuntimeError(f"depth_memory_unwind: backward of edge {edge} out of order")
            raw = self._recovered
        if edge == self.total_edges - 1 and not self.grad_loaded:
            self.dm.zero_()
            self.dp.zero_()
        khat, s, lam = raw["khat"], raw["s"], raw["lam"]
        kappa = torch.empty_like(khat)
        c = torch.empty_like(s)
        p_prev = torch.empty_like(self.p)
        with torch.no_grad():
            p_step2(self.p, khat, s, lam, kappa, recover=True, c_out=c, p_prev_out=p_prev)
            dq, dv, dkappa_m, dkbar_m, duv = backward2(
                self.m, self.dm, raw["q"], read_grad, kappa, khat, s, raw["v"], c
            )
            d_kb, d_lam = p_backward(p_prev, khat * s.unsqueeze(-1), lam, dkappa_m, self.dp)
            dkbar = dkbar_m + d_kb
            dkhat = dkbar * s.unsqueeze(-1)
            dscale = (dkbar * khat).sum(-1) + duv
            self.p.copy_(p_prev)
        if edge == 0:
            self._phase, self._next, self._edge0 = "forward", 0, None
        else:
            self._unwind, self._recovered = edge - 1, None
        return dkhat, dq, dv, dscale, d_lam


class DepthMemoryStep(torch.autograd.Function):
    """One edge of a workspace as an autograd node: the read is the output; the state is not."""

    @staticmethod
    def forward(ctx, workspace, edge: int, khat, query, value, scale, forgetting):
        ctx.workspace, ctx.edge = workspace, edge
        ctx.value_dtype = value.dtype
        read = workspace.forward(edge, khat, query, value, scale, forgetting)
        return read.to(value.dtype)

    @staticmethod
    def backward(ctx, read_grad):
        dkhat, dq, dvalue, dscale, dforget = ctx.workspace.backward(ctx.edge, read_grad)
        return None, None, dkhat, dq, dvalue.to(ctx.value_dtype), dscale, dforget


def depth_memory_step(workspace, edge, khat, query, value, scale, forgetting) -> torch.Tensor:
    """The read of step ``edge`` (see :class:`DepthMemoryStep`)."""
    return DepthMemoryStep.apply(workspace, edge, khat, query, value, scale, forgetting)


class DepthMemoryStepX(torch.autograd.Function):
    """``depth_memory_unwind`` with the CuTe kernels: the write step of ``edge`` as a
    pass-through node on the stream x. The read is taken by :class:`ReadFromWorkspace`,
    whose backward runs the step's backward and leaves the step's input gradients in the
    workspace; this node's backward hands them to the address networks."""

    @staticmethod
    def forward(ctx, workspace, edge: int, x, khat, query, value, scale, forgetting):
        ctx.workspace, ctx.edge, ctx.value_dtype = workspace, edge, value.dtype
        workspace.forward(edge, khat, query, value, scale, forgetting)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_x):
        dkhat, dq, dvalue, dscale, dforget = ctx.workspace.grad_stash.pop(ctx.edge)
        return None, None, grad_x, dkhat, dq, dvalue.to(ctx.value_dtype), dscale, dforget


class ReadFromWorkspace(torch.autograd.Function):
    """The read of ``edge`` from the workspace (the step's own read in the forward, the read
    kernel's under recompute). ``x`` is the step's output stream: it orders this node after the
    step, so the read's backward (the step's backward) runs before the step node's."""

    @staticmethod
    def forward(ctx, workspace, edge: int, x):
        ctx.workspace, ctx.edge = workspace, edge
        return workspace.read(edge).reshape(x.shape).to(x.dtype)

    @staticmethod
    def backward(ctx, read_grad):
        ctx.workspace.grad_stash[ctx.edge] = ctx.workspace.backward(ctx.edge, read_grad)
        return None, None, None


class StreamX(torch.autograd.Function):
    """x out of the record stream [slots, ..., h] (depth_memory_unwind with the CuTe kernels; slot
    first, so x and the records are contiguous blocks).
    Only x carries a gradient: the backward hands x's gradient to every slot as a broadcast,
    nothing allocated; the record slots' gradient is never read (StreamWrite takes the records
    as data)."""

    @staticmethod
    def forward(ctx, streams):
        ctx.shape = streams.shape
        return streams[0]

    @staticmethod
    def backward(ctx, grad):
        return grad.unsqueeze(0).expand(ctx.shape)


class StreamWrite(torch.autograd.Function):
    """The record stream after the step of ``edge``: x, then the incoming records with the
    edge's own written in by the workspace (in the forward; under recompute they pass through).
    ``streams`` None: the embedding write, the stream starts here."""

    @staticmethod
    def forward(ctx, workspace, edge: int, x, streams):
        out = x.new_empty((workspace.stream_slots(x.dtype), *x.shape))
        records = out[1:]
        if streams is None:
            records.zero_()
        else:
            records.copy_(streams[1:])
        workspace.put_record(edge, records)
        out[0].copy_(x)
        return out

    @staticmethod
    def backward(ctx, grad):
        return None, None, grad[0], None


def stream_x(streams) -> torch.Tensor:
    return StreamX.apply(streams)


def stream_write(workspace, edge, x, streams) -> torch.Tensor:
    return StreamWrite.apply(workspace, edge, x, None if streams is None else streams.detach())


def depth_memory_step_x(workspace, edge, x, khat, query, value, scale, forgetting) -> torch.Tensor:
    return DepthMemoryStepX.apply(workspace, edge, x, khat, query, value, scale, forgetting)


def read_from_workspace(workspace, edge, x) -> torch.Tensor:
    return ReadFromWorkspace.apply(workspace, edge, x)


def make_workspace(config, total_edges: int):
    """The state backend of ``config.depth_memory_impl`` ("torch" or "cute")."""
    groups, key_dim = config.depth_memory_groups, config.depth_memory_key_dim
    value_dim = config.hidden_size // groups
    if config.depth_memory_impl == "cute":
        from cute_ldm.workspace import CuteDepthMemoryWorkspace, CuteRecomputeWorkspace

        settings = dict(
            total_edges=total_edges,
            groups=groups,
            key_dim=key_dim,
            value_dim=value_dim,
            alpha=config.depth_memory_alpha,
            span=config.depth_memory_window,
            running_covariance_dtype=config.depth_memory_covariance_dtype,
        )
        if getattr(config, "depth_memory_unwind", False):
            # full recompute: the per-edge tapes travel in the stream as records, the recomputed
            # layers read from the running state, nothing of the state is checkpointed; the
            # HybridStack sets the recompute groups (CuteRecomputeWorkspace.layout)
            return CuteRecomputeWorkspace(**settings)
        return CuteDepthMemoryWorkspace(**settings)
    if getattr(config, "depth_memory_unwind", False):
        return UnwindingDepthMemoryWorkspace(total_edges, groups, key_dim, value_dim,
                                             config.depth_memory_alpha)
    return TorchDepthMemoryWorkspace(total_edges, groups, key_dim, value_dim,
                                     config.depth_memory_alpha)



# --------------------------------------------------------------------------------------------
# Segments: the state across recompute chunks and pipeline stages (depth_memory_segments)
# --------------------------------------------------------------------------------------------
# One workspace holds one micro-batch's state and tapes, which forces PP 1 and no full recompute
# when the whole stack is one sequence of edges. With segments, every recompute chunk (or, without
# recompute, every stage) is its own sequence of edges: its start loads the state carried in the
# streams, its end writes the state back into them, and the backward runs the other way (the
# workspaces' load_grad / export_grad, the CuTe window fold included). Between segments the streams
# are [s, b, 2 + state slots, h]: x, the pending read r, then the state (M, P) as raw bytes in slots
# of the activation dtype -- a container, never added to (only these functions read it, forward and
# backward).


def _bytes(t: torch.Tensor) -> torch.Tensor:
    """``t`` as raw bytes per token: [n, ...] -> [n, nbytes] uint8 (no copy if contiguous)."""
    t = t.contiguous()
    return t.reshape(t.shape[0], -1).view(torch.uint8)


def state_slots(config) -> int:
    """How many [s, b, h] slots the carried state -- and its gradient, the larger of the two --
    takes for this config (the layouts of the workspaces' ``state_layout``)."""
    ws = make_workspace(config, total_edges=1)
    return max(
        -(-sum(_numel(shape) * dtype.itemsize for shape, dtype in layout) // (2 * config.hidden_size))
        for layout in ws.state_layout()
    )


def _numel(shape) -> int:
    count = 1
    for size in shape:
        count *= size
    return count


def _state_region(full: torch.Tensor) -> torch.Tensor:
    """The state slots of [s, b, 2 + slots, h] streams as raw bytes per token, [n, bytes] -- a
    strided view, no copy (the streams must be contiguous)."""
    s_, b_, slots, h = full.shape
    rows = full.view(s_ * b_, slots * h).view(torch.uint8)
    return rows[:, 2 * h * full.element_size():]


def _copy_bytes(dst_tensors, region: torch.Tensor, *, into_region: bool) -> None:
    """Copy the tensors' bytes to (``into_region``) or from the state region, back to back; the
    region's tail is zeroed when written. No intermediate copies."""
    offset = 0
    for t in dst_tensors:
        flat = t.view(t.shape[0], -1).view(torch.uint8)
        width = flat.shape[1]
        if into_region:
            region[:, offset : offset + width].copy_(flat)
        else:
            flat.copy_(region[:, offset : offset + width])
        offset += width
    if into_region:
        region[:, offset:].zero_()


class SegmentedWorkspace:
    """A workspace whose edges are numbered per segment: ``begin(base, count)`` makes edges
    base .. base + count - 1 its local 0 .. count - 1 (the CuTe windows start at the segment's
    first edge). Buffers are sized for the largest segment (``max_edges``)."""

    def __init__(self, workspace, max_edges: int):
        self.ws, self.max_edges = workspace, max_edges
        self.base, self.count = 0, max_edges

    def ensure(self, n, device):
        count = self.ws.total_edges
        self.ws.total_edges = self.max_edges
        self.ws.ensure(n, device)
        self.ws.total_edges = count

    def begin(self, base: int, count: int, *, carried: bool) -> None:
        self.base, self.count = base, count
        self.ws.total_edges = count
        self.ws.state_loaded = carried
        self.ws.grad_loaded = False

    def forward(self, edge, *args):
        return self.ws.forward(edge - self.base, *args)

    def backward(self, edge, read_grad):
        return self.ws.backward(edge - self.base, read_grad)


class SegmentEnter(torch.autograd.Function):
    """[s, b, 2 + slots, h] -> [s, b, 2, h]: load the carried state into the workspace (or start
    from zero: the embedding's segment); the backward exports dL/d(state) into the state slots."""

    @staticmethod
    def forward(ctx, workspace: SegmentedWorkspace, base: int, count: int, carried: bool, streams):
        s, b, slots, h = streams.shape
        workspace.ensure(s * b, streams.device)
        workspace.begin(base, count, carried=carried)
        if carried:  # straight into the workspace's state buffers
            _copy_bytes(workspace.ws.export_state(), _state_region(streams.contiguous()),
                        into_region=False)
            workspace.ws.state_loaded = True
        ctx.workspace, ctx.cfg = workspace, (base, count, carried, s, b, slots, h)
        return streams[:, :, :2].clone()

    @staticmethod
    def backward(ctx, grad):
        ws = ctx.workspace
        base, count, carried, s, b, slots, h = ctx.cfg
        out = grad.new_empty((s, b, slots, h))
        out[:, :, :2] = grad
        if carried:
            _, glayout = ws.ws.state_layout()
            grads = [t.to(dtype).reshape(t.shape[0], *shape)
                     for t, (shape, dtype) in zip(ws.ws.export_grad(), glayout)]
            _copy_bytes(grads, _state_region(out), into_region=True)
        else:
            out[:, :, 2:].zero_()
        return None, None, None, None, out


class SegmentExit(torch.autograd.Function):
    """[s, b, 2, h] -> [s, b, 2 + slots, h]: append the workspace's state after the segment's last
    edge; the backward loads the incoming dL/d(state) before the segment's steps run backward."""

    @staticmethod
    def forward(ctx, workspace: SegmentedWorkspace, base: int, count: int, carried: bool,
                state_slots: int, streams):
        s, b, _, h = streams.shape
        out = streams.new_empty((s, b, 2 + state_slots, h))
        out[:, :, :2] = streams
        _copy_bytes(workspace.ws.export_state(), _state_region(out), into_region=True)
        ctx.workspace, ctx.cfg = workspace, (base, count, carried, s, b, state_slots, h)
        return out

    @staticmethod
    def backward(ctx, grad):
        ws = ctx.workspace
        base, count, carried, s, b, state_slots, h = ctx.cfg
        ws.begin(base, count, carried=carried)  # (the forward / recompute just ran this segment)
        _copy_bytes(ws.ws.grad_buffers(), _state_region(grad.contiguous()), into_region=False)
        ws.ws.grad_loaded = True
        return None, None, None, None, None, grad[:, :, :2].contiguous()
