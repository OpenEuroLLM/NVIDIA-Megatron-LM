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

    def backward(self, edge, read_grad):
        """The reverse of step ``edge`` (called in strictly decreasing edge order): the
        gradients of (khat, query, value, scale, forgetting [G]), and ``M`` unwound to
        ``M_{edge-1}``."""
        if edge == self.total_edges - 1 and not self.grad_loaded:
            self.dm.zero_()
            self.dp.zero_()
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


def make_workspace(config, total_edges: int):
    """The state backend of ``config.depth_memory_impl`` ("torch" or "cute")."""
    groups, key_dim = config.depth_memory_groups, config.depth_memory_key_dim
    value_dim = config.hidden_size // groups
    if config.depth_memory_impl == "cute":
        from cute_ldm.workspace import CuteDepthMemoryWorkspace

        return CuteDepthMemoryWorkspace(
            total_edges=total_edges,
            groups=groups,
            key_dim=key_dim,
            value_dim=value_dim,
            alpha=config.depth_memory_alpha,
            span=config.depth_memory_window,
            running_covariance_dtype=config.depth_memory_covariance_dtype,
        )
    return TorchDepthMemoryWorkspace(total_edges, groups, key_dim, value_dim,
                                     config.depth_memory_alpha)

