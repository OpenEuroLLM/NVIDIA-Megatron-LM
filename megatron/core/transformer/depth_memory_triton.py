# Copyright (c) 2026, OpenEuroLLM. All rights reserved.

"""Fused Triton kernels of the LDM state passes over M for ``depth_memory_unwind``.

M (and dM) is ``[N * G, K, D]`` fp32; every kernel is one program per (token, group) row with
the ``K x D`` tile in registers, so each pass reads (and writes) M once instead of once per
einsum / elementwise op:

    forward:  u = vbar - M' kbar;  M <- M + kappa u';  read = M' q
    recover:  u = (vbar - M' kbar) / c;  read = M' q            (M is not written)
    backward: dq = M dr;  dMt = dM + q dr';  du = dMt' kappa;  dkappa = dMt u;
              M <- M - kappa u';  dkbar = -M du;  vbar = u + M' kbar;  dM <- dMt - kbar du'

The small per-row operands (kbar, kappa, q: [rows, K]; vbar, u, dr: [rows, D]) are fp32.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _forward_kernel(m_ptr, kbar_ptr, kappa_ptr, vbar_ptr, q_ptr, read_ptr,
                    K: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, K)
    d = tl.arange(0, BD)
    dmask = d < D
    tile = row * K * D + k[:, None] * D + d[None, :]
    mask = dmask[None, :]
    m = tl.load(m_ptr + tile, mask=mask, other=0.0)
    kbar = tl.load(kbar_ptr + row * K + k)
    kappa = tl.load(kappa_ptr + row * K + k)
    q = tl.load(q_ptr + row * K + k)
    vbar = tl.load(vbar_ptr + row * D + d, mask=dmask, other=0.0)
    u = vbar - tl.sum(m * kbar[:, None], axis=0)
    m = m + kappa[:, None] * u[None, :]
    tl.store(m_ptr + tile, m, mask=mask)
    tl.store(read_ptr + row * D + d, tl.sum(m * q[:, None], axis=0), mask=dmask)


@triton.jit
def _recover_kernel(m_ptr, kbar_ptr, c_ptr, vbar_ptr, q_ptr, u_ptr, read_ptr,
                    K: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, K)
    d = tl.arange(0, BD)
    dmask = d < D
    tile = row * K * D + k[:, None] * D + d[None, :]
    m = tl.load(m_ptr + tile, mask=dmask[None, :], other=0.0)
    kbar = tl.load(kbar_ptr + row * K + k)
    q = tl.load(q_ptr + row * K + k)
    c = tl.load(c_ptr + row)
    vbar = tl.load(vbar_ptr + row * D + d, mask=dmask, other=0.0)
    u = (vbar - tl.sum(m * kbar[:, None], axis=0)) / c
    tl.store(u_ptr + row * D + d, u, mask=dmask)
    tl.store(read_ptr + row * D + d, tl.sum(m * q[:, None], axis=0), mask=dmask)


@triton.jit
def _backward_kernel(m_ptr, dm_ptr, q_ptr, dr_ptr, kappa_ptr, u_ptr, kbar_ptr,
                     dq_ptr, du_ptr, dkappa_ptr, dkbar_ptr, vbar_ptr,
                     K: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, K)
    d = tl.arange(0, BD)
    dmask = d < D
    tile = row * K * D + k[:, None] * D + d[None, :]
    mask = dmask[None, :]
    m = tl.load(m_ptr + tile, mask=mask, other=0.0)
    dm = tl.load(dm_ptr + tile, mask=mask, other=0.0)
    q = tl.load(q_ptr + row * K + k)
    kappa = tl.load(kappa_ptr + row * K + k)
    kbar = tl.load(kbar_ptr + row * K + k)
    dr = tl.load(dr_ptr + row * D + d, mask=dmask, other=0.0)
    u = tl.load(u_ptr + row * D + d, mask=dmask, other=0.0)
    tl.store(dq_ptr + row * K + k, tl.sum(m * dr[None, :], axis=1))
    dmt = dm + q[:, None] * dr[None, :]
    du = tl.sum(dmt * kappa[:, None], axis=0)
    tl.store(dkappa_ptr + row * K + k, tl.sum(dmt * u[None, :], axis=1))
    m = m - kappa[:, None] * u[None, :]
    tl.store(m_ptr + tile, m, mask=mask)
    tl.store(dkbar_ptr + row * K + k, -tl.sum(m * du[None, :], axis=1))
    tl.store(vbar_ptr + row * D + d, u + tl.sum(m * kbar[:, None], axis=0), mask=dmask)
    tl.store(dm_ptr + tile, dmt - kbar[:, None] * du[None, :], mask=mask)
    tl.store(du_ptr + row * D + d, du, mask=dmask)


def _rows(t, width):
    return t.reshape(-1, width).contiguous()


def _launch_args(m):
    rows, k, d = m.shape[0] * m.shape[1], m.shape[2], m.shape[3]
    return rows, k, d, triton.next_power_of_2(d)


def forward_step(m, kbar, kappa, vbar, q):
    """In place on m [N, G, K, D]; returns the read [N, G, D]."""
    rows, k, d, bd = _launch_args(m)
    read = torch.empty(m.shape[0], m.shape[1], d, device=m.device, dtype=torch.float32)
    _forward_kernel[(rows,)](m, _rows(kbar, k), _rows(kappa, k), _rows(vbar, d), _rows(q, k),
                             read, K=k, D=d, BD=bd, num_warps=4)
    return read


def recover_step(m, kbar, c, vbar, q):
    """(u, read) of the edge m is at, m not written."""
    rows, k, d, bd = _launch_args(m)
    u = torch.empty(m.shape[0], m.shape[1], d, device=m.device, dtype=torch.float32)
    read = torch.empty_like(u)
    _recover_kernel[(rows,)](m, _rows(kbar, k), c.reshape(-1).contiguous(), _rows(vbar, d),
                             _rows(q, k), u, read, K=k, D=d, BD=bd, num_warps=4)
    return u, read


def backward_step(m, dm, q, dr, kappa, u, kbar):
    """In place on m (unwound) and dm; returns (dq, du, dkappa, dkbar_m, vbar)."""
    rows, k, d, bd = _launch_args(m)
    n, g = m.shape[0], m.shape[1]
    f32 = dict(device=m.device, dtype=torch.float32)
    dq, dkappa, dkbar = (torch.empty(n, g, k, **f32) for _ in range(3))
    du, vbar = (torch.empty(n, g, d, **f32) for _ in range(2))
    _backward_kernel[(rows,)](m, dm, _rows(q, k), _rows(dr, d), _rows(kappa, k), _rows(u, d),
                              _rows(kbar, k), dq, du, dkappa, dkbar, vbar,
                              K=k, D=d, BD=bd, num_warps=4)
    return dq, du, dkappa, dkbar, vbar


# ---- the covariance P ([rows, K, K] fp32), R rows per program ----------------------------------
#   forward:  a = P kbar, s = lam + kbar . a, kappa = a / s, P <- (P - kappa b') / lam (b = P' kbar)
#   recover:  kappa = P_e kbar, c = 1 - kappa . kbar, P_{e-1} = lam (P_e + kappa kappa' / c)
#   backward: the vjp of (kappa, P_new) w.r.t. (P_{e-1}, kbar, lam) at (g_kappa, g_P)


@triton.jit
def _p_forward_kernel(p_ptr, kbar_ptr, lam_ptr, kappa_ptr, rows, G,
                      K: tl.constexpr, R: tl.constexpr):
    r = tl.program_id(0) * R + tl.arange(0, R)
    rmask = r < rows
    i = tl.arange(0, K)
    r64 = r.to(tl.int64)
    tile = r64[:, None, None] * K * K + i[None, :, None] * K + i[None, None, :]
    m3 = rmask[:, None, None]
    p = tl.load(p_ptr + tile, mask=m3, other=0.0)
    kb = tl.load(kbar_ptr + r64[:, None] * K + i[None, :], mask=rmask[:, None], other=0.0)
    lam = tl.load(lam_ptr + (r % G), mask=rmask, other=1.0)
    a = tl.sum(p * kb[:, None, :], axis=2)            # P kbar
    b = tl.sum(p * kb[:, :, None], axis=1)            # P' kbar
    s = lam + tl.sum(kb * a, axis=1)
    kappa = a / s[:, None]
    p = (p - kappa[:, :, None] * b[:, None, :]) / lam[:, None, None]
    tl.store(p_ptr + tile, p, mask=m3)
    tl.store(kappa_ptr + r64[:, None] * K + i[None, :], kappa, mask=rmask[:, None])


@triton.jit
def _p_recover_kernel(p_ptr, kbar_ptr, lam_ptr, kappa_ptr, c_ptr, p_prev_ptr, rows, G,
                      K: tl.constexpr, R: tl.constexpr):
    r = tl.program_id(0) * R + tl.arange(0, R)
    rmask = r < rows
    i = tl.arange(0, K)
    r64 = r.to(tl.int64)
    tile = r64[:, None, None] * K * K + i[None, :, None] * K + i[None, None, :]
    m3 = rmask[:, None, None]
    p = tl.load(p_ptr + tile, mask=m3, other=0.0)
    kb = tl.load(kbar_ptr + r64[:, None] * K + i[None, :], mask=rmask[:, None], other=0.0)
    lam = tl.load(lam_ptr + (r % G), mask=rmask, other=1.0)
    kappa = tl.sum(p * kb[:, None, :], axis=2)
    c = 1.0 - tl.sum(kappa * kb, axis=1)
    p_prev = lam[:, None, None] * (p + kappa[:, :, None] * (kappa / c[:, None])[:, None, :])
    tl.store(kappa_ptr + r64[:, None] * K + i[None, :], kappa, mask=rmask[:, None])
    tl.store(c_ptr + r64, c, mask=rmask)
    tl.store(p_prev_ptr + tile, p_prev, mask=m3)


@triton.jit
def _p_backward_kernel(p_ptr, kbar_ptr, lam_ptr, gk_ptr, gp_ptr, dkb_ptr, dlam_ptr, rows, G,
                       K: tl.constexpr, R: tl.constexpr):
    """p_ptr: P_{e-1}; gp_ptr: dL/dP_e in, dL/dP_{e-1} out (in place); dkb (+=), dlam per row."""
    r = tl.program_id(0) * R + tl.arange(0, R)
    rmask = r < rows
    i = tl.arange(0, K)
    r64 = r.to(tl.int64)
    tile = r64[:, None, None] * K * K + i[None, :, None] * K + i[None, None, :]
    m3 = rmask[:, None, None]
    p = tl.load(p_ptr + tile, mask=m3, other=0.0)
    gpn = tl.load(gp_ptr + tile, mask=m3, other=0.0)
    vec = r64[:, None] * K + i[None, :]
    kb = tl.load(kbar_ptr + vec, mask=rmask[:, None], other=0.0)
    gk = tl.load(gk_ptr + vec, mask=rmask[:, None], other=0.0)
    lam = tl.load(lam_ptr + (r % G), mask=rmask, other=1.0)
    a = tl.sum(p * kb[:, None, :], axis=2)            # P kb
    b = tl.sum(p * kb[:, :, None], axis=1)            # P' kb
    s = lam + tl.sum(kb * a, axis=1)
    gm = gpn / lam[:, None, None]                     # dL/dX, X = P - a b' / s = lam P_new
    x = p - a[:, :, None] * b[:, None, :] / s[:, None, None]
    gmb = tl.sum(gm * b[:, None, :], axis=2)          # G b
    gta = tl.sum(gm * a[:, :, None], axis=1)          # G' a
    agb = tl.sum(a * gmb, axis=1)                     # a' G b
    ga = gk / s[:, None] - gmb / s[:, None]
    gb = -gta / s[:, None]
    gs = (-tl.sum(gk * a, axis=1) + agb) / (s * s)
    ga = ga + gs[:, None] * kb
    gkb = gs[:, None] * a
    gp = gm + ga[:, :, None] * kb[:, None, :] + kb[:, :, None] * gb[:, None, :]
    gkb = gkb + tl.sum(p * ga[:, :, None], axis=1) + tl.sum(p * gb[:, None, :], axis=2)
    glam = gs - tl.sum(tl.sum(gm * x, axis=2), axis=1) / lam
    tl.store(gp_ptr + tile, gp, mask=m3)
    tl.store(dkb_ptr + vec, gkb, mask=rmask[:, None])
    tl.store(dlam_ptr + r64, glam, mask=rmask)


_R = 16


def p_forward(p, kbar, lam, kappa_out):
    """In place on p [N, G, K, K]; kappa_out [N, G, K] written."""
    n, g, k = kbar.shape
    rows = n * g
    _p_forward_kernel[(triton.cdiv(rows, _R),)](p, kbar.contiguous(), lam.contiguous(), kappa_out,
                                                 rows, g, K=k, R=_R, num_warps=4)


def p_recover(p, kbar, lam, kappa_out, c_out, p_prev_out):
    n, g, k = kbar.shape
    rows = n * g
    _p_recover_kernel[(triton.cdiv(rows, _R),)](p, kbar.contiguous(), lam.contiguous(),
                                                 kappa_out, c_out, p_prev_out, rows, g,
                                                 K=k, R=_R, num_warps=4)


def p_backward(p_prev, kbar, lam, g_kappa, g_p):
    """g_p (dL/dP_e) becomes dL/dP_{e-1} in place; returns (d kbar [N, G, K], d lam [G])."""
    n, g, k = kbar.shape
    rows = n * g
    dkb = torch.empty_like(kbar)
    dlam = torch.empty(n, g, device=kbar.device, dtype=torch.float32)
    _p_backward_kernel[(triton.cdiv(rows, _R),)](p_prev, kbar.contiguous(), lam.contiguous(),
                                                  g_kappa.contiguous(), g_p, dkb, dlam, rows, g,
                                                  K=k, R=_R, num_warps=4)
    return dkb, dlam.sum(0)


@triton.jit
def _unwind_backward_kernel(m_ptr, dm_ptr, q_ptr, dr_ptr, kappa_ptr, kbar_ptr, vbar_ptr, c_ptr,
                            dq_ptr, du_ptr, dkappa_ptr, dkbar_ptr,
                            K: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    """The backward of the edge M is at, with u recovered in the same pass:
    u = (vbar - M_e' kbar) / c, then as _backward_kernel (M unwound, dM updated)."""
    row = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, K)
    d = tl.arange(0, BD)
    dmask = d < D
    tile = row * K * D + k[:, None] * D + d[None, :]
    mask = dmask[None, :]
    m = tl.load(m_ptr + tile, mask=mask, other=0.0)
    dm = tl.load(dm_ptr + tile, mask=mask, other=0.0)
    q = tl.load(q_ptr + row * K + k)
    kappa = tl.load(kappa_ptr + row * K + k)
    kbar = tl.load(kbar_ptr + row * K + k)
    c = tl.load(c_ptr + row)
    dr = tl.load(dr_ptr + row * D + d, mask=dmask, other=0.0)
    vbar = tl.load(vbar_ptr + row * D + d, mask=dmask, other=0.0)
    u = (vbar - tl.sum(m * kbar[:, None], axis=0)) / c
    tl.store(dq_ptr + row * K + k, tl.sum(m * dr[None, :], axis=1))
    dmt = dm + q[:, None] * dr[None, :]
    du = tl.sum(dmt * kappa[:, None], axis=0)
    tl.store(dkappa_ptr + row * K + k, tl.sum(dmt * u[None, :], axis=1))
    m = m - kappa[:, None] * u[None, :]
    tl.store(m_ptr + tile, m, mask=mask)
    tl.store(dkbar_ptr + row * K + k, -tl.sum(m * du[None, :], axis=1))
    tl.store(dm_ptr + tile, dmt - kbar[:, None] * du[None, :], mask=mask)
    tl.store(du_ptr + row * D + d, du, mask=dmask)


def unwind_backward_step(m, dm, q, dr, kappa, kbar, vbar, c):
    """In place on m (unwound) and dm; returns (dq, du, dkappa_m, dkbar_m)."""
    rows, k, d, bd = _launch_args(m)
    n, g = m.shape[0], m.shape[1]
    f32 = dict(device=m.device, dtype=torch.float32)
    dq, dkappa, dkbar = (torch.empty(n, g, k, **f32) for _ in range(3))
    du = torch.empty(n, g, d, **f32)
    _unwind_backward_kernel[(rows,)](m, dm, _rows(q, k), _rows(dr, d), _rows(kappa, k),
                                     _rows(kbar, k), _rows(vbar, d), c.reshape(-1).contiguous(),
                                     dq, du, dkappa, dkbar, K=k, D=d, BD=bd, num_warps=4)
    return dq, du, dkappa, dkbar


# ---- v2: raw inputs (khat, scale, bf16 value, query), bf16 outputs, R rows per program --------
# kbar = scale * khat and vbar = scale * value are formed in registers; the read and d value are
# written in the activation dtype; the backward also returns sum_d du * value (for d scale).


@triton.jit
def _fwd2_kernel(m_ptr, khat_ptr, s_ptr, kappa_ptr, v_ptr, q_ptr, read_ptr, rows,
                 K: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, R: tl.constexpr):
    r = tl.program_id(0) * R + tl.arange(0, R)
    rmask = r < rows
    r64 = r.to(tl.int64)
    k = tl.arange(0, K)
    d = tl.arange(0, BD)
    dmask = d < D
    tile = r64[:, None, None] * K * D + k[None, :, None] * D + d[None, None, :]
    m3 = rmask[:, None, None] & dmask[None, None, :]
    vec_k = r64[:, None] * K + k[None, :]
    vec_d = r64[:, None] * D + d[None, :]
    m2d = rmask[:, None] & dmask[None, :]
    m = tl.load(m_ptr + tile, mask=m3, other=0.0)
    s = tl.load(s_ptr + r64, mask=rmask, other=0.0)
    kbar = tl.load(khat_ptr + vec_k, mask=rmask[:, None], other=0.0) * s[:, None]
    kappa = tl.load(kappa_ptr + vec_k, mask=rmask[:, None], other=0.0)
    q = tl.load(q_ptr + vec_k, mask=rmask[:, None], other=0.0)
    vbar = tl.load(v_ptr + vec_d, mask=m2d, other=0.0).to(tl.float32) * s[:, None]
    u = vbar - tl.sum(m * kbar[:, :, None], axis=1)
    m = m + kappa[:, :, None] * u[:, None, :]
    tl.store(m_ptr + tile, m, mask=m3)
    read = tl.sum(m * q[:, :, None], axis=1)
    tl.store(read_ptr + vec_d, read.to(read_ptr.dtype.element_ty), mask=m2d)


@triton.jit
def _bwd2_kernel(m_ptr, dm_ptr, q_ptr, dr_ptr, kappa_ptr, khat_ptr, s_ptr, v_ptr, c_ptr,
                 dq_ptr, dv_ptr, dkappa_ptr, dkbar_ptr, duv_ptr, rows,
                 K: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, R: tl.constexpr):
    r = tl.program_id(0) * R + tl.arange(0, R)
    rmask = r < rows
    r64 = r.to(tl.int64)
    k = tl.arange(0, K)
    d = tl.arange(0, BD)
    dmask = d < D
    tile = r64[:, None, None] * K * D + k[None, :, None] * D + d[None, None, :]
    m3 = rmask[:, None, None] & dmask[None, None, :]
    vec_k = r64[:, None] * K + k[None, :]
    vec_d = r64[:, None] * D + d[None, :]
    m2d = rmask[:, None] & dmask[None, :]
    m2k = rmask[:, None]
    m = tl.load(m_ptr + tile, mask=m3, other=0.0)
    dm = tl.load(dm_ptr + tile, mask=m3, other=0.0)
    s = tl.load(s_ptr + r64, mask=rmask, other=1.0)
    c = tl.load(c_ptr + r64, mask=rmask, other=1.0)
    q = tl.load(q_ptr + vec_k, mask=m2k, other=0.0)
    kappa = tl.load(kappa_ptr + vec_k, mask=m2k, other=0.0)
    kbar = tl.load(khat_ptr + vec_k, mask=m2k, other=0.0) * s[:, None]
    value = tl.load(v_ptr + vec_d, mask=m2d, other=0.0).to(tl.float32)
    dr = tl.load(dr_ptr + vec_d, mask=m2d, other=0.0).to(tl.float32)
    u = (value * s[:, None] - tl.sum(m * kbar[:, :, None], axis=1)) / c[:, None]
    tl.store(dq_ptr + vec_k, tl.sum(m * dr[:, None, :], axis=2), mask=m2k)
    dmt = dm + q[:, :, None] * dr[:, None, :]
    du = tl.sum(dmt * kappa[:, :, None], axis=1)
    tl.store(dkappa_ptr + vec_k, tl.sum(dmt * u[:, None, :], axis=2), mask=m2k)
    m = m - kappa[:, :, None] * u[:, None, :]
    tl.store(m_ptr + tile, m, mask=m3)
    tl.store(dkbar_ptr + vec_k, -tl.sum(m * du[:, None, :], axis=2), mask=m2k)
    tl.store(dm_ptr + tile, dmt - kbar[:, :, None] * du[:, None, :], mask=m3)
    tl.store(dv_ptr + vec_d, (du * s[:, None]).to(dv_ptr.dtype.element_ty), mask=m2d)
    tl.store(duv_ptr + r64, tl.sum(du * value, axis=1), mask=rmask)


@triton.jit
def _p_fwd2_kernel(p_ptr, khat_ptr, s_ptr, lam_ptr, kappa_ptr, c_ptr, p_prev_ptr, rows, G,
                   RECOVER: tl.constexpr, K: tl.constexpr, R: tl.constexpr):
    """RECOVER=0: the forward update of P (kappa out); 1: the recovery at P_e (kappa, c, P_{e-1})."""
    r = tl.program_id(0) * R + tl.arange(0, R)
    rmask = r < rows
    i = tl.arange(0, K)
    r64 = r.to(tl.int64)
    tile = r64[:, None, None] * K * K + i[None, :, None] * K + i[None, None, :]
    m3 = rmask[:, None, None]
    vec = r64[:, None] * K + i[None, :]
    p = tl.load(p_ptr + tile, mask=m3, other=0.0)
    s = tl.load(s_ptr + r64, mask=rmask, other=0.0)
    kb = tl.load(khat_ptr + vec, mask=rmask[:, None], other=0.0) * s[:, None]
    lam = tl.load(lam_ptr + (r % G), mask=rmask, other=1.0)
    if RECOVER:
        kappa = tl.sum(p * kb[:, None, :], axis=2)
        c = 1.0 - tl.sum(kappa * kb, axis=1)
        p_prev = lam[:, None, None] * (p + kappa[:, :, None] * (kappa / c[:, None])[:, None, :])
        tl.store(c_ptr + r64, c, mask=rmask)
        tl.store(p_prev_ptr + tile, p_prev, mask=m3)
    else:
        a = tl.sum(p * kb[:, None, :], axis=2)
        b = tl.sum(p * kb[:, :, None], axis=1)
        kappa = a / (lam + tl.sum(kb * a, axis=1))[:, None]
        tl.store(p_ptr + tile, (p - kappa[:, :, None] * b[:, None, :]) / lam[:, None, None], mask=m3)
    tl.store(kappa_ptr + vec, kappa, mask=rmask[:, None])


_RM = 4  # rows per program of the M passes


def _m_args(m):
    n, g, k, d = m.shape
    return n * g, k, d, triton.next_power_of_2(d)


def forward2(m, khat, scale, kappa, value, query, read_dtype):
    rows, k, d, bd = _m_args(m)
    read = torch.empty(m.shape[0], m.shape[1], d, device=m.device, dtype=read_dtype)
    _fwd2_kernel[(triton.cdiv(rows, _RM),)](
        m, khat.contiguous(), scale.contiguous(), kappa, value.contiguous(), query.contiguous(),
        read, rows, K=k, D=d, BD=bd, R=_RM, num_warps=4)
    return read


def backward2(m, dm, query, read_grad, kappa, khat, scale, value, c):
    """In place on m (unwound) and dm; returns (dq, d value [value dtype], dkappa_m, dkbar_m,
    sum_d du * value)."""
    rows, k, d, bd = _m_args(m)
    n, g = m.shape[0], m.shape[1]
    f32 = dict(device=m.device, dtype=torch.float32)
    dq, dkappa, dkbar = (torch.empty(n, g, k, **f32) for _ in range(3))
    dv = torch.empty(n, g, d, device=m.device, dtype=value.dtype)
    duv = torch.empty(n, g, **f32)
    _bwd2_kernel[(triton.cdiv(rows, _RM),)](
        m, dm, query.contiguous(), read_grad.contiguous(), kappa, khat.contiguous(),
        scale.contiguous(), value.contiguous(), c, dq, dv, dkappa, dkbar, duv, rows,
        K=k, D=d, BD=bd, R=_RM, num_warps=4)
    return dq, dv, dkappa, dkbar, duv


def p_step2(p, khat, scale, lam, kappa_out, recover=False, c_out=None, p_prev_out=None):
    n, g, k = khat.shape
    rows = n * g
    dummy = kappa_out
    _p_fwd2_kernel[(triton.cdiv(rows, _R),)](
        p, khat.contiguous(), scale.contiguous(), lam.contiguous(), kappa_out,
        c_out if recover else dummy, p_prev_out if recover else p, rows, g,
        RECOVER=recover, K=k, R=_R, num_warps=4)
