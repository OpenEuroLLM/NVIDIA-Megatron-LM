# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


# pylint: disable=missing-function-docstring, missing-class-docstring

import torch
import torch.nn.functional as F

from megatron.core.jit import jit_fuser
from megatron.core.utils import nvtx_decorator

###### BIAS SWIGLU FUSION/ NO AUTOGRAD ################


@jit_fuser
def swiglu(y):
    """Performs SwiGLU (Swish-Gated Linear Unit) activation function.

    Args:
        y (torch.Tensor): Input tensor to be split into two halves along the last dimension.

    Returns:
        torch.Tensor: Result of SwiGLU activation: SiLU(y1) * y2, where y1, y2 are the split halves.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2


@jit_fuser
def bias_swiglu(y, bias):
    """Performs SwiGLU activation with bias addition.

    Args:
        y (torch.Tensor): Input tensor.
        bias (torch.Tensor): Bias tensor to be added to input.

    Returns:
        torch.Tensor: Result of bias addition followed by SwiGLU activation.
    """
    y = y + bias
    return swiglu(y)


@jit_fuser
def weighted_swiglu(y, weights):
    dtype = y.dtype
    res = swiglu(y) * weights
    return res.to(dtype)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@jit_fuser
def swiglu_back(g, y):
    """Computes the gradient for the SwiGLU activation function.

    Args:
        g (torch.Tensor): Gradient tensor from the subsequent layer.
        y (torch.Tensor): Input tensor that was used in the forward pass.

    Returns:
        torch.Tensor: Gradient with respect to the input tensor, computed using the
            chain rule and the derivative of the SiLU activation function.
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.cat(
        (g * torch.sigmoid(y_1) * (1 + y_1 * (1 - torch.sigmoid(y_1))) * y_2, g * F.silu(y_1)), -1
    )


@jit_fuser
def bias_swiglu_back(g, y, bias):
    """Computes the gradient for the biased SwiGLU activation function.

    Args:
        g (torch.Tensor): Gradient tensor from the subsequent layer.
        y (torch.Tensor): Input tensor that was used in the forward pass.
        bias (torch.Tensor): Bias tensor that was added in the forward pass.

    Returns:
        torch.Tensor: Gradient with respect to the input tensor, computed after
            applying the bias addition.
    """
    y = y + bias
    return swiglu_back(g, y)


@jit_fuser
def weighted_swiglu_back(g, y, weights):
    input_dtype = y.dtype
    w_dtype = weights.dtype
    input_grad = swiglu_back(g * weights, y)
    # precison of w may be higher than y and g, so we need to cast g to w_dtype
    weights_grad = swiglu(y) * g.to(w_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


# Clamped SwiGLU (gpt-oss / DeepSeek-V4.1: activation_func_clamp_value): the gate is capped from
# above, the linear half to [-limit, limit]; the gradient is zero where a clamp is active.
@jit_fuser
def clamped_swiglu(y, limit: float):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1.clamp(max=limit)) * y_2.clamp(min=-limit, max=limit)


@jit_fuser
def clamped_swiglu_back(g, y, limit: float):
    y_1, y_2 = torch.chunk(y, 2, -1)
    c_1 = y_1.clamp(max=limit)
    c_2 = y_2.clamp(min=-limit, max=limit)
    s_1 = torch.sigmoid(c_1)
    d_1 = g * s_1 * (1 + c_1 * (1 - s_1)) * c_2 * (y_1 <= limit)
    d_2 = g * F.silu(c_1) * (y_2.abs() <= limit)
    return torch.cat((d_1, d_2), -1)


@jit_fuser
def weighted_clamped_swiglu(y, weights, limit: float):
    """``clamped_swiglu(y) * weights`` in the dtype of ``y`` -- one fused kernel (the
    probability-weighted expert activation of TEGroupedMLP / MoonEP with clamping)."""
    dtype = y.dtype
    y_1, y_2 = torch.chunk(y, 2, -1)
    res = F.silu(y_1.clamp(max=limit)) * y_2.clamp(min=-limit, max=limit) * weights
    return res.to(dtype)


@jit_fuser
def weighted_clamped_swiglu_back(g, y, weights, limit: float):
    """The gradients of :func:`weighted_clamped_swiglu` w.r.t. ``y`` (dtype of ``y``) and
    ``weights`` (dtype of ``weights``), fused."""
    input_dtype, w_dtype = y.dtype, weights.dtype
    y_1, y_2 = torch.chunk(y, 2, -1)
    c_1 = y_1.clamp(max=limit)
    c_2 = y_2.clamp(min=-limit, max=limit)
    s_1 = torch.sigmoid(c_1)
    act_1 = c_1 * s_1
    gw = g * weights
    d_1 = gw * s_1 * (1 + c_1 * (1 - s_1)) * c_2 * (y_1 <= limit)
    d_2 = gw * act_1 * (y_2.abs() <= limit)
    weights_grad = torch.sum(act_1 * c_2 * g.to(w_dtype), dim=-1, keepdim=True)
    return torch.cat((d_1, d_2), -1).to(input_dtype), weights_grad.to(w_dtype)


def _swiglu_fwd(y, limit):
    return swiglu(y) if limit is None else clamped_swiglu(y, limit)


def _swiglu_bwd(g, y, limit):
    return swiglu_back(g, y) if limit is None else clamped_swiglu_back(g, y, limit)


class BiasSwiGLUFunction(torch.autograd.Function):
    """Custom autograd function for SwiGLU activation with bias support."""

    @staticmethod
    @nvtx_decorator()
    def forward(ctx, input, bias, fp8_input_store, cpu_offload_input, limit=None):
        """Forward pass of biased SwiGLU activation.

        Args:
            ctx: Autograd context object for saving tensors for backward pass.
            input (torch.Tensor): Input tensor to apply SwiGLU to.
            bias (torch.Tensor): Bias tensor to be added to input before SwiGLU.
            fp8_input_store (bool): If True, stores intermediate values in FP8 format.

        Returns:
            torch.Tensor: Result of applying bias addition followed by SwiGLU activation.
        """
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
            bias.activation_offloading = True
        ctx.save_for_backward(input_for_backward, bias)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        ctx.limit = limit
        return bias_swiglu(input, bias) if limit is None else clamped_swiglu(input + bias, limit)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        """Backward pass of biased SwiGLU activation.

        Args:
            ctx: Autograd context object containing saved tensors from forward pass.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            tuple: Tuple containing:
                - Gradient with respect to the input tensor
                - Gradient with respect to the bias tensor
                - None for fp8_input_store parameter
        """
        input, bias = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp = _swiglu_bwd(grad_output, input + bias, ctx.limit)
        return tmp, tmp, None, None, None


class SwiGLUFunction(torch.autograd.Function):
    """Custom autograd function for SwiGLU activation without bias."""

    @staticmethod
    @nvtx_decorator()
    def forward(ctx, input, fp8_input_store, cpu_offload_input, limit=None):
        """Forward pass of SwiGLU activation.

        Args:
            ctx: Autograd context object for saving tensors for backward pass.
            input (torch.Tensor): Input tensor to apply SwiGLU to.
            fp8_input_store (bool): If True, stores intermediate values in FP8 format.

        Returns:
            torch.Tensor: Result of applying SwiGLU activation.
        """
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
        ctx.save_for_backward(input_for_backward)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        ctx.limit = limit
        return _swiglu_fwd(input, limit)

    @staticmethod
    @nvtx_decorator()
    def backward(ctx, grad_output):
        """Backward pass of SwiGLU activation.

        Args:
            ctx: Autograd context object containing saved tensors from forward pass.
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            tuple: Tuple containing:
                - Gradient with respect to the input tensor
                - None for fp8_input_store parameter
        """
        input = ctx.saved_tensors[0]
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp = _swiglu_bwd(grad_output, input, ctx.limit)
        return tmp, None, None, None


class WeightedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weights, fp8_input_store, limit=None):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward, weights)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        ctx.limit = limit
        if limit is None:
            return weighted_swiglu(input, weights)
        return weighted_clamped_swiglu(input, weights, limit)

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        if ctx.limit is None:
            tmp, wgrad = weighted_swiglu_back(grad_output, input, weights)
        else:
            tmp, wgrad = weighted_clamped_swiglu_back(grad_output, input, weights, ctx.limit)
        return tmp, wgrad, None, None


def bias_swiglu_impl(input, bias, fp8_input_store=False, cpu_offload_input=False, limit=None):
    """Implementation of biased SwiGLU that handles different input shapes.

    This function reshapes the input if necessary, applies the SwiGLU activation
    (with or without bias), and restores the original shape.

    Args:
        input (torch.Tensor): Input tensor to apply SwiGLU activation.
        bias (torch.Tensor, optional): Bias tensor to be added to input. If None,
            uses the bias-free SwiGLU variant.
        fp8_input_store (bool, optional): Whether to store intermediate values in FP8 format.
            Defaults to False.

    Returns:
        torch.Tensor: Result of biased SwiGLU activation.

    Raises:
        AssertionError: If input tensor does not have 2 or 3 dimensions.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        output = BiasSwiGLUFunction.apply(input, bias, fp8_input_store, cpu_offload_input, limit)
    else:
        output = SwiGLUFunction.apply(input, fp8_input_store, cpu_offload_input, limit)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


def weighted_bias_swiglu_impl(input, bias, weights, fp8_input_store=False, limit=None):
    """
    Token-wise-weighted bias swiglu fusion.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    if bias is not None:
        raise NotImplementedError("Bias is not supported for weighted swiglu fusion")
    else:
        output = WeightedSwiGLUFunction.apply(input, weights, fp8_input_store, limit)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


# bias_swiglu_impl = BiasSwiGLUFunction.apply
# swiglu_impl = SwiGLUFunction.apply
