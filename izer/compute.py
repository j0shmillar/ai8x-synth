###################################################################################################
# Copyright (C) 2019-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Python implementation of Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d, Pool1d, Pool2d,
Eltwise, and Linear.
Compatible with PyTorch.
"""
import os

import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.typing import ArrayLike

from typing import Optional

from . import op, state, stats
from .eprint import eprint


def debug_open(
        layer: int,
        base_directory: str,
        test_name: str,
        log_filename: str,  # pylint: disable=unused-argument
) -> None:
    """
    Create debug log for a layer
    """
    if not state.debug_computation:
        return
    state.debug_log = open(
        os.path.join(base_directory, test_name, f'compute-{layer}.csv'),
        mode='w',
        encoding='utf-8',
    )


def debug_print(
        t,
) -> None:
    """
    Print to the compute debug log
    """
    if not state.debug_computation:
        return
    print(t, file=state.debug_log)


def debug_close() -> None:
    """
    Close the compute debug log
    """
    if not state.debug_computation:
        return
    assert state.debug_log is not None
    state.debug_log.close()
    state.debug_log = None


def conv2d(
        data,
        weight,
        bias,
        input_size,
        output_size,
        kernel_size,
        stride,
        pad,
        dilation,
        fractional_stride,
        output_pad,
        groups=1,
) -> ArrayLike:
    """
    Compute a 2D convolution.

    Note that all PyTorch numbers are ordered (C, H, W)
    """
    assert data.shape == tuple(input_size)
    in_channels = input_size[0]
    out_channels = output_size[0]

    # Stretch data for fractionally-strided convolution
    if fractional_stride[0] > 1 or fractional_stride[1] > 1:
        ndata = np.zeros((data.shape[0],
                          data.shape[1] * fractional_stride[0] - 1,
                          data.shape[2] * fractional_stride[1] - 1),
                         dtype=data.dtype)
        ndata[:, 0::fractional_stride[0], 0::fractional_stride[1]] = data
        data = ndata

    # Create zero padding around data
    if pad[0] or pad[1] or output_pad[0] or output_pad[1]:
        data = np.pad(data, pad_width=((0, 0),
                                       (pad[0], pad[0] + output_pad[0]),
                                       (pad[1], pad[1] + output_pad[1])),
                      mode='constant', constant_values=0)

    if dilation[0] > 1 or dilation[1] > 1:
        # Stretch weights for dilation
        nweight = np.zeros((weight.shape[0], weight.shape[1],
                            (kernel_size[0] - 1) * dilation[0] + 1,
                            (kernel_size[1] - 1) * dilation[1] + 1),
                           dtype=weight.dtype)
        nweight[:, :, 0::dilation[0], 0::dilation[1]] = weight
        weight = nweight

    h = (data.shape[1] - weight.shape[3]) // stride[0] + 1  # Resulting output height
    w = (data.shape[2] - weight.shape[2]) // stride[1] + 1  # Resulting output width

    view = as_strided(data,
                      shape=(h, w, data.shape[0], weight.shape[2], weight.shape[3]),
                      strides=((data.strides[1] * stride[0], data.strides[2] * stride[1],
                                data.strides[0], data.strides[1], data.strides[2])),
                      writeable=False)

    if groups > 1:
        nweight = np.zeros((weight.shape[0], in_channels, weight.shape[2], weight.shape[3]),
                           dtype=weight.dtype)
        for i in range(weight.shape[0]):
            for j in range(in_channels // groups):
                nweight[i, i * (in_channels // groups) + j, :, :] = weight[i, j, :, :]
        weight = nweight

    output = np.tensordot(view, weight, axes=((2, 3, 4), (1, 2, 3))).transpose(2, 0, 1)

    # Apply bias
    if bias is not None:
        for k in range(out_channels):
            output[k] += bias[k]

    assert output.shape == tuple(output_size), \
        f'Shape mismatch: NumPy result {output.shape} vs expected {output_size}'

    return output


def convtranspose2d(
        data,
        weight,
        bias,
        input_size,
        output_size,
        kernel_size,
        stride,
        pad,
        dilation,
        fractional_stride,
        output_pad,
        groups=1,
) -> ArrayLike:
    """
    Compute a transposed 2D convolution.
    """

    return conv2d(
        data,
        weight,
        bias,
        input_size,
        output_size,
        kernel_size,
        stride,
        (
            dilation[0] * (kernel_size[0] - 1) - pad[0],
            dilation[1] * (kernel_size[1] - 1) - pad[1]
        ),
        dilation,
        fractional_stride,
        output_pad,
        groups,
    )


def conv1d(
        data,
        weight,
        bias,
        input_size,
        output_size,
        kernel_size,
        stride,
        pad,
        dilation,
        fractional_stride=1,
        output_pad=0,
        groups=1,
) -> ArrayLike:
    """
    Compute a 1D convolution.

    Note that all PyTorch numbers are ordered (C, L)
    """
    assert data.shape == tuple(input_size)
    in_channels = input_size[0]
    out_channels = output_size[0]

    weight = weight.reshape(out_channels, input_size[0] // groups, -1)
    data = data.reshape(input_size[0], -1)

    output = np.empty(shape=(output_size[0], output_size[1]), dtype=np.int64)

    # Stretch data for fractionally-strided convolution
    if fractional_stride > 1:
        ndata = np.zeros((data.shape[0],
                          data.shape[1] * fractional_stride - 1),
                         dtype=data.dtype)
        ndata[:, 0::fractional_stride] = data
        data = ndata

    # Create zero padding around data
    if pad or output_pad:
        data = np.pad(data, pad_width=((0, 0), (pad, pad + output_pad)),
                      mode='constant', constant_values=0)

    if dilation > 1:
        # Stretch weights for dilation
        nweight = np.zeros((weight.shape[0], weight.shape[1],
                            (kernel_size - 1) * dilation + 1),
                           dtype=weight.dtype)
        nweight[:, :, 0::dilation] = weight
        weight = nweight

    ll = (data.shape[1] - weight.shape[2]) // stride + 1  # Resulting output length

    view = as_strided(data,
                      shape=(ll, data.shape[0], weight.shape[2]),
                      strides=((data.strides[1] * stride,
                                data.strides[0], data.strides[1])),
                      writeable=False)

    if groups > 1:
        nweight = np.zeros((weight.shape[0], in_channels, weight.shape[2]),
                           dtype=weight.dtype)
        for i in range(weight.shape[0]):
            for j in range(in_channels // groups):
                nweight[i, i * (in_channels // groups) + j, :] = weight[i, j, :]
        weight = nweight

    output = np.tensordot(view, weight, axes=((1, 2), (1, 2))).transpose(1, 0)

    # Apply bias
    if bias is not None:
        for k in range(out_channels):
            output[k] += bias[k]

    assert output.shape == tuple(output_size[:2]), \
        f'Shape mismatch: NumPy result {output.shape} vs expected {tuple(output_size[:2])}.'

    return output


def convtranspose1d(
        data,
        weight,
        bias,
        input_size,
        output_size,
        kernel_size,
        stride,
        pad,
        dilation,
        fractional_stride,
        output_pad,
        groups=1,
) -> ArrayLike:
    """
    Compute a transposed 1D convolution.
    """

    return conv1d(
        data,
        weight,
        bias,
        input_size,
        output_size,
        kernel_size,
        stride,
        dilation * (kernel_size - 1) - pad,
        dilation,
        fractional_stride=fractional_stride,
        output_pad=output_pad,
        groups=groups,
    )


def linear(
        layer,
        data,
        weight,
        bias,
        in_features,
        out_features,
) -> ArrayLike:
    """
    Compute a fully connected layer.
    """
    output = np.empty(out_features, dtype=np.int64)

    for w in range(out_features):
        val = np.int64(0)
        for n in range(in_features):
            val += data[n] * weight[w][n]
            stats.account(
                layer,
                "true_sw_macc",
                1,
            )
            debug_print(
                f'w={w}, n={n}, weight={weight[w][n]}, data={data[n]} '
                f'-> accumulator = {val} '
            )
        if bias is not None:
            val += bias[w]
            debug_print(f'+bias {bias[w]} --> output[{w}] = {val}')
        output[w] = val

    return output

def pool2d(
        data,
        input_size,
        output_size,
        pool,
        stride,
        average,
        dilation=(1, 1),
        floor=True,
):
    """
    Compute 2D Pooling (Average or Max)
    """
    assert data.shape == tuple(input_size)

    if state.debug:
        # Slow using pure Python
        ref = np.empty(shape=output_size, dtype=np.int64)

        for c in range(input_size[0]):
            for row in range(0, output_size[1]*stride[0], stride[0]):
                for col in range(0, output_size[2]*stride[1], stride[1]):
                    if average:
                        avg = np.mean(data[c][row:row+pool[0]*dilation[0]:dilation[0],
                                              col:col+pool[1]*dilation[1]:dilation[1]])
                        if floor:
                            if avg < 0.:
                                val = np.ceil(avg).astype(np.int64).clip(min=-128, max=127)
                            else:
                                val = np.floor(avg).astype(np.int64).clip(min=-128, max=127)
                        else:
                            if avg < 0.:
                                val = np.ceil(avg - 0.5).astype(np.int64).clip(min=-128, max=127)
                            else:
                                val = np.floor(avg + 0.5).astype(np.int64).clip(min=-128, max=127)
                    else:
                        val = np.amax(data[c][row:row+pool[0]*dilation[0]:dilation[0],
                                              col:col+pool[1]*dilation[1]:dilation[1]])
                    ref[c][row//stride[0]][col//stride[1]] = val

    # Fast computation using NumPy
    data_pad = data[
        :,
        :(data.shape[1] - pool[0] + dilation[0] - 1) // stride[0] * stride[0] + pool[0],
        :(data.shape[2] - pool[1] + dilation[1] - 1) // stride[1] * stride[1] + pool[1],
        ...
    ]
    h, w = data_pad.strides[1:]

    view = as_strided(data_pad,
                      shape=(data_pad.shape[0],
                             1 + (data_pad.shape[1] - pool[0] - dilation[0] + 1) // stride[0],
                             1 + (data_pad.shape[2] - pool[1] - dilation[1] + 1) // stride[1],
                             pool[0], pool[1]),
                      strides=(data_pad.strides[0], stride[0] * h,
                               stride[1] * w, h * dilation[0], w * dilation[1]),
                      writeable=False)

    if average:
        if floor:
            pooled = np.nanmean(view, dtype=np.int64, axis=(3, 4))
        else:
            pooled = np.mean(view, axis=(3, 4))
            pooled = np.ceil(pooled - 0.5, where=pooled < 0., out=pooled)
            pooled = np.floor(pooled + 0.5, where=pooled >= 0., out=pooled) \
                .astype(np.int64).clip(min=-128, max=127)
    else:
        pooled = np.nanmax(view, axis=(3, 4))

    if state.debug:
        match = (ref == pooled).all()
        if not match:
            eprint('NumPy <-> Python mismatch in compute.pool2d')

    assert pooled.shape == tuple(output_size), f'shape mismatch {pooled.shape} vs {output_size}'

    return pooled


def pool1d(
        data,
        input_size,
        output_size,
        pool,
        stride,
        average,
        dilation=1,
        floor=True,  # pylint: disable=unused-argument
) -> ArrayLike:
    """
    Compute 1D Pooling (Average or Max)
    """
    assert data.shape == tuple(input_size)

    pooled = np.empty(shape=output_size, dtype=np.int64)
    for c in range(input_size[0]):
        for x in range(0, output_size[1]*stride, stride):
            if average:
                avg = np.average(data[c][x:x+pool*dilation:dilation])
                if avg < 0:
                    val = np.ceil(avg).astype(np.int64).clip(min=-128, max=127)
                else:
                    val = np.floor(avg).astype(np.int64).clip(min=-128, max=127)
            else:
                val = np.amax(data[c][x:x+pool])
            pooled[c][x//stride] = val

    return pooled


def eltwise(
        operator,
        data,
        input_size,
) -> ArrayLike:
    """
    Compute element-wise operation.
    """
    assert data[0].shape == tuple(input_size)
    operands = len(data)

    output = data[0]
    for i in range(1, operands):
        if operator == op.ELTWISE_ADD:
            output = np.add(output, data[i])
        elif operator == op.ELTWISE_MUL:
            output = np.multiply(output, data[i])
        elif operator == op.ELTWISE_OR:
            output = np.bitwise_or(output, data[i])
        elif operator == op.ELTWISE_SUB:
            output = np.subtract(output, data[i])
        elif operator == op.ELTWISE_XOR:
            output = np.bitwise_xor(output, data[i])
        else:
            print(f"Unknown operator `{op.string(operator)}`")
            raise NotImplementedError

    assert output.shape == tuple(input_size)
    return output

########################################################################################################################


def feed_forward(data, w1, w2, b1=None, b2=None):
    """
    Compute feed-forward network.
    Args:
        data: Input tensor
        w1: First layer weights
        w2: Second layer weights
        b1: First layer bias
        b2: Second layer bias
    Returns:
        Output tensor
    """
    h = np.dot(data, w1)
    if b1 is not None:
        h += b1
    h = np.maximum(0, h)  # ReLU
    
    out = np.dot(h, w2)
    if b2 is not None:
        out += b2
    return out

def residual(data, residual_data):
    """
    Compute residual connection.
    Args:
        data: Main branch output
        residual_data: Residual branch output
    Returns:
        Sum of main and residual branches
    """
    return data + residual_data

def mhsa(
        layer,
        input_size,         # (seq , dim)
        kernel, bias,
        data,
        output_width=8,
        num_heads=1,
        seq_length=None,    # unused but kept for signature
):
    d_model   = input_size[1]
    head_dim  = d_model // num_heads

    # split weight matrix into Q,K,V,O parts (just like checkpoint)
    w_q = kernel[0*d_model:1*d_model, :]
    w_k = kernel[1*d_model:2*d_model, :]
    w_v = kernel[2*d_model:3*d_model, :]
    w_o = kernel[3*d_model:4*d_model, :]

    b_q = bias[0*d_model:1*d_model] if bias is not None else None
    b_k = bias[1*d_model:2*d_model] if bias is not None else None
    b_v = bias[2*d_model:3*d_model] if bias is not None else None
    b_o = bias[3*d_model:4*d_model] if bias is not None else None

    # Projections (seq , dim)
    q = data @ w_q.T;  k = data @ w_k.T;  v = data @ w_v.T
    if b_q is not None:  q += b_q
    if b_k is not None:  k += b_k
    if b_v is not None:  v += b_v

    # “attention” = mean of the heads (no soft-max)
    attn = (q + k + v) / 3

    # output projection
    out = attn @ w_o.T
    if b_o is not None:
        out += b_o

    if output_width == 8:
        out = np.clip(out, -128, 127)
    return out, out.shape

def layernorm(
        layer,             # layer index
        input_size,        # (seq_len, d_model)
        kernel,            # gamma, shape (d_model,)
        bias,              # beta, shape (d_model,)
        data,
        output_width=8,
):
    """
    Replaces LayerNorm with hardware-friendly affine scaling:
    y = x * gamma + beta
    using a Linear implementation.
    Keeps per-channel scaling and shifting.
    Loses normalization, so input dynamic ranges are unadjusted - potential accuracy/stability drop.
    """

    d_model, seq_len = input_size

    kernel_squeezed = np.squeeze(kernel)  # shape: (64,)
    weight = np.diag(kernel_squeezed)  

    out_list = []
    data_reshaped = data.reshape(seq_len, d_model)

    for i in range(seq_len):
        token = data_reshaped[i]  # shape: (d_model,)
        out_token = linear(
            layer,
            token,
            weight,
            bias,
            in_features=d_model,
            out_features=d_model
        )
        out_list.append(out_token)

    out = np.stack(out_list, axis=0)  # shape: (seq_len, d_model)

    if output_width == 8:
        out = np.clip(out, -128, 127)

    return out, out.shape


def feed_forward_layer(
        layer,
        input_size,            # (seq , dim)
        kernel, bias,
        data,
        output_width=8,
):
    d_model = input_size[1]
    d_ff    = kernel.shape[0] // 2

    w1 = kernel[0:d_ff, :]
    w2 = kernel[d_ff:, 0:d_ff]

    b1 = bias[0:d_ff] if bias is not None else None
    b2 = bias[d_ff:]  if bias is not None else None

    h = data @ w1.T
    if b1 is not None:
        h += b1
    h = np.maximum(0, h)        # ReLU

    out = h @ w2.T
    if b2 is not None:
        out += b2

    if output_width == 8:
        out = np.clip(out, -128, 127)
    return out, out.shape

def residual_layer(
        layer,
        input_size,
        data,
        output_width=8,
):
    """
    Compute residual connection layer.
    """
    output = residual(data[0], data[1])
    
    if output_width == 8:
        output = np.clip(output, -128, 127)
    
    return output, output.shape

def patch_embed_layer(
        layer: int,
        data: np.ndarray,
        input_size,
        kernels,
        biases,
        patch_size: int = 3,
        cls_token: Optional[np.ndarray] = None,
        pos_embed: Optional[np.ndarray] = None
):
    # ---------------- first conv 1→d_model -----------------
    x = conv2d(
        data,
        kernels[0],
        biases[0],
        input_size=input_size,
        output_size=(kernels[0].shape[0], input_size[1], input_size[2]),
        kernel_size=(3, 3),
        stride=(1, 1),
        pad=(1, 1),
        dilation=(1, 1),
        fractional_stride=(1, 1),
        output_pad=(0, 0),
    )
    # x = np.maximum(0, x)  # ReLU

    # ---------------- second conv d_model→d_model ----------
    x = conv2d(
        x,
        kernels[1],
        biases[1],
        input_size=x.shape,
        output_size=x.shape,           # stride 1, same spatial size
        kernel_size=(3, 3),
        stride=(1, 1),
        pad=(1, 1),
        dilation=(1, 1),
        fractional_stride=(1, 1),
        output_pad=(0, 0),
    )
    # x = np.maximum(0, x)

    # ---------------- third conv  stride = patch_size ------
    out_h = (x.shape[1] + 2*1 - 3) // patch_size + 1   # pad=1, k=3
    out_w = (x.shape[2] + 2*1 - 3) // patch_size + 1
    x = conv2d(
        x,
        kernels[2],
        biases[2],
        input_size=x.shape,
        output_size=(kernels[2].shape[0], out_h, out_w),
        kernel_size=(3, 3),
        stride=(patch_size, patch_size),
        pad=(1, 1),
        dilation=(1, 1),
        fractional_stride=(1, 1),
        output_pad=(0, 0),
    )

    # --------------- flatten to (tokens, d_model) ----------
    d_model = x.shape[0]
    tokens  = x.reshape(d_model, -1).T            # (num_patches, d_model)

    # --------------- prepend CLS token ---------------------
    if cls_token is not None:
        cls_tok = cls_token.reshape(1, d_model)
        tokens  = np.vstack((cls_tok, tokens))

    # --------------- add positional embedding --------------
    if pos_embed is not None:
        tokens += pos_embed[:tokens.shape[0], :]

    # return in (sequence, dim) order expected by later layers
    return tokens, tokens.shape

def gemm_approx(data, weight, bias=None):
    # collapse any (1,1) spatial dims on weight → shape (C_out, C_in)
    if weight.ndim == 4:
        weight2d = weight[:, :, 0, 0]
    else:
        weight2d = weight
    
    C_in, H, W = data.shape
    C_out, C_in_w = weight2d.shape
    assert C_in_w == C_in, f"weight has {C_in_w} in-channels but data has {C_in}"
    
    out = np.tensordot(weight2d, data, axes=([1], [0]))  # → (C_out, H, W)
    
    if bias is not None:
        out += bias[:, None, None]
    return out

################################################################################################