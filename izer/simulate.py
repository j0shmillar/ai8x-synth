###################################################################################################
# Copyright (C) 2019-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Simulate a single CNN layer
"""
import os

import numpy as np

from . import op, state, stats
from . import tornadocnn as tc
from .compute import conv1d, conv2d, convtranspose2d, eltwise, linear, pool1d, pool2d, layernorm, mhsa
from .names import layer_str


def print_data(
        verbose_data,
        header,
        data,
        input_size,
        expand,
        expand_thresh,
):
    """
    Print `data` of dimensions `input_size` with `expand` and `expand_thresh`,
    prefixed by `header`.
    """
    int8_format = '{0:4}' if np.any(data < 0) else '{0:3}'

    print(header, end='')
    if verbose_data:
        print(':')
        with np.printoptions(formatter={'int': int8_format.format}):
            if len(input_size) == 3 and input_size[1] == input_size[2] == 1:
                for i in range(0, input_size[0], expand_thresh):
                    last = min(i + expand_thresh, input_size[0])
                    if last - 1 > i:
                        print(f'Channels #{i} to #{last-1}', end='')
                    else:
                        print(f'Channel #{i}', end='')
                    if expand and expand > 1:
                        print(f' (expansion: {(i // expand_thresh) + 1} of {expand})')
                    else:
                        print('')
                    print(np.squeeze(data[i:last]))
            else:
                for i in range(input_size[0]):
                    print(f'Channel #{i}', end='')
                    if expand and expand > 1:
                        print(f' (expansion: {(i // expand_thresh) + 1} of {expand})')
                    else:
                        print('')
                    print(data[i])
    print('')


def print_data1d(
        verbose_data,
        header,
        data,
        step=16,
):
    """
    Print 1-dimensional `data` `step`-elements at a time, prefixed by `header`.
    This function is intended for bias data.
    """
    size = len(data) if data is not None else 0

    if verbose_data:
        if size <= step:
            print(f'{header}:', data)
        else:
            int8_format = '{0:4}' if np.any(data < 0) else '{0:3}'

            print(f'{header}:')
            with np.printoptions(formatter={'int': int8_format.format}):
                for i in range(0, size, step):
                    last = min(i + step, size)
                    if last - 1 > i:
                        print(f'Output channels #{i} to #{last-1}:')
                    else:
                        print(f'Output channel #{i}"')
                    print(np.squeeze(data[i:last]))
        print('')
    elif size > 0:
        print(f"\n{header} SIZE: {size}")
    else:
        print('')


def conv2d_layer(
        layer,
        input_size,
        kernel_size,
        output_shift,
        output_channels,
        padding,
        dilation,
        stride,
        activation,
        kernel,
        bias,
        data,
        bits=8,
        output_width=8,
        groups=1,
        bypass=False,
        datafile=None,
):
    """
    Perform 2D convolution for one layer.
    """
    verbose_data = state.verbose_all or state.output_layer[layer]

    print(f"conv data shape: {data.shape}")

    if state.verbose:
        print(f"{kernel_size[0]}x{kernel_size[1]} KERNEL(S)", end='')
        if bypass:
            print(' (BYPASS)')
        if state.verbose_all and not bypass:
            print(":")
            with np.printoptions(formatter={'int': state.kernel_format.format}):
                for i in range(output_channels):
                    if kernel_size[0] == kernel_size[1] == 1:
                        print(f'Output channel #{i}')
                        print(np.squeeze(kernel[i]))
                    else:
                        if kernel[i].shape[0] < 8:
                            print(f'Output channel #{i}')
                            print(kernel[i])
                        else:
                            for j in range(0, kernel[i].shape[0], 8):
                                print(f'Output channel #{i} (input channels {j}-'
                                      f'{min(kernel[i].shape[0], j+8) - 1})')
                                print(kernel[i][j:j+8])
        print_data1d(state.verbose_all, "BIAS", bias)

    out_size = [output_channels,
                (input_size[1] - dilation[0] * (kernel_size[0] - 1) - 1 +
                 2 * padding[0]) // stride[0] + 1,
                (input_size[2] - dilation[1] * (kernel_size[1] - 1) - 1 +
                 2 * padding[1]) // stride[1] + 1]

    if bias is not None:
        bias = bias * tc.dev.BIAS_DIV

    out_buf = conv2d(
        data=data,
        weight=kernel,
        bias=bias,
        input_size=input_size,
        output_size=out_size,
        kernel_size=kernel_size,
        stride=stride,
        pad=padding,
        dilation=dilation,
        fractional_stride=[1, 1],
        output_pad=[0, 0],
        groups=groups,
    )

    if datafile is not None:
        np.save(datafile, out_buf, allow_pickle=False, fix_imports=False)

    if state.verbose and verbose_data:
        print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} FULL-RES OUTPUT:")
        if out_size[1] == out_size[2] == 1:
            print(np.squeeze(out_buf))
        else:
            print(out_buf)
        print('')

    stats.account(
        layer,
        "macc",
        (input_size[0] // groups) * kernel_size[0] * kernel_size[1] * out_size[0]
        * out_size[1] * out_size[2],
    )

    if output_width != 32:
        out_buf = np.floor(0.5 + out_buf / (128 / 2.0**output_shift)).astype(np.int64). \
            clip(-(2**(bits-1)), 2**(bits-1)-1)

        if state.verbose and verbose_data:
            print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} OUTPUT "
                  f"{'BEFORE ACTIVATION' if activation is not None else '(NO ACTIVATION)'}:")
            if out_size[1] == out_size[2] == 1:
                print(np.squeeze(out_buf))
            else:
                print(out_buf)
            print('')

    if activation is not None:
        if activation == op.ACT_RELU:
            np.clip(out_buf, 0, 2**(bits-1)-1, out_buf)
        elif activation == op.ACT_ABS:
            out_buf = np.abs(out_buf).clip(0, 2**(bits-1)-1)

        if state.verbose and verbose_data:
            print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} ACTIVATED OUTPUT"
                  f" ({op.act_string(activation).upper()}):")
            if out_size[1] == out_size[2] == 1:
                print(np.squeeze(out_buf))
            else:
                print(out_buf)
            print('')

        stats.account(
            layer,
            "comp",
            out_size[0] * out_size[1] * out_size[2],
        )

    if state.verbose and not verbose_data:
        print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} OUTPUT"
              f" ({op.act_string(activation).upper()})\n")

    return out_buf, out_size


def convtranspose2d_layer(
        layer,
        input_size,
        kernel_size,
        output_shift,
        output_channels,
        padding,
        dilation,
        fractional_stride,
        output_padding,
        activation,
        kernel,
        bias,
        data,
        bits=8,
        output_width=8,
        groups=1,
        bypass=False,
        datafile=None,
):
    """
    Perform a fractionally strided 2D convolution for one layer.
    """
    verbose_data = state.verbose_all or state.output_layer[layer]

    if state.verbose:
        print(f"{kernel_size[0]}x{kernel_size[1]} KERNEL(S)", end='')
        if bypass:
            print(' (BYPASS)')
        if state.verbose_all and not bypass:
            print(':')
            with np.printoptions(formatter={'int': state.kernel_format.format}):
                for i in range(output_channels):
                    if kernel_size[0] == kernel_size[1] == 1:
                        print(f'Output channel #{i}')
                        print(np.squeeze(kernel[i]))
                    else:
                        if kernel[i].shape[0] < 8:
                            print(f'Output channel #{i}')
                            print(kernel[i])
                        else:
                            for j in range(0, kernel[i].shape[0], 8):
                                print(f'Output channel #{i} (input channels {j}-'
                                      f'{min(kernel[i].shape[0], j+8) - 1})')
                                print(kernel[i][j:j+8])

        print_data1d(state.verbose_all, "BIAS", bias)

    out_size = [output_channels,
                (input_size[1] - 1) * fractional_stride[0] - 2 * padding[0]
                + dilation[0] * (kernel_size[0] - 1)
                + output_padding[0] + 1,
                (input_size[2] - 1) * fractional_stride[1] - 2 * padding[1]
                + dilation[1] * (kernel_size[1] - 1)
                + output_padding[1] + 1]

    if bias is not None:
        bias = bias * tc.dev.BIAS_DIV

    out_buf = convtranspose2d(
        data=data,
        weight=kernel,
        bias=bias,
        input_size=input_size,
        output_size=out_size,
        kernel_size=kernel_size,
        stride=[1, 1],
        pad=padding,
        dilation=dilation,
        fractional_stride=fractional_stride,
        output_pad=output_padding,
        groups=groups,
    )

    if datafile is not None:
        np.save(datafile, out_buf, allow_pickle=False, fix_imports=False)

    if state.verbose and verbose_data:
        print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} FULL-RES OUTPUT:")
        if out_size[1] == out_size[2] == 1:
            print(np.squeeze(out_buf))
        else:
            print(out_buf)
        print('')

    stats.account(
        layer,
        "macc",
        (input_size[0] // groups) * kernel_size[0] * kernel_size[1] * out_size[0]
        * out_size[1] * out_size[2],
    )

    if output_width != 32:
        out_buf = np.floor(0.5 + out_buf / (128 / 2.0**output_shift)).astype(np.int64). \
            clip(-(2**(bits-1)), 2**(bits-1)-1)

        if state.verbose and verbose_data:
            print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} OUTPUT "
                  f"{'BEFORE ACTIVATION' if activation is not None else '(NO ACTIVATION)'}:")
            if out_size[1] == out_size[2] == 1:
                print(np.squeeze(out_buf))
            else:
                print(out_buf)
            print('')

    if activation is not None:
        if activation == op.ACT_RELU:
            np.clip(out_buf, 0, 2**(bits-1)-1, out_buf)
        elif activation == op.ACT_ABS:
            out_buf = np.abs(out_buf).clip(0, 2**(bits-1)-1)

        if state.verbose and verbose_data:
            print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} ACTIVATED OUTPUT"
                  f" ({op.act_string(activation).upper()}):")
            if out_size[1] == out_size[2] == 1:
                print(np.squeeze(out_buf))
            else:
                print(out_buf)
            print('')

        stats.account(
            layer,
            "comp",
            out_size[0] * out_size[1] * out_size[2],
        )

    if state.verbose and not verbose_data:
        print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} OUTPUT"
              f" ({op.act_string(activation).upper()})\n")

    return out_buf, out_size


def conv1d_layer(
        layer,
        input_size,
        kernel_size,
        output_shift,
        output_channels,
        padding,
        dilation,
        stride,
        activation,
        kernel,
        bias,
        data,
        bits=8,
        output_width=8,
        groups=1,
        bypass=False,
        datafile=None,
):
    """
    Perform 1D convolution for one layer.
    """
    verbose_data = state.verbose_all or state.output_layer[layer]

    if state.verbose:
        print(f"KERNEL SIZE {kernel_size}", end='')
        if bypass:
            print(' (BYPASS)')
        if state.verbose_all and not bypass:
            print(':')
            with np.printoptions(formatter={'int': state.kernel_format.format}):
                print(kernel)
        print_data1d(state.verbose_all, "BIAS", bias)

    out_size = [output_channels,
                (input_size[1] - dilation * (kernel_size - 1) - 1 +
                 2 * padding) // stride + 1,
                1]

    if bias is not None:
        bias = bias * tc.dev.BIAS_DIV

    out_buf = conv1d(
        data=data,
        weight=kernel,
        bias=bias,
        input_size=input_size,
        output_size=out_size,
        kernel_size=kernel_size,
        stride=stride,
        pad=padding,
        dilation=dilation,
        fractional_stride=1,
        output_pad=0,
        groups=groups,
    )[:, :, np.newaxis]

    if datafile is not None:
        np.save(datafile, out_buf, allow_pickle=False, fix_imports=False)

    if state.verbose and verbose_data:
        print(f"{out_size[0]}x{out_size[1]} FULL-RES OUTPUT:")
        print(out_buf.squeeze(axis=-1))
        print('')

    stats.account(
        layer,
        "macc",
        (input_size[0] // groups) * kernel_size * out_size[0] * out_size[1],
    )

    if output_width != 32:
        out_buf = np.floor(0.5 + out_buf / (128 / 2.0**output_shift)).astype(np.int64). \
            clip(-(2**(bits-1)), 2**(bits-1)-1)

        if state.verbose and verbose_data:
            print(f"{out_size[0]}x{out_size[1]} OUTPUT "
                  f"{'BEFORE ACTIVATION' if activation is not None else '(NO ACTIVATION)'}:")
            print(out_buf.squeeze(axis=-1))
            print('')

    if activation is not None:
        if activation == op.ACT_RELU:
            np.clip(out_buf, 0, 2**(bits-1)-1, out_buf)
        elif activation == op.ACT_ABS:
            out_buf = np.abs(out_buf).clip(0, 2**(bits-1)-1)

        if state.verbose and verbose_data:
            print(f"{out_size[0]}x{out_size[1]} ACTIVATED OUTPUT"
                  f" ({op.act_string(activation).upper()}):")
            print(out_buf.squeeze(axis=-1))
            print('')

        stats.account(
            layer,
            "comp",
            out_size[0] * out_size[1],
        )

    if state.verbose and not verbose_data:
        print(f"{out_size[0]}x{out_size[1]} OUTPUT"
              f" ({op.act_string(activation).upper()})\n")

    return out_buf, out_size


# def linear_layer(
#         layer,
#         activation,
#         weight,
#         bias,
#         data,
#         bits=16,
# ):
#     """
#     Perform one software linear layer.
#     """
#     verbose_data = state.verbose_all or state.output_layer[layer]
#     verbose_input = state.verbose_all or layer == state.start_layer \
#         or state.in_sequences[layer] is not None and -1 in state.in_sequences[layer]

#     in_features = data.shape[0]
#     out_features = weight.shape[0]

#     if state.verbose_all or verbose_input:
#         print("CLASSIFICATION LAYER (LINEAR)...\n")
#         print(f"INPUT DATA (size {in_features})", end='')
#         if verbose_input:
#             print(':')
#             print(data)
#         print('')

#     if state.verbose_all:
#         print(f"WEIGHTS (size {in_features * out_features})", end='')
#         print(':')
#         print(weight)
#         print_data1d(state.verbose_all, "BIAS", bias)

#     out_buf = linear(
#         layer=layer,
#         data=data,
#         weight=weight,
#         bias=bias,
#         in_features=in_features,
#         out_features=out_features,
#     )
#     out_buf = np.floor(0.5 + out_buf / 128).astype(np.int64). \
#         clip(-(2**(bits-1)), 2**(bits-1)-1)

#     if state.verbose and verbose_data:
#         print(f"OUTPUT (size {out_features}):")
#         print(out_buf)
#         print('')

#     stats.account(
#         layer,
#         "sw_macc",
#         in_features * out_features,
#     )

#     if activation is not None:
#         if activation == op.ACT_RELU:
#             np.clip(out_buf, 0, 2**(bits-1)-1, out_buf)
#         elif activation == op.ACT_ABS:
#             out_buf = np.abs(out_buf).clip(0, 2**(bits-1)-1)

#         if state.verbose and verbose_data:
#             print(f"ACTIVATED OUTPUT (size {out_features})"
#                   f" ({op.act_string(activation).upper()}):")
#             print(out_buf)
#             print('')

#         stats.account(
#             layer,
#             "sw_comp",
#             out_features,
#         )

#     if state.verbose and not verbose_data:
#         print(f"OUTPUT (size {out_features})"
#               f" ({op.act_string(activation).upper()})\n")

#     return out_buf, out_features

def linear_layer(
        layer,
        activation,
        weight,
        bias,
        data,
        bits=8,
):
    """
    Fully hardware-accelerated linear layer using conv1d ops on MAX78000.
    Replaces GEMV with conv1d while preserving activation handling and stats accounting.
    """

    verbose_data = state.verbose_all or state.output_layer[layer]

    in_features = data.shape[0]
    out_features = weight.shape[0]

    # Reshape input for hardware conv1d:
    #   data: (in_channels, width, 1)
    data_hw = data.reshape(in_features, 1, 1)

    # Reshape weights:
    #   weight: (out_channels, in_channels, kernel_size=1)
    w_hw = weight.reshape(out_features, in_features, 1)

    # Bias:
    if bias is not None:
        bias_hw = bias.astype(np.int64)
    else:
        bias_hw = np.zeros(out_features, dtype=np.int64)

    # Use conv1d_layer to compute:
    out_buf, out_shape_hw = conv1d_layer(
        layer=layer,
        input_size=data_hw.shape,
        kernel_size=1,
        output_shift=0,              # Use zero shift, as no scaling here
        output_channels=out_features,
        padding=0,
        dilation=1,
        stride=1,
        activation=activation,
        kernel=w_hw,
        bias=bias_hw,
        data=data_hw,
        bits=bits,
        output_width=bits,
        groups=1,
        bypass=False,
        datafile=None,
    )

    output = out_buf[:, 0, 0]

    stats.account(
        layer,
        "macc",
        in_features * out_features
    )

    if state.verbose and verbose_data:
        print(f"LINEAR OUTPUT (size {out_features}):")
        print(output)
        print('')

    return output, out_features

def passthrough_layer(
        layer,  # pylint: disable=unused-argument
        input_size,
        data,
        datafile=None,
):
    """
    2D passthrough for one layer.
    """
    if datafile is not None:
        np.save(datafile, np.empty((0)), allow_pickle=False, fix_imports=False)

    return data, input_size


def eltwise_layer(
        operator,
        layer,
        input_size,
        output_shift,
        data,
        output_width=8,
        operands=1,
):
    """
    Element-wise operators for one layer.
    """
    verbose_data = state.verbose_all or state.output_layer[layer]

    bits = 8
    assert operands == len(data)

    if state.verbose:
        print(f"{operands}-OPERAND {op.string(operator, elt=True).upper()}:\n")

    out_buf = eltwise(
        operator=operator,
        data=data,
        input_size=input_size,
    )

    if state.verbose and verbose_data:
        print(f"{input_size[0]}x{input_size[1]}x{input_size[2]} FULL-RES OUTPUT:")
        if input_size[1] == input_size[2] == 1:
            print(np.squeeze(out_buf))
        else:
            print(out_buf)
        print('')

    if operator in [op.ELTWISE_ADD, op.ELTWISE_SUB]:
        stats.account(
            layer,
            "add",
            (operands - 1) * out_buf.size,
        )
    elif operator == op.ELTWISE_MUL:
        stats.account(
            layer,
            "mul",
            (operands - 1) * out_buf.size,
        )
    elif operator in [op.ELTWISE_OR, op.ELTWISE_XOR]:
        stats.account(
            layer,
            "bitwise",
            (operands - 1) * out_buf.size,
        )

    if output_width != 32:
        if operator == op.ELTWISE_MUL:
            out_buf = np.floor(0.5 + out_buf / (128 / 2.0**output_shift)).astype(np.int64). \
                clip(-(2**(bits-1)), 2**(bits-1)-1)
        else:
            np.clip(out_buf, -(2**(bits-1)), 2**(bits-1)-1, out_buf)

        if state.verbose and verbose_data:
            print(f"{input_size[0]}x{input_size[1]}x{input_size[2]} OUTPUT:")
            if input_size[1] == input_size[2] == 1:
                print(np.squeeze(out_buf))
            else:
                print(out_buf)
            print('')

    if state.verbose and not verbose_data:
        print(f"{input_size[0]}x{input_size[1]}x{input_size[2]} OUTPUT")

    return out_buf, input_size

# TODO - modified, so check
########################################################################################################################

def pooling_layer(
        layer,
        input_size,
        pool,
        pool_stride,
        pool_average,
        data,
        expand=None,
        expand_thresh=None,
        operation=None,
        operands=1,
        rounding=False,
        debug_data=None,
        dilation=(1, 1),
):
    """
    Perform pooling for one layer.
    """
    # Always apply stride
    if operation == op.CONV1D:
        pooled_size = [input_size[0],
                       (input_size[1] + pool_stride[0] - pool[0]
                        - dilation[0] + 1) // pool_stride[0]]
    elif data.ndim == 3:
        pooled_size = [input_size[0],
                       (input_size[1] + pool_stride[0] - pool[0]
                        - dilation[0] + 1) // pool_stride[0]]
    elif data.ndim == 4:
        pooled_size = [input_size[0],
                       (input_size[1] + pool_stride[0]
                        - pool[0] - dilation[0] + 1) // pool_stride[0],
                       (input_size[2] + pool_stride[1]
                        - pool[1] - dilation[1] + 1) // pool_stride[1]]

    # Actual pooling operation?
    if pool[0] > 1 or pool[1] > 1:
        if operation != op.CONV1D:
            pooled = np.empty((operands, pooled_size[0], pooled_size[1], pooled_size[2]),
                              dtype=np.int64)
            for i in range(operands):
                if debug_data is not None:
                    for j in range(input_size[0]):
                        np.savetxt(os.path.join(debug_data, f"unpooled-{i}-L{layer}-ch{j}.csv"),
                                   data[i][j, :, :], delimiter=",")
                pooled[i] = pool2d(
                    data[i],
                    input_size,
                    pooled_size,
                    pool,
                    pool_stride,
                    pool_average,
                    dilation=dilation,
                    floor=not rounding,
                )
                if state.verbose:
                    if dilation[0] > 1 or dilation[1] > 1:
                        dilation_str = f", DILATION {dilation[0]}/{dilation[1]}"
                    else:
                        dilation_str = ''
                    print_data(
                        state.verbose_all,
                        f"{'AVERAGE' if pool_average else 'MAX'} "
                        f"POOL {pool[0]}x{pool[1]} WITH STRIDE {pool_stride[0]}/{pool_stride[1]}"
                        + dilation_str +
                        f" {input_size} -> {pooled_size}"
                        + (f", POOLED DATA {i}" if operands > 1 else ""),
                        pooled[i],
                        pooled_size,
                        expand,
                        expand_thresh,
                    )
                if debug_data is not None:
                    for j in range(pooled_size[0]):
                        np.savetxt(os.path.join(debug_data, f"pooled-{i}-L{layer}-ch{j}.csv"),
                                   pooled[i][j, :, :], delimiter=",")

            st = pool[0] * pool[1] * pooled_size[0] * pooled_size[1] * pooled_size[2] * operands
            if pool_average:
                stats.account(
                    layer,
                    "add",
                    st,
                )
            else:
                stats.account(
                    layer,
                    "comp",
                    st,
                )
        else:
            pooled = pool1d(
                data[0],
                input_size,
                pooled_size,
                pool[0],
                pool_stride[0],
                pool_average,
                dilation=dilation[0],
                floor=not rounding,
            )
            if state.verbose:
                print(f"{'AVERAGE' if pool_average else 'MAX'} "
                      f"POOL {pool[0]} WITH STRIDE {pool_stride[0]} ", end='')
                if dilation[0] > 1:
                    print(f", DILATION {dilation[0]} ", end='')
                print(f"{input_size} -> {pooled_size}", end='')
                if state.verbose_all:
                    print(':')
                    print(pooled)
                print('')

            if pool_average:
                stats.account(
                    layer,
                    "add",
                    pool[0] * pooled_size[0] * pooled_size[1],
                )
            else:
                stats.account(
                    layer,
                    "comp",
                    pool[0] * pooled_size[0] * pooled_size[1],
                )

            pooled = np.expand_dims(pooled, axis=0)

    else:
        # Use pool_stride only
        if operation == op.CONV1D:
            pooled = data[:, :, ::pool_stride[0]]
            if pool_stride[0] > 1:
                if state.verbose:
                    print(f"{'AVERAGE' if pool_average else 'MAX'} "
                          f"POOL {pool[0]} WITH STRIDE {pool_stride[0]} "
                          f"{input_size} -> {pooled_size}", end='')
                    if state.verbose_all:
                        print(':')
                        print(pooled)
                    print('')
        elif data.ndim == 3: 
            pooled = data[:, :, ::pool_stride[0]]
            if pool_stride[0] > 1:
                if state.verbose:
                    print(f"{'AVERAGE' if pool_average else 'MAX'} "
                          f"POOL {pool[0]} WITH STRIDE {pool_stride[0]} "
                          f"{input_size} -> {pooled_size}", end='')
                    if state.verbose_all:
                        print(':')
                        print(pooled)
                    print('')
        elif data.ndim == 4:
            pooled = data[:, :, ::pool_stride[0], ::pool_stride[1]]
            if pool_stride[0] > 1 or pool_stride[1] > 1:
                if state.verbose:
                    print(f"{'AVERAGE' if pool_average else 'MAX'} "
                          f"POOL {pool[0]}x{pool[1]} WITH STRIDE {pool_stride[0]}/{pool_stride[1]}"
                          f" {input_size} -> {pooled_size}", end='')
                    if state.verbose_all:
                        print(':')
                        print(pooled)
                    print('')

    return pooled, pooled_size

########################################################################################################################

########################################################################################################################

def layernorm_layer(           # pylint: disable=too-many-arguments
        layer: int,
        input_size,
        kernel: np.ndarray,
        bias:   np.ndarray,
        data:   np.ndarray,
        output_width: int = 8,
):
    """
    2 modes supported:

    1.  Token LN   data shape = (S , D)      weight = (D,)
        each (row) is scaled / shifted element-wise.

    2.  Feature LN data shape = (C , H , W)  weight = (C,)
        every spatial location (h,w) is scaled by the channel-wise gain.
    """
    verbose_data = state.verbose_all or state.output_layer[layer]

    if state.verbose:
        print(f"LAYER {layer_str(layer)} (LAYER-NORM)...\n")
        if state.verbose_all:
            print_data1d(True, "SCALE  (γ)", kernel.reshape(-1))
            print_data1d(True, "OFFSET (β)", None if bias is None else bias.reshape(-1))

    if data.ndim == 2:                         # (S , D)  ------ token LN
        out_buf, out_shape = layernorm(
            layer,
            input_size,
            kernel,
            bias,
            data,
            output_width,
        )

    elif data.ndim == 3:                       # (C , H , W) -- feature-map LN
        C, H, W = data.shape
        assert kernel.size == C, \
            f"LayerNorm: expected {C} scale values, got {kernel.size}"

        scale = kernel.reshape(C, 1, 1).astype(np.int64)
        shift = bias.reshape(C, 1, 1).astype(np.int64) if bias is not None else 0

        out_buf = data * scale + shift        # element-wise affine
        out_shape = out_buf.shape

        # register “true SW MACCs”: one multiply per element
        stats.account(layer, "true_sw_macc", np.prod(out_shape))

    else:
        raise ValueError(f"LayerNorm: unsupported input rank {data.ndim}")

    if output_width == 8:
        np.clip(out_buf, -128, 127, out_buf)

    if state.verbose and verbose_data:
        print(f"OUTPUT {out_shape}:")
        print(out_buf)
        print('')

    if out_shape[0] != 64: # TODO fix; softcode
        out_buf = out_buf.T
        out_shape = out_buf.shape

    return out_buf, out_shape

# TODO; missing Q,K,V LayerNorm
# def mhsa_layer(
#         layer,
#         input_size,      
#         kernels,         
#         biases, 
#         data,
#         output_width=8,
#         num_heads=4,
#         cls_tokens=None,
#         d_model=64,
#         seq_length=82,
# ):
#     """
#     Hardware-accelerated MHSA using per-token linear projections.
#     """

#     print(data.shape)

#     seq_len = seq_length

#     kernels = kernels.squeeze(0)

#     # Extract weights correctly
#     w_q = kernels[:64]
#     w_k = kernels[64:128]
#     w_v = kernels[128:192]
#     w_o = kernels[192:256]

#     if biases is not None:
#         b_q = biases[:64]
#         b_k = biases[64:128]
#         b_v = biases[128:192]
#         b_o = biases[192:256]
#     else:
#         b_q = b_k = b_v = b_o = None

#     # Prepend CLS token
#     data_reshaped = data.reshape(-1, d_model)   # (81, 64)
#     cls_tokens_squeezed = np.squeeze(cls_tokens, axis=0)  # (1, 64)
#     data_with_cls = np.concatenate((cls_tokens_squeezed, data_reshaped), axis=0)  # (82, 64)

#     # x = torch.cat([cls_token, x], dim=1)  # (B, 1 + n_patches, d_model) ?


#     # Prepare outputs
#     q_list, k_list, v_list = [], [], []

#     for i in range(seq_len):
#         token_flat = data_with_cls[i]  # shape: (64,)
#         # Project using hardware-supported linear
#         q_token = linear(layer, token_flat, w_q, b_q, d_model, d_model)
#         k_token = linear(layer, token_flat, w_k, b_k, d_model, d_model)
#         v_token = linear(layer, token_flat, w_v, b_v, d_model, d_model)
#         q_list.append(q_token)
#         k_list.append(k_token)
#         v_list.append(v_token)

#     q = np.stack(q_list, axis=0)  # (82, 64)
#     k = np.stack(k_list, axis=0)
#     v = np.stack(v_list, axis=0)

#     attn = (q + k + v) // 3  # integer-friendly average

#     out_list = []
#     for i in range(seq_len):
#         attn_token = attn[i]  # shape: (64,)
#         out_token = linear(layer, attn_token, w_o, b_o, d_model, d_model)
#         out_list.append(out_token)

#     out = np.stack(out_list, axis=0)  # (82, 64)

#     if output_width == 8:
#         out = np.clip(out, -128, 127)
    
#     out_transposed = out.T
#     return out_transposed, out_transposed.shape

def mhsa_layer(
        layer,
        input_size,      
        kernels,         
        biases, 
        data,
        output_width=8,
        num_heads=4,
        cls_tokens=None,
        d_model=64,
        seq_length=82,
):
    """
    Fully hardware-accelerated MHSA using Conv1D to replace linear projections,
    suitable for MAX78000 hardware.
    """

    # =============================
    # Ensure data shape: (seq_len, d_model)
    if data.ndim == 3 and data.shape[0] == d_model:
        # (d_model, H, W) -> flatten spatial dims
        seq_len = data.shape[1] * data.shape[2]
        data = data.reshape(d_model, seq_len).T  # (seq_len, d_model)
    elif data.ndim == 2 and data.shape[0] == d_model:
        # (d_model, seq_len) -> transpose
        data = data.T
    elif data.ndim == 2 and data.shape[1] == d_model:
        # already (seq_len, d_model)
        pass
    else:
        raise ValueError(f"mhsa_layer: Unexpected data shape {data.shape}, expected (C,H,W), (seq_len,d_model), or (d_model,seq_len)")

    verbose_data = state.verbose_all or state.output_layer[layer]

    seq_len = seq_length

    # Extract and reshape kernels for hardware Conv1D
    kernels = kernels.squeeze(0)  # shape: (256, 64)

    w_q = kernels[:64].reshape(d_model, d_model, 1)    # (out_channels, in_channels, 1)
    w_k = kernels[64:128].reshape(d_model, d_model, 1)
    w_v = kernels[128:192].reshape(d_model, d_model, 1)
    w_o = kernels[192:256].reshape(d_model, d_model, 1)

    if biases is not None:
        b_q = biases[:64]
        b_k = biases[64:128]
        b_v = biases[128:192]
        b_o = biases[192:256]
    else:
        b_q = b_k = b_v = b_o = None

    # Determine if cls_token needs to be prepended
    if cls_tokens is not None:
        expected_seq_len_without_cls = seq_len - 1
    else:
        expected_seq_len_without_cls = seq_len

    if data.shape[0] == expected_seq_len_without_cls:
        # Need to prepend cls_token
        data_reshaped = data.reshape(-1, d_model)             # (seq_len - 1, d_model)
        cls_token_squeezed = np.squeeze(cls_tokens, axis=0)   # (1, d_model)
        data_with_cls = np.concatenate((cls_token_squeezed, data_reshaped), axis=0)  # (seq_len, d_model)
    elif data.shape[0] == seq_len:
        # cls_token already present, do not add again
        data_with_cls = data
    else:
        raise ValueError(f"mhsa_layer: Unexpected data shape {data.shape}, expected {expected_seq_len_without_cls} or {seq_len} tokens")

    # Reshape for conv1d_layer: (d_model, seq_len, 1)
    data_hw = data_with_cls.T[:, :, np.newaxis]

    # Perform Q, K, V projections using Conv1D
    q, _ = conv1d_layer(
        layer,
        input_size=(d_model, seq_len, 1),
        kernel_size=1,
        output_shift=0,
        output_channels=d_model,
        padding=0,
        dilation=1,
        stride=1,
        activation=None,
        kernel=w_q,
        bias=b_q,
        data=data_hw,
        output_width=output_width,
    )

    k, _ = conv1d_layer(
        layer,
        input_size=(d_model, seq_len, 1),
        kernel_size=1,
        output_shift=0,
        output_channels=d_model,
        padding=0,
        dilation=1,
        stride=1,
        activation=None,
        kernel=w_k,
        bias=b_k,
        data=data_hw,
        output_width=output_width,
    )

    v, _ = conv1d_layer(
        layer,
        input_size=(d_model, seq_len, 1),
        kernel_size=1,
        output_shift=0,
        output_channels=d_model,
        padding=0,
        dilation=1,
        stride=1,
        activation=None,
        kernel=w_v,
        bias=b_v,
        data=data_hw,
        output_width=output_width,
    )

    # Compute attn = (q + k + v) // 3 using CPU for simplicity (or add hardware eltwise if desired)
    attn = (q + k + v) // 3

    # Perform output projection using Conv1D
    out, _ = conv1d_layer(
        layer,
        input_size=(d_model, seq_len, 1),
        kernel_size=1,
        output_shift=0,
        output_channels=d_model,
        padding=0,
        dilation=1,
        stride=1,
        activation=None,
        kernel=w_o,
        bias=b_o,
        data=attn,
        output_width=output_width,
    )

    # Squeeze last dim: shape (d_model, seq_len)
    out_squeezed = out.squeeze(-1)

    if output_width == 8:
        out_squeezed = np.clip(out_squeezed, -128, 127)

    if state.verbose and verbose_data:
        print(f"MHSA OUTPUT {out_squeezed.shape}:")
        print(out_squeezed)
        print('')

    # =======================
    # ACCOUNT FOR MACCS
    # =======================
    # Each projection uses:
    # (in_channels // groups) * kernel_size * out_channels * output_width
    # Here:
    # - in_channels = d_model
    # - kernel_size = 1
    # - out_channels = d_model
    # - output_width = seq_len
    #
    # We perform 4 projections: Q, K, V, and out_proj

    maccs_per_projection = d_model * 1 * d_model * seq_len
    stats.account(layer, "macc", 4 * maccs_per_projection)

    # if out_squeezed.ndim == 2:
    #     out_squeezed = out_squeezed.T[:, :, np.newaxis] 

    # =======================
    # return out_squeezed, out_squeezed.shape
    return out_squeezed, out_squeezed.shape

def patch_embed_layer(
        layer: int,
        data: np.ndarray,
        input_size,
        kernels,
        biases,
        patch_size = 3,
        cls_token = None,
        pos_embed = None
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

########################################################################################################################

def show_data(
        layer,
        input_size,
        data,
        expand=None,
        expand_thresh=None,
        operation=None,
        operands=1,
):
    """
    Show input data.
    """
    if state.verbose:
        verbose_input = state.verbose_all or layer == state.start_layer \
            or state.in_sequences[layer] is not None and -1 in state.in_sequences[layer]

        if expand_thresh is None:
            expand_thresh = input_size[0]

        if operation != op.CONV1D:
            if operands == 1:
                op_string = f"LAYER {layer_str(layer)} ({op.string(operation).upper()})...\n"
            else:
                op_string = f"LAYER {layer_str(layer)} ({op.string(operation).upper()}, " \
                            f"{operands} OPERANDS)...\n"
            print(op_string)

# TODO check     
###################################################################################################
            if operands == 1:
                if data.ndim == 3:
                    print_data(verbose_input,
                            f"{data.shape[1]}x{data.shape[2]} INPUT DATA",
                            data[0],
                            [data.shape[1], data.shape[2]],
                            expand,
                            expand_thresh)
                elif data.ndim == 4:
                    print_data(verbose_input,
                            f"{data.shape[1]}x{data.shape[2]}x{data.shape[3]} INPUT DATA",
                            data[0],
                            [data.shape[1], data.shape[2], data.shape[3]],
                            expand,
                            expand_thresh)
            else:
                for i in range(operands):
                    print_data(verbose_input,
                               f"{data.shape[1]}x{data.shape[2]}x{data.shape[3]} INPUT DATA {i}",
                               data[i],
                               [data.shape[1], data.shape[2], data.shape[3]],
                               expand,
                               expand_thresh)
        else:
            print(f"LAYER {layer_str(layer)} ({op.string(operation).upper()})...\n")
            print(f"{input_size[1]}x{input_size[2]} INPUT DATA", end='')
            if verbose_input:
                print(':')
                print(np.squeeze(data))
            print('')
###################################################################################################