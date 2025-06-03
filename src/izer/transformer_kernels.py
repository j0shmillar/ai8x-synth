"""
Hardware-optimized transformer kernels for MAX78000.
These kernels are optimized for the MAX78000's hardware capabilities.
"""
import numpy as np
from typing import Tuple, Optional

from . import op, state
from . import tornadocnn as tc
from .kernels import _INVALID_VALUE, _WORDS_PER_KERNEL

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def attention_kernel(
        data: np.ndarray,
        w_q: np.ndarray,
        w_k: np.ndarray,
        w_v: np.ndarray,
        w_o: np.ndarray,
        b_q: Optional[np.ndarray] = None,
        b_k: Optional[np.ndarray] = None,
        b_v: Optional[np.ndarray] = None,
        b_o: Optional[np.ndarray] = None,
        num_heads: int = 1,
        output_width: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hardware-optimized attention kernel for MAX78000.
    Uses pointwise operations for Q, K, V projections and output projection.
    
    Args:
        data: Input tensor of shape (batch_size, seq_len, d_model)
        w_q: Query weight matrix of shape (d_model, d_model)
        w_k: Key weight matrix of shape (d_model, d_model)
        w_v: Value weight matrix of shape (d_model, d_model)
        w_o: Output weight matrix of shape (d_model, d_model)
        b_q: Query bias vector of shape (d_model,)
        b_k: Key bias vector of shape (d_model,)
        b_v: Value bias vector of shape (d_model,)
        b_o: Output bias vector of shape (d_model,)
        num_heads: Number of attention heads
        output_width: Output bit width (8 or 32)
    
    Returns:
        Tuple of (output tensor, output shape)
    """
    batch_size, seq_len, d_model = data.shape
    head_dim = d_model // num_heads
    
    # Project Q, K, V using pointwise operations
    q = np.matmul(data, w_q.T)
    if b_q is not None:
        q += b_q

    k = np.matmul(data, w_k.T)
    if b_k is not None:
        k += b_k

    v = np.matmul(data, w_v.T)
    if b_v is not None:
        v += b_v

    # Reshape for multi-head attention
    q = q.reshape(batch_size, seq_len, num_heads, head_dim)
    k = k.reshape(batch_size, seq_len, num_heads, head_dim)
    v = v.reshape(batch_size, seq_len, num_heads, head_dim)

    # Transpose for attention
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    # Compute attention scores
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    attn = softmax(scores, axis=-1)
    
    # Apply attention to values
    out = np.matmul(attn, v)
    out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    
    # Final projection
    out = np.matmul(out, w_o.T)
    if b_o is not None:
        out += b_o
    
    # Clip output if using 8-bit precision
    if output_width == 8:
        out = np.clip(out, -128, 127)
    
    return out, out.shape

def layer_norm_kernel(
        data: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
        eps: float = 1e-5,
        output_width: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hardware-optimized layer normalization kernel for MAX78000.
    Uses pointwise operations for mean, variance, scaling, and shifting.
    
    Args:
        data: Input tensor of shape (batch_size, seq_len, d_model)
        weight: Scale parameter of shape (d_model,)
        bias: Shift parameter of shape (d_model,)
        eps: Small constant for numerical stability
        output_width: Output bit width (8 or 32)
    
    Returns:
        Tuple of (output tensor, output shape)
    """
    # Compute mean and variance
    mean = np.mean(data, axis=-1, keepdims=True)
    var = np.var(data, axis=-1, keepdims=True)
    
    # Normalize
    out = (data - mean) / np.sqrt(var + eps)
    
    # Scale and shift
    out = out * weight
    if bias is not None:
        out += bias
    
    # Clip output if using 8-bit precision
    if output_width == 8:
        out = np.clip(out, -128, 127)
    
    return out, out.shape

def feed_forward_kernel(
        data: np.ndarray,
        w1: np.ndarray,
        w2: np.ndarray,
        b1: Optional[np.ndarray] = None,
        b2: Optional[np.ndarray] = None,
        output_width: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hardware-optimized feed-forward network kernel for MAX78000.
    Uses pointwise operations for linear layers and ReLU activation.
    
    Args:
        data: Input tensor of shape (batch_size, seq_len, d_model)
        w1: First weight matrix of shape (d_ff, d_model)
        w2: Second weight matrix of shape (d_model, d_ff)
        b1: First bias vector of shape (d_ff,)
        b2: Second bias vector of shape (d_model,)
        output_width: Output bit width (8 or 32)
    
    Returns:
        Tuple of (output tensor, output shape)
    """
    # First linear layer
    out = np.matmul(data, w1.T)
    if b1 is not None:
        out += b1
    
    # ReLU activation
    out = np.maximum(0, out)
    
    # Second linear layer
    out = np.matmul(out, w2.T)
    if b2 is not None:
        out += b2
    
    # Clip output if using 8-bit precision
    if output_width == 8:
        out = np.clip(out, -128, 127)
    
    return out, out.shape

def positional_encoding_kernel(
        seq_len: int,
        d_model: int,
        output_width: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hardware-optimized positional encoding kernel for MAX78000.
    Pre-computes sine and cosine positional encodings.
    
    Args:
        seq_len: Sequence length
        d_model: Model dimension
        output_width: Output bit width (8 or 32)
    
    Returns:
        Tuple of (positional encoding matrix, shape)
    """
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rads = pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    
    # Apply sine to even indices and cosine to odd indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    # Clip output if using 8-bit precision
    if output_width == 8:
        angle_rads = np.clip(angle_rads, -128, 127)
    
    return angle_rads, angle_rads.shape

def residual_kernel(
        data: np.ndarray,
        residual_data: np.ndarray,
        output_width: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hardware-optimized residual connection kernel for MAX78000.
    Uses element-wise addition.
    
    Args:
        data: Main branch output of shape (batch_size, seq_len, d_model)
        residual_data: Residual branch output of shape (batch_size, seq_len, d_model)
        output_width: Output bit width (8 or 32)
    
    Returns:
        Tuple of (output tensor, output shape)
    """
    out = data + residual_data
    
    # Clip output if using 8-bit precision
    if output_width == 8:
        out = np.clip(out, -128, 127)
    
    return out, out.shape 