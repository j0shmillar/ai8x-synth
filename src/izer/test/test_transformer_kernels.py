"""
Tests for hardware-specific transformer kernels on MAX78000.
"""
import numpy as np
import pytest
from izer.transformer_kernels import (
    attention_kernel,
    layer_norm_kernel,
    feed_forward_kernel,
    positional_encoding_kernel,
    residual_kernel
)

def test_attention_kernel():
    """Test hardware-optimized attention kernel"""
    # Test single-head attention
    batch_size, seq_len, d_model = 2, 4, 8
    data = np.random.randn(batch_size, seq_len, d_model)
    w_q = np.eye(d_model)
    w_k = np.eye(d_model)
    w_v = np.eye(d_model)
    w_o = np.eye(d_model)
    
    # Test without bias
    out, out_shape = attention_kernel(
        data, w_q, w_k, w_v, w_o,
        num_heads=1,
        output_width=8
    )
    assert out_shape == (batch_size, seq_len, d_model)
    assert np.all(out >= -128) and np.all(out <= 127)  # Check 8-bit range
    
    # Test with bias
    b_q = np.ones(d_model)
    b_k = np.ones(d_model)
    b_v = np.ones(d_model)
    b_o = np.ones(d_model)
    out, _ = attention_kernel(
        data, w_q, w_k, w_v, w_o,
        b_q, b_k, b_v, b_o,
        num_heads=1,
        output_width=8
    )
    assert out_shape == (batch_size, seq_len, d_model)
    assert np.all(out >= -128) and np.all(out <= 127)
    
    # Test multi-head attention
    d_model = 16  # Must be divisible by num_heads
    num_heads = 4
    data = np.random.randn(batch_size, seq_len, d_model)
    w_q = np.eye(d_model)
    w_k = np.eye(d_model)
    w_v = np.eye(d_model)
    w_o = np.eye(d_model)
    
    out, out_shape = attention_kernel(
        data, w_q, w_k, w_v, w_o,
        num_heads=num_heads,
        output_width=8
    )
    assert out_shape == (batch_size, seq_len, d_model)
    assert np.all(out >= -128) and np.all(out <= 127)

def test_layer_norm_kernel():
    """Test hardware-optimized layer normalization kernel"""
    # Test basic layer norm
    batch_size, seq_len, d_model = 2, 4, 8
    data = np.random.randn(batch_size, seq_len, d_model)
    weight = np.ones(d_model)
    bias = np.zeros(d_model)
    
    # Test without bias
    out, out_shape = layer_norm_kernel(
        data, weight,
        output_width=8
    )
    assert out_shape == (batch_size, seq_len, d_model)
    assert np.all(out >= -128) and np.all(out <= 127)
    
    # Test with bias
    bias = np.ones(d_model)
    out, _ = layer_norm_kernel(
        data, weight, bias,
        output_width=8
    )
    assert out_shape == (batch_size, seq_len, d_model)
    assert np.all(out >= -128) and np.all(out <= 127)
    
    # Test with custom weight
    weight = np.ones(d_model) * 2
    out, _ = layer_norm_kernel(
        data, weight, bias,
        output_width=8
    )
    assert out_shape == (batch_size, seq_len, d_model)
    assert np.all(out >= -128) and np.all(out <= 127)

def test_feed_forward_kernel():
    """Test hardware-optimized feed-forward network kernel"""
    # Test basic feed-forward
    batch_size, seq_len, d_model = 2, 4, 8
    d_ff = 32
    data = np.random.randn(batch_size, seq_len, d_model)
    w1 = np.random.randn(d_ff, d_model)
    w2 = np.random.randn(d_model, d_ff)
    
    # Test without bias
    out, out_shape = feed_forward_kernel(
        data, w1, w2,
        output_width=8
    )
    assert out_shape == (batch_size, seq_len, d_model)
    assert np.all(out >= -128) and np.all(out <= 127)
    
    # Test with bias
    b1 = np.ones(d_ff)
    b2 = np.ones(d_model)
    out, _ = feed_forward_kernel(
        data, w1, w2, b1, b2,
        output_width=8
    )
    assert out_shape == (batch_size, seq_len, d_model)
    assert np.all(out >= -128) and np.all(out <= 127)
    
    # Test with ReLU activation (negative inputs)
    data = np.random.randn(batch_size, seq_len, d_model) * -1
    out, _ = feed_forward_kernel(
        data, w1, w2, b1, b2,
        output_width=8
    )
    assert out_shape == (batch_size, seq_len, d_model)
    assert np.all(out >= -128) and np.all(out <= 127)

def test_positional_encoding_kernel():
    """Test hardware-optimized positional encoding kernel"""
    # Test basic positional encoding
    seq_len, d_model = 4, 8
    out, out_shape = positional_encoding_kernel(
        seq_len, d_model,
        output_width=8
    )
    assert out_shape == (seq_len, d_model)
    assert np.all(out >= -128) and np.all(out <= 127)
    
    # Test sine/cosine pattern
    assert np.all(np.abs(out[:, 0::2]) <= 1)  # sine values
    assert np.all(np.abs(out[:, 1::2]) <= 1)  # cosine values
    
    # Test with larger dimensions
    seq_len, d_model = 16, 32
    out, out_shape = positional_encoding_kernel(
        seq_len, d_model,
        output_width=8
    )
    assert out_shape == (seq_len, d_model)
    assert np.all(out >= -128) and np.all(out <= 127)

def test_residual_kernel():
    """Test hardware-optimized residual connection kernel"""
    # Test basic residual
    batch_size, seq_len, d_model = 2, 4, 8
    data = np.random.randn(batch_size, seq_len, d_model)
    residual_data = np.random.randn(batch_size, seq_len, d_model)
    
    out, out_shape = residual_kernel(
        data, residual_data,
        output_width=8
    )
    assert out_shape == (batch_size, seq_len, d_model)
    assert np.all(out >= -128) and np.all(out <= 127)
    
    # Test with large values (should be clipped)
    data = np.ones((batch_size, seq_len, d_model)) * 100
    residual_data = np.ones((batch_size, seq_len, d_model)) * 100
    out, _ = residual_kernel(
        data, residual_data,
        output_width=8
    )
    assert out_shape == (batch_size, seq_len, d_model)
    assert np.all(out >= -128) and np.all(out <= 127)
    
    # Test with negative values
    data = np.ones((batch_size, seq_len, d_model)) * -100
    residual_data = np.ones((batch_size, seq_len, d_model)) * -100
    out, _ = residual_kernel(
        data, residual_data,
        output_width=8
    )
    assert out_shape == (batch_size, seq_len, d_model)
    assert np.all(out >= -128) and np.all(out <= 127)

def test_kernel_precision():
    """Test kernel precision handling"""
    batch_size, seq_len, d_model = 2, 4, 8
    data = np.random.randn(batch_size, seq_len, d_model)
    
    # Test 32-bit precision
    data_scaled = data * 1e3  # Ensure output exceeds 8-bit range
    out, _ = attention_kernel(
        data_scaled, np.eye(d_model), np.eye(d_model),
        np.eye(d_model), np.eye(d_model),
        output_width=32
    )
    assert np.any(out < -128) or np.any(out > 127)  # Should not be clipped
    
    # Test 8-bit precision
    out, _ = attention_kernel(
        data, np.eye(d_model), np.eye(d_model),
        np.eye(d_model), np.eye(d_model),
        output_width=8
    )
    assert np.all(out >= -128) and np.all(out <= 127)  # Should be clipped

if __name__ == '__main__':
    pytest.main([__file__]) 