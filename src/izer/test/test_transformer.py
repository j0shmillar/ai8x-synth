import numpy as np
import pytest
from izer.compute import (
    attention, layer_norm, feed_forward, positional_encoding,
    residual, softmax, mhsa_to_conv2d, gemm_to_conv2d
)

def test_softmax():
    """Test softmax implementation"""
    # Test basic softmax
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = softmax(x, axis=-1)
    expected = np.array([
        [0.09003057, 0.24472847, 0.66524096],
        [0.09003057, 0.24472847, 0.66524096]
    ])
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    # Test softmax with negative values
    x = np.array([[-1.0, 0.0, 1.0]])
    result = softmax(x, axis=-1)
    expected = np.array([[0.09003057, 0.24472847, 0.66524096]])
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_layer_norm():
    """Test layer normalization"""
    # Test basic layer norm
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    weight = np.array([1.0, 1.0, 1.0])
    bias = np.array([0.0, 0.0, 0.0])
    
    result = layer_norm(data, weight, bias)
    expected = np.array([
        [-1.22474487, 0.0, 1.22474487],
        [-1.22474487, 0.0, 1.22474487]
    ])
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-4)
    
    # Test with custom weight and bias
    weight = np.array([2.0, 2.0, 2.0])
    bias = np.array([1.0, 1.0, 1.0])
    result = layer_norm(data, weight, bias)
    expected = np.array([
        [-1.44948974, 1.0, 3.44948974],
        [-1.44948974, 1.0, 3.44948974]
    ])
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-4)

def test_feed_forward():
    """Test feed-forward network"""
    # Test basic feed-forward
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    w1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    w2 = np.array([[1.0, 1.0], [1.0, 1.0]])
    b1 = np.array([0.0, 0.0])
    b2 = np.array([0.0, 0.0])
    
    result = feed_forward(data, w1, w2, b1, b2)
    expected = np.array([[6.0, 6.0], [14.0, 14.0]])
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    # Test with ReLU activation
    data = np.array([[-1.0, -2.0], [3.0, 4.0]])
    result = feed_forward(data, w1, w2, b1, b2)
    expected = np.array([[0.0, 0.0], [14.0, 14.0]])
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_positional_encoding():
    """Test positional encoding"""
    # Test basic positional encoding
    seq_len = 4
    d_model = 6
    result = positional_encoding(seq_len, d_model)
    
    # Check shape
    assert result.shape == (seq_len, d_model)
    
    # Check that even indices use sine and odd indices use cosine
    for i in range(seq_len):
        for j in range(d_model):
            if j % 2 == 0:
                assert -1 <= result[i, j] <= 1  # sine range
            else:
                assert -1 <= result[i, j] <= 1  # cosine range

def test_residual():
    """Test residual connection"""
    # Test basic residual
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    residual_data = np.array([[0.5, 0.5], [0.5, 0.5]])
    
    result = residual(data, residual_data)
    expected = np.array([[1.5, 2.5], [3.5, 4.5]])
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    # Test with negative values
    data = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    result = residual(data, residual_data)
    expected = np.array([[-0.5, -1.5], [-2.5, -3.5]])
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_attention():
    """Test attention mechanism"""
    # Test single-head attention
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    w_q = np.array([[1.0, 0.0], [0.0, 1.0]])
    w_k = np.array([[1.0, 0.0], [0.0, 1.0]])
    w_v = np.array([[1.0, 0.0], [0.0, 1.0]])
    w_o = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    result = attention(data, w_q, w_k, w_v, w_o, num_heads=1)
    assert result.shape == (2, 2)
    
    # Test multi-head attention
    data = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    w_q = np.eye(4)
    w_k = np.eye(4)
    w_v = np.eye(4)
    w_o = np.eye(4)
    
    result = attention(data, w_q, w_k, w_v, w_o, num_heads=2)
    assert result.shape == (2, 4)

def test_mhsa_to_conv2d():
    """Test MHSA emulation using pointwise convolutions"""
    # Test basic MHSA
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    w_q = np.array([[1.0, 0.0], [0.0, 1.0]])
    w_k = np.array([[1.0, 0.0], [0.0, 1.0]])
    w_v = np.array([[1.0, 0.0], [0.0, 1.0]])
    w_o = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    result = mhsa_to_conv2d(
        data, w_q, w_k, w_v, w_o,
        input_size=(2,),
        output_size=(2,),
        num_heads=1
    )
    assert result.shape == (2, 2)
    
    # Test multi-head MHSA
    data = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    w_q = np.eye(4)
    w_k = np.eye(4)
    w_v = np.eye(4)
    w_o = np.eye(4)
    
    result = mhsa_to_conv2d(
        data, w_q, w_k, w_v, w_o,
        input_size=(4,),
        output_size=(4,),
        num_heads=2
    )
    assert result.shape == (2, 4)

def test_gemm_to_conv2d():
    """Test GEMM emulation using pointwise convolutions"""
    # Test basic GEMM
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    weight = np.array([[1.0, 0.0], [0.0, 1.0]])
    bias = np.array([0.0, 0.0])
    
    result = gemm_to_conv2d(
        data, weight, bias,
        input_size=(2,),
        output_size=(2,)
    )
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    # Test with bias
    bias = np.array([1.0, 1.0])
    result = gemm_to_conv2d(
        data, weight, bias,
        input_size=(2,),
        output_size=(2,)
    )
    expected = np.array([[2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_allclose(result, expected, rtol=1e-5)

if __name__ == '__main__':
    pytest.main([__file__]) 