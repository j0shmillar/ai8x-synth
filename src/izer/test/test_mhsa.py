import numpy as np
from izer.compute import mhsa_to_conv2d, softmax

def test_mhsa_to_conv2d():
    # Create dummy data and weights
    N, C_in, C_out, num_heads = 10, 20, 30, 2
    data = np.random.randn(N, C_in)
    w_q = np.random.randn(C_in, C_out)
    w_k = np.random.randn(C_in, C_out)
    w_v = np.random.randn(C_in, C_out)
    w_o = np.random.randn(C_out, C_out)
    bias_q = np.random.randn(C_out)
    bias_k = np.random.randn(C_out)
    bias_v = np.random.randn(C_out)
    bias_o = np.random.randn(C_out)

    # Run MHSA emulation using pointwise convolutions
    output = mhsa_to_conv2d(data, w_q, w_k, w_v, w_o, bias_q, bias_k, bias_v, bias_o, (C_in,), (C_out,), num_heads)

    # Reference implementation using direct matrix operations
    q = np.dot(data, w_q) + bias_q
    k = np.dot(data, w_k) + bias_k
    v = np.dot(data, w_v) + bias_v
    q = q.reshape(N, num_heads, -1)
    k = k.reshape(N, num_heads, -1)
    v = v.reshape(N, num_heads, -1)
    a = np.matmul(q, k.transpose(0, 2, 1))
    a = a / np.sqrt(a.shape[-1])
    a = softmax(a, axis=-1)
    o = np.matmul(a, v)
    o = o.reshape(N, -1)
    expected_output = np.dot(o, w_o) + bias_o

    # Verify output shape
    assert output.shape == (N, C_out), f"Expected output shape {(N, C_out)}, got {output.shape}"
    # Verify correctness by comparing with reference output
    assert np.allclose(output, expected_output, rtol=1e-5, atol=1e-5), "MHSA emulation output does not match reference output"

if __name__ == "__main__":
    test_mhsa_to_conv2d()
    print("MHSA emulation test passed!") 