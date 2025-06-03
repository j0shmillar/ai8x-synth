/*
 * Hardware-optimized transformer kernels for MAX78000
 * These kernels are optimized for the MAX78000's hardware capabilities
 */

#include <stdint.h>
#include <string.h>
#include "max78000.h"
#include "cnn.h"

// Helper function for softmax
static void softmax_q7(int8_t *input, int8_t *output, int len) {
    int32_t max_val = input[0];
    int32_t sum = 0;
    
    // Find max value for numerical stability
    for (int i = 1; i < len; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // Compute exp and sum
    for (int i = 0; i < len; i++) {
        int32_t exp_val = input[i] - max_val;
        // Approximate exp using lookup table or shift operations
        // For MAX78000, we can use the hardware's exp approximation
        output[i] = cnn_exp(exp_val);
        sum += output[i];
    }
    
    // Normalize
    for (int i = 0; i < len; i++) {
        output[i] = (output[i] * 127) / sum;
    }
}

// Hardware-optimized attention kernel
void attention_kernel_q7(
    int8_t *data,          // Input tensor (batch_size, seq_len, d_model)
    int8_t *w_q,           // Query weights (d_model, d_model)
    int8_t *w_k,           // Key weights (d_model, d_model)
    int8_t *w_v,           // Value weights (d_model, d_model)
    int8_t *w_o,           // Output weights (d_model, d_model)
    int8_t *b_q,           // Query bias (d_model)
    int8_t *b_k,           // Key bias (d_model)
    int8_t *b_v,           // Value bias (d_model)
    int8_t *b_o,           // Output bias (d_model)
    int8_t *output,        // Output tensor
    int batch_size,
    int seq_len,
    int d_model,
    int num_heads,
    int output_width
) {
    int head_dim = d_model / num_heads;
    int8_t *q = (int8_t *)malloc(batch_size * seq_len * d_model);
    int8_t *k = (int8_t *)malloc(batch_size * seq_len * d_model);
    int8_t *v = (int8_t *)malloc(batch_size * seq_len * d_model);
    int8_t *scores = (int8_t *)malloc(batch_size * num_heads * seq_len * seq_len);
    int8_t *attn = (int8_t *)malloc(batch_size * num_heads * seq_len * seq_len);
    
    // Project Q, K, V using hardware-optimized matrix multiplication
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            // Use MAX78000's hardware matrix multiplication
            cnn_matmul_q7(
                &data[b * seq_len * d_model + s * d_model],
                w_q,
                &q[b * seq_len * d_model + s * d_model],
                d_model,
                d_model
            );
            if (b_q) {
                cnn_add_q7(
                    &q[b * seq_len * d_model + s * d_model],
                    b_q,
                    &q[b * seq_len * d_model + s * d_model],
                    d_model
                );
            }
            
            // Similar for K and V projections
            cnn_matmul_q7(
                &data[b * seq_len * d_model + s * d_model],
                w_k,
                &k[b * seq_len * d_model + s * d_model],
                d_model,
                d_model
            );
            if (b_k) {
                cnn_add_q7(
                    &k[b * seq_len * d_model + s * d_model],
                    b_k,
                    &k[b * seq_len * d_model + s * d_model],
                    d_model
                );
            }
            
            cnn_matmul_q7(
                &data[b * seq_len * d_model + s * d_model],
                w_v,
                &v[b * seq_len * d_model + s * d_model],
                d_model,
                d_model
            );
            if (b_v) {
                cnn_add_q7(
                    &v[b * seq_len * d_model + s * d_model],
                    b_v,
                    &v[b * seq_len * d_model + s * d_model],
                    d_model
                );
            }
        }
    }
    
    // Compute attention scores
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    int32_t score = 0;
                    for (int d = 0; d < head_dim; d++) {
                        score += q[b * seq_len * d_model + i * d_model + h * head_dim + d] *
                                k[b * seq_len * d_model + j * d_model + h * head_dim + d];
                    }
                    scores[b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j] = 
                        score / (head_dim * 128);  // Scale by sqrt(head_dim)
                }
            }
        }
    }
    
    // Apply softmax to attention scores
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int i = 0; i < seq_len; i++) {
                softmax_q7(
                    &scores[b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len],
                    &attn[b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len],
                    seq_len
                );
            }
        }
    }
    
    // Apply attention to values and project output
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            // Use hardware-optimized matrix multiplication for final projection
            cnn_matmul_q7(
                &attn[b * num_heads * seq_len * seq_len + s * seq_len],
                w_o,
                &output[b * seq_len * d_model + s * d_model],
                d_model,
                d_model
            );
            if (b_o) {
                cnn_add_q7(
                    &output[b * seq_len * d_model + s * d_model],
                    b_o,
                    &output[b * seq_len * d_model + s * d_model],
                    d_model
                );
            }
        }
    }
    
    // Clip output if using 8-bit precision
    if (output_width == 8) {
        for (int i = 0; i < batch_size * seq_len * d_model; i++) {
            if (output[i] < -128) output[i] = -128;
            if (output[i] > 127) output[i] = 127;
        }
    }
    
    free(q);
    free(k);
    free(v);
    free(scores);
    free(attn);
}

// Hardware-optimized layer normalization kernel
void layer_norm_kernel_q7(
    int8_t *data,          // Input tensor (batch_size, seq_len, d_model)
    int8_t *weight,        // Scale parameter (d_model)
    int8_t *bias,          // Shift parameter (d_model)
    int8_t *output,        // Output tensor
    int batch_size,
    int seq_len,
    int d_model,
    int output_width
) {
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            // Compute mean using hardware-optimized sum
            int32_t sum = 0;
            for (int d = 0; d < d_model; d++) {
                sum += data[b * seq_len * d_model + s * d_model + d];
            }
            int8_t mean = sum / d_model;
            
            // Compute variance
            int32_t var_sum = 0;
            for (int d = 0; d < d_model; d++) {
                int32_t diff = data[b * seq_len * d_model + s * d_model + d] - mean;
                var_sum += diff * diff;
            }
            int8_t var = var_sum / d_model;
            
            // Normalize, scale, and shift
            for (int d = 0; d < d_model; d++) {
                int32_t normalized = (data[b * seq_len * d_model + s * d_model + d] - mean) * 128 / var;
                output[b * seq_len * d_model + s * d_model + d] = 
                    (normalized * weight[d] + (bias ? bias[d] : 0)) / 128;
            }
        }
    }
    
    // Clip output if using 8-bit precision
    if (output_width == 8) {
        for (int i = 0; i < batch_size * seq_len * d_model; i++) {
            if (output[i] < -128) output[i] = -128;
            if (output[i] > 127) output[i] = 127;
        }
    }
}

// Hardware-optimized feed-forward network kernel
void feed_forward_kernel_q7(
    int8_t *data,          // Input tensor (batch_size, seq_len, d_model)
    int8_t *w1,            // First weight matrix (d_ff, d_model)
    int8_t *w2,            // Second weight matrix (d_model, d_ff)
    int8_t *b1,            // First bias vector (d_ff)
    int8_t *b2,            // Second bias vector (d_model)
    int8_t *output,        // Output tensor
    int batch_size,
    int seq_len,
    int d_model,
    int d_ff,
    int output_width
) {
    int8_t *hidden = (int8_t *)malloc(batch_size * seq_len * d_ff);
    
    // First linear layer
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            cnn_matmul_q7(
                &data[b * seq_len * d_model + s * d_model],
                w1,
                &hidden[b * seq_len * d_ff + s * d_ff],
                d_model,
                d_ff
            );
            if (b1) {
                cnn_add_q7(
                    &hidden[b * seq_len * d_ff + s * d_ff],
                    b1,
                    &hidden[b * seq_len * d_ff + s * d_ff],
                    d_ff
                );
            }
        }
    }
    
    // ReLU activation
    for (int i = 0; i < batch_size * seq_len * d_ff; i++) {
        if (hidden[i] < 0) hidden[i] = 0;
    }
    
    // Second linear layer
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            cnn_matmul_q7(
                &hidden[b * seq_len * d_ff + s * d_ff],
                w2,
                &output[b * seq_len * d_model + s * d_model],
                d_ff,
                d_model
            );
            if (b2) {
                cnn_add_q7(
                    &output[b * seq_len * d_model + s * d_model],
                    b2,
                    &output[b * seq_len * d_model + s * d_model],
                    d_model
                );
            }
        }
    }
    
    // Clip output if using 8-bit precision
    if (output_width == 8) {
        for (int i = 0; i < batch_size * seq_len * d_model; i++) {
            if (output[i] < -128) output[i] = -128;
            if (output[i] > 127) output[i] = 127;
        }
    }
    
    free(hidden);
}

// Hardware-optimized positional encoding kernel
void positional_encoding_kernel_q7(
    int8_t *output,        // Output tensor (seq_len, d_model)
    int seq_len,
    int d_model,
    int output_width
) {
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < d_model; i++) {
            float angle = pos / powf(10000.0f, 2.0f * i / d_model);
            if (i % 2 == 0) {
                output[pos * d_model + i] = (int8_t)(127.0f * sinf(angle));
            } else {
                output[pos * d_model + i] = (int8_t)(127.0f * cosf(angle));
            }
        }
    }
    
    // Clip output if using 8-bit precision
    if (output_width == 8) {
        for (int i = 0; i < seq_len * d_model; i++) {
            if (output[i] < -128) output[i] = -128;
            if (output[i] > 127) output[i] = 127;
        }
    }
}

// Hardware-optimized residual connection kernel
void residual_kernel_q7(
    int8_t *data,          // Main branch output (batch_size, seq_len, d_model)
    int8_t *residual_data, // Residual branch output (batch_size, seq_len, d_model)
    int8_t *output,        // Output tensor
    int batch_size,
    int seq_len,
    int d_model,
    int output_width
) {
    // Element-wise addition using hardware-optimized operations
    for (int i = 0; i < batch_size * seq_len * d_model; i++) {
        output[i] = data[i] + residual_data[i];
    }
    
    // Clip output if using 8-bit precision
    if (output_width == 8) {
        for (int i = 0; i < batch_size * seq_len * d_model; i++) {
            if (output[i] < -128) output[i] = -128;
            if (output[i] > 127) output[i] = 127;
        }
    }
}

// Hardware-optimized patch embedding kernel
void patch_embedding_kernel_q7(
    int8_t *data,          // Input tensor (batch_size, channels, height, width)
    int8_t *weight,        // Embedding weights (d_model, channels, kernel_size, kernel_size)
    int8_t *bias,          // Embedding bias (d_model)
    int8_t *output,        // Output tensor (batch_size, seq_len, d_model)
    int batch_size,
    int channels,
    int height,
    int width,
    int d_model,
    int kernel_size,
    int stride,
    int output_width
) {
    // Use hardware-optimized convolution for patch embedding
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < height; h += stride) {
            for (int w = 0; w < width; w += stride) {
                // Extract patch and apply convolution
                if (kernel_size == 4) {
                    // Special handling for 4x4 kernels
                    cnn_conv2d_q7_4x4(
                        &data[b * channels * height * width + h * width + w],
                        weight,
                        &output[b * (height/stride) * (width/stride) * d_model + 
                               (h/stride) * (width/stride) * d_model + (w/stride) * d_model],
                        channels,
                        d_model,
                        stride,
                        stride,
                        0,  // No padding
                        0,  // No padding
                        1,  // No dilation
                        1   // No groups
                    );
                } else {
                    cnn_conv2d_q7(
                        &data[b * channels * height * width + h * width + w],
                        weight,
                        &output[b * (height/stride) * (width/stride) * d_model + 
                               (h/stride) * (width/stride) * d_model + (w/stride) * d_model],
                        channels,
                        d_model,
                        kernel_size,
                        kernel_size,
                        stride,
                        stride,
                        0,  // No padding
                        0,  // No padding
                        1,  // No dilation
                        1   // No groups
                    );
                }
                
                // Add bias if provided
                if (bias) {
                    cnn_add_q7(
                        &output[b * (height/stride) * (width/stride) * d_model + 
                               (h/stride) * (width/stride) * d_model + (w/stride) * d_model],
                        bias,
                        &output[b * (height/stride) * (width/stride) * d_model + 
                               (h/stride) * (width/stride) * d_model + (w/stride) * d_model],
                        d_model
                    );
                }
            }
        }
    }
    
    // Clip output if using 8-bit precision
    if (output_width == 8) {
        for (int i = 0; i < batch_size * (height/stride) * (width/stride) * d_model; i++) {
            if (output[i] < -128) output[i] = -128;
            if (output[i] > 127) output[i] = 127;
        }
    }
} 