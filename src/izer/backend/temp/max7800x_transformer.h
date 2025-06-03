/*
 * Hardware-optimized transformer kernels for MAX78000
 * These kernels are optimized for the MAX78000's hardware capabilities
 */

#ifndef MAX7800X_TRANSFORMER_H
#define MAX7800X_TRANSFORMER_H

#include <stdint.h>

// Attention kernel
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
);

// Layer normalization kernel
void layer_norm_kernel_q7(
    int8_t *data,          // Input tensor (batch_size, seq_len, d_model)
    int8_t *weight,        // Scale parameter (d_model)
    int8_t *bias,          // Shift parameter (d_model)
    int8_t *output,        // Output tensor
    int batch_size,
    int seq_len,
    int d_model,
    int output_width
);

// Feed-forward network kernel
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
);

// Positional encoding kernel
void positional_encoding_kernel_q7(
    int8_t *output,        // Output tensor (seq_len, d_model)
    int seq_len,
    int d_model,
    int output_width
);

// Residual connection kernel
void residual_kernel_q7(
    int8_t *data,          // Main branch output (batch_size, seq_len, d_model)
    int8_t *residual_data, // Residual branch output (batch_size, seq_len, d_model)
    int8_t *output,        // Output tensor
    int batch_size,
    int seq_len,
    int d_model,
    int output_width
);

// Patch embedding kernel
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
);

#endif // MAX7800X_TRANSFORMER_H 