/*
 * Transformer compiler for MAX78000
 * Optimizes transformer workloads for the MAX78000 hardware
 */

#ifndef MAX7800X_TRANSFORMER_COMPILER_H
#define MAX7800X_TRANSFORMER_COMPILER_H

#include <stdint.h>
#include "max7800x_pattern.h"
#include "max7800x_memory.h"

// Compiler configuration
typedef struct {
    int max_batch_size;     // Maximum batch size
    int max_seq_len;        // Maximum sequence length
    int max_d_model;        // Maximum model dimension
    int max_num_heads;      // Maximum number of attention heads
    int max_d_ff;           // Maximum feed-forward dimension
    int output_width;       // Output bit width (8 or 32)
    int use_hardware_accel; // Whether to use hardware acceleration
} transformer_compiler_config_t;

// Compiler statistics
typedef struct {
    int num_attention_ops;      // Number of attention operations
    int num_layer_norm_ops;     // Number of layer normalization operations
    int num_feed_forward_ops;   // Number of feed-forward operations
    int num_residual_ops;       // Number of residual connections
    int total_memory_used;      // Total memory used (bytes)
    int total_compute_ops;      // Total compute operations
} transformer_compiler_stats_t;

// Initialize transformer compiler
void transformer_compiler_init(transformer_compiler_config_t *config);

// Compile transformer model
int transformer_compiler_compile(
    const char *model_path,     // Path to input model
    const char *output_path,    // Path to output compiled model
    transformer_compiler_config_t *config,
    transformer_compiler_stats_t *stats
);

// Optimize transformer operations
int transformer_compiler_optimize(
    pattern_t *patterns,        // Found patterns
    int num_patterns,           // Number of patterns
    memory_pool_t *memory_pool, // Memory pool
    transformer_compiler_config_t *config
);

// Generate hardware-specific code
int transformer_compiler_generate_code(
    const char *output_path,    // Path to output code
    pattern_t *patterns,        // Optimized patterns
    int num_patterns,           // Number of patterns
    transformer_compiler_config_t *config
);

// Free transformer compiler
void transformer_compiler_free(void);

#endif // MAX7800X_TRANSFORMER_COMPILER_H 