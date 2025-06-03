/*
 * Pattern matching implementation for transformer operations on MAX78000
 */

#include "max7800x_pattern.h"
#include <stdlib.h>
#include <string.h>

// Initialize pattern matcher
void pattern_matcher_init(pattern_matcher_t *matcher, int max_patterns) {
    matcher->patterns = (pattern_t *)malloc(max_patterns * sizeof(pattern_t));
    matcher->num_patterns = 0;
    matcher->max_patterns = max_patterns;
}

// Add pattern to matcher
int pattern_matcher_add(pattern_matcher_t *matcher, pattern_type_t type, int *node_indices, int num_nodes) {
    if (matcher->num_patterns >= matcher->max_patterns) {
        return -1;  // Too many patterns
    }
    
    pattern_t *pattern = &matcher->patterns[matcher->num_patterns++];
    pattern->type = type;
    pattern->start_node = node_indices[0];
    pattern->end_node = node_indices[num_nodes - 1];
    pattern->num_nodes = num_nodes;
    pattern->node_indices = (int *)malloc(num_nodes * sizeof(int));
    memcpy(pattern->node_indices, node_indices, num_nodes * sizeof(int));
    
    return 0;
}

// Find patterns in computation graph
int pattern_matcher_find(pattern_matcher_t *matcher, int *graph, int num_nodes, pattern_t *found_patterns, int max_found) {
    int num_found = 0;
    
    // Check each pattern
    for (int i = 0; i < matcher->num_patterns; i++) {
        pattern_t *pattern = &matcher->patterns[i];
        
        // Check if pattern matches at each position
        for (int j = 0; j <= num_nodes - pattern->num_nodes; j++) {
            int matches = 1;
            
            // Check each node in pattern
            for (int k = 0; k < pattern->num_nodes; k++) {
                if (graph[j + k] != pattern->node_indices[k]) {
                    matches = 0;
                    break;
                }
            }
            
            if (matches) {
                // Pattern found
                if (num_found < max_found) {
                    found_patterns[num_found] = *pattern;
                    found_patterns[num_found].start_node = j;
                    found_patterns[num_found].end_node = j + pattern->num_nodes - 1;
                    num_found++;
                }
            }
        }
    }
    
    return num_found;
}

// Free pattern matcher
void pattern_matcher_free(pattern_matcher_t *matcher) {
    for (int i = 0; i < matcher->num_patterns; i++) {
        free(matcher->patterns[i].node_indices);
    }
    free(matcher->patterns);
}

// Pattern matching helper functions
int is_attention_pattern(int *nodes, int num_nodes) {
    // Check for Q, K, V projections followed by attention computation
    if (num_nodes < 4) return 0;
    
    // Check for matrix multiplications (Q, K, V projections)
    if (nodes[0] != OP_MATMUL || nodes[1] != OP_MATMUL || nodes[2] != OP_MATMUL) {
        return 0;
    }
    
    // Check for attention computation
    if (nodes[3] != OP_ATTENTION) {
        return 0;
    }
    
    return 1;
}

int is_layer_norm_pattern(int *nodes, int num_nodes) {
    // Check for mean, variance, normalization operations
    if (num_nodes < 3) return 0;
    
    // Check for mean computation
    if (nodes[0] != OP_MEAN) {
        return 0;
    }
    
    // Check for variance computation
    if (nodes[1] != OP_VARIANCE) {
        return 0;
    }
    
    // Check for normalization
    if (nodes[2] != OP_NORMALIZE) {
        return 0;
    }
    
    return 1;
}

int is_feed_forward_pattern(int *nodes, int num_nodes) {
    // Check for two linear layers with ReLU in between
    if (num_nodes < 3) return 0;
    
    // Check for first linear layer
    if (nodes[0] != OP_MATMUL) {
        return 0;
    }
    
    // Check for ReLU activation
    if (nodes[1] != OP_RELU) {
        return 0;
    }
    
    // Check for second linear layer
    if (nodes[2] != OP_MATMUL) {
        return 0;
    }
    
    return 1;
}

int is_positional_encoding_pattern(int *nodes, int num_nodes) {
    // Check for sine/cosine operations
    if (num_nodes < 2) return 0;
    
    // Check for sine operation
    if (nodes[0] != OP_SIN) {
        return 0;
    }
    
    // Check for cosine operation
    if (nodes[1] != OP_COS) {
        return 0;
    }
    
    return 1;
}

int is_residual_pattern(int *nodes, int num_nodes) {
    // Check for element-wise addition
    if (num_nodes != 1) return 0;
    
    // Check for addition operation
    if (nodes[0] != OP_ADD) {
        return 0;
    }
    
    return 1;
}

int is_transformer_block_pattern(int *nodes, int num_nodes) {
    // Check for complete transformer block pattern
    if (num_nodes < 7) return 0;
    
    // Check for layer norm
    if (!is_layer_norm_pattern(&nodes[0], 3)) {
        return 0;
    }
    
    // Check for attention
    if (!is_attention_pattern(&nodes[3], 4)) {
        return 0;
    }
    
    // Check for residual connection
    if (!is_residual_pattern(&nodes[7], 1)) {
        return 0;
    }
    
    // Check for layer norm
    if (!is_layer_norm_pattern(&nodes[8], 3)) {
        return 0;
    }
    
    // Check for feed-forward network
    if (!is_feed_forward_pattern(&nodes[11], 3)) {
        return 0;
    }
    
    // Check for residual connection
    if (!is_residual_pattern(&nodes[14], 1)) {
        return 0;
    }
    
    return 1;
} 