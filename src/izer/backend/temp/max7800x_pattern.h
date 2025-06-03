/*
 * Pattern matching for transformer operations on MAX78000
 * Identifies transformer blocks in the computation graph
 */

#ifndef MAX7800X_PATTERN_H
#define MAX7800X_PATTERN_H

#include <stdint.h>

// Pattern types
typedef enum {
    PATTERN_ATTENTION,
    PATTERN_LAYER_NORM,
    PATTERN_FEED_FORWARD,
    PATTERN_POSITIONAL_ENCODING,
    PATTERN_RESIDUAL,
    PATTERN_TRANSFORMER_BLOCK
} pattern_type_t;

// Pattern structure
typedef struct {
    pattern_type_t type;    // Type of pattern
    int start_node;         // Start node index
    int end_node;           // End node index
    int num_nodes;          // Number of nodes in pattern
    int *node_indices;      // Indices of nodes in pattern
} pattern_t;

// Pattern matcher structure
typedef struct {
    pattern_t *patterns;    // Array of patterns
    int num_patterns;       // Number of patterns
    int max_patterns;       // Maximum number of patterns
} pattern_matcher_t;

// Initialize pattern matcher
void pattern_matcher_init(pattern_matcher_t *matcher, int max_patterns);

// Add pattern to matcher
int pattern_matcher_add(pattern_matcher_t *matcher, pattern_type_t type, int *node_indices, int num_nodes);

// Find patterns in computation graph
int pattern_matcher_find(pattern_matcher_t *matcher, int *graph, int num_nodes, pattern_t *found_patterns, int max_found);

// Free pattern matcher
void pattern_matcher_free(pattern_matcher_t *matcher);

// Pattern matching helper functions
int is_attention_pattern(int *nodes, int num_nodes);
int is_layer_norm_pattern(int *nodes, int num_nodes);
int is_feed_forward_pattern(int *nodes, int num_nodes);
int is_positional_encoding_pattern(int *nodes, int num_nodes);
int is_residual_pattern(int *nodes, int num_nodes);
int is_transformer_block_pattern(int *nodes, int num_nodes);

#endif // MAX7800X_PATTERN_H 