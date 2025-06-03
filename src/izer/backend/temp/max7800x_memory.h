/*
 * Memory management for transformer operations on MAX78000
 * Handles efficient allocation and deallocation of memory for transformer operations
 */

#ifndef MAX7800X_MEMORY_H
#define MAX7800X_MEMORY_H

#include <stdint.h>
#include <stddef.h>

// Memory pool configuration
#define MAX7800X_MEMORY_POOL_SIZE (1024 * 1024)  // 1MB memory pool
#define MAX7800X_ALIGNMENT 8  // 8-byte alignment for optimal performance

// Memory block structure
typedef struct {
    void *ptr;              // Pointer to allocated memory
    size_t size;            // Size of allocated memory
    int is_used;            // Whether the block is currently in use
    const char *name;       // Name of the block for debugging
} memory_block_t;

// Memory pool structure
typedef struct {
    uint8_t pool[MAX7800X_MEMORY_POOL_SIZE];  // Memory pool
    memory_block_t blocks[32];                // Memory blocks
    int num_blocks;                           // Number of allocated blocks
} memory_pool_t;

// Initialize memory pool
void memory_pool_init(memory_pool_t *pool);

// Allocate memory from pool
void *memory_pool_alloc(memory_pool_t *pool, size_t size, const char *name);

// Free memory back to pool
void memory_pool_free(memory_pool_t *pool, void *ptr);

// Get memory usage statistics
void memory_pool_stats(memory_pool_t *pool, size_t *total_used, size_t *total_free);

// Reset memory pool (free all blocks)
void memory_pool_reset(memory_pool_t *pool);

// Memory alignment helper
static inline size_t align_size(size_t size) {
    return (size + MAX7800X_ALIGNMENT - 1) & ~(MAX7800X_ALIGNMENT - 1);
}

#endif // MAX7800X_MEMORY_H 