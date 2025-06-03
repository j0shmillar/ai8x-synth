/*
 * Memory management implementation for transformer operations on MAX78000
 */

#include "max7800x_memory.h"
#include <string.h>

// Initialize memory pool
void memory_pool_init(memory_pool_t *pool) {
    memset(pool, 0, sizeof(memory_pool_t));
}

// Allocate memory from pool
void *memory_pool_alloc(memory_pool_t *pool, size_t size, const char *name) {
    // Align size for optimal performance
    size = align_size(size);
    
    // Check if we have enough space
    if (size > MAX7800X_MEMORY_POOL_SIZE) {
        return NULL;
    }
    
    // Find a free block
    for (int i = 0; i < pool->num_blocks; i++) {
        if (!pool->blocks[i].is_used) {
            // Check if this block is large enough
            if (pool->blocks[i].size >= size) {
                pool->blocks[i].is_used = 1;
                pool->blocks[i].name = name;
                return pool->blocks[i].ptr;
            }
        }
    }
    
    // If we get here, we need to create a new block
    if (pool->num_blocks >= 32) {
        return NULL;  // Too many blocks
    }
    
    // Calculate offset for new block
    size_t offset = 0;
    for (int i = 0; i < pool->num_blocks; i++) {
        if (pool->blocks[i].is_used) {
            offset += align_size(pool->blocks[i].size);
        }
    }
    
    // Create new block
    pool->blocks[pool->num_blocks].ptr = &pool->pool[offset];
    pool->blocks[pool->num_blocks].size = size;
    pool->blocks[pool->num_blocks].is_used = 1;
    pool->blocks[pool->num_blocks].name = name;
    
    return pool->blocks[pool->num_blocks++].ptr;
}

// Free memory back to pool
void memory_pool_free(memory_pool_t *pool, void *ptr) {
    // Find the block
    for (int i = 0; i < pool->num_blocks; i++) {
        if (pool->blocks[i].ptr == ptr) {
            pool->blocks[i].is_used = 0;
            return;
        }
    }
}

// Get memory usage statistics
void memory_pool_stats(memory_pool_t *pool, size_t *total_used, size_t *total_free) {
    *total_used = 0;
    *total_free = MAX7800X_MEMORY_POOL_SIZE;
    
    for (int i = 0; i < pool->num_blocks; i++) {
        if (pool->blocks[i].is_used) {
            *total_used += pool->blocks[i].size;
            *total_free -= pool->blocks[i].size;
        }
    }
}

// Reset memory pool (free all blocks)
void memory_pool_reset(memory_pool_t *pool) {
    for (int i = 0; i < pool->num_blocks; i++) {
        pool->blocks[i].is_used = 0;
    }
    pool->num_blocks = 0;
} 