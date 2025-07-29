#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <stdint.h>
#include "cnn.h"  // or whatever provides CNN_NUM_OUTPUTS, q15_t, etc.

void softmax_q17p14_q15(const int32_t *input, uint32_t length, q15_t *output);
void softmax_shift_q17p14_q15(const int32_t *input, uint32_t length, q15_t *output);

#endif // SOFTMAX_H