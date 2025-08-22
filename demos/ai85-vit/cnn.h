/**************************************************************************************************
* Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
*
* Maxim Integrated Products, Inc. Default Copyright Notice:
* https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
**************************************************************************************************/

/*
 * This header file was automatically @generated for the ai85-vit network from a template.
 * Please do not edit; instead, edit the template and regenerate.
 */

#ifndef __CNN_H__
#define __CNN_H__

#include <stdint.h>
typedef int32_t q31_t;
typedef int16_t q15_t;

/* Return codes */
#define CNN_FAIL 0
#define CNN_OK 1

/*
  SUMMARY OF OPS
  Hardware: 46,962,304 ops (46,778,624 macc; 137,024 comp; 46,656 add; 0 mul; 0 bitwise)
            True MACs: 0
    Layer 0 (patch_embed.conv.0): 501,760 ops (451,584 macc; 50,176 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1 (patch_embed.conv.1): 28,951,552 ops (28,901,376 macc; 50,176 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2 (patch_embed.conv.2): 3,037,824 ops (2,985,984 macc; 5,184 comp; 46,656 add; 0 mul; 0 bitwise)
    Layer 3 (blocks.0.norm1): 331,776 ops (331,776 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4 (blocks.0.attn): 1,343,488 ops (1,343,488 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5 (blocks.0.norm2): 335,872 ops (335,872 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6 (blocks.0.ff.0): 1,353,984 ops (1,343,488 macc; 10,496 comp; 0 add; 0 mul; 0 bitwise)
    Layer 7 (blocks.0.ff.3): 1,343,488 ops (1,343,488 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 8 (blocks.1.norm1): 335,872 ops (335,872 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 9 (blocks.1.attn): 1,343,488 ops (1,343,488 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 10 (blocks.1.norm2): 335,872 ops (335,872 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 11 (blocks.1.ff.0): 1,353,984 ops (1,343,488 macc; 10,496 comp; 0 add; 0 mul; 0 bitwise)
    Layer 12 (blocks.1.ff.3): 1,343,488 ops (1,343,488 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 13 (blocks.2.norm1): 335,872 ops (335,872 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 14 (blocks.2.attn): 1,343,488 ops (1,343,488 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 15 (blocks.2.norm2): 335,872 ops (335,872 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 16 (blocks.2.ff.0): 1,353,984 ops (1,343,488 macc; 10,496 comp; 0 add; 0 mul; 0 bitwise)
    Layer 17 (blocks.2.ff.3): 1,343,488 ops (1,343,488 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 18 (norm): 335,872 ops (335,872 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 19 (head): 1,280 ops (1,280 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 173,696 bytes out of 442,368 bytes total (39.3%)
  Bias memory:   1,418 bytes out of 2,048 bytes total (69.2%)
*/

/* Number of outputs for this network */
#define CNN_NUM_OUTPUTS 10

/* Port pin actions used to signal that processing is active */

#define CNN_START LED_On(1)
#define CNN_COMPLETE LED_Off(1)
#define SYS_START LED_On(0)
#define SYS_COMPLETE LED_Off(0)

/* Run software SoftMax on unloaded data */
void softmax_q17p14_q15(const q31_t * vec_in, const uint16_t dim_vec, q15_t * p_out);
/* Shift the input, then calculate SoftMax */
void softmax_shift_q17p14_q15(q31_t * vec_in, const uint16_t dim_vec, uint8_t in_shift, q15_t * p_out);

/* Stopwatch - holds the runtime when accelerator finishes */
extern volatile uint32_t cnn_time;

/* Custom memcopy routines used for weights and data */
void memcpy32(uint32_t *dst, const uint32_t *src, int n);
void memcpy32_const(uint32_t *dst, int n);

/* Enable clocks and power to accelerator, enable interrupt */
int cnn_enable(uint32_t clock_source, uint32_t clock_divider);

/* Disable clocks and power to accelerator */
int cnn_disable(void);

/* Perform minimum accelerator initialization so it can be configured */
int cnn_init(void);

/* Configure accelerator for the given network */
int cnn_configure(void);

/* Load accelerator weights */
int cnn_load_weights(void);

/* Verify accelerator weights (debug only) */
int cnn_verify_weights(void);

/* Load accelerator bias values (if needed) */
int cnn_load_bias(void);

/* Start accelerator processing */
int cnn_start(void);

/* Force stop accelerator */
int cnn_stop(void);

/* Continue accelerator after stop */
int cnn_continue(void);

/* Unload results from accelerator */
int cnn_unload(uint32_t *out_buf);

/* Turn on the boost circuit */
int cnn_boost_enable(mxc_gpio_regs_t *port, uint32_t pin);

/* Turn off the boost circuit */
int cnn_boost_disable(mxc_gpio_regs_t *port, uint32_t pin);

#endif // __CNN_H__
