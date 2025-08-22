/******************************************************************************
 *
 * Copyright (C) 2022-2023 Maxim Integrated Products, Inc. (now owned by 
 * Analog Devices, Inc.),
 * Copyright (C) 2023-2024 Analog Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ******************************************************************************/

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc_device.h"
#include "mxc_sys.h"
#include "gcfr_regs.h"
#include "fcr_regs.h"
#include "icc.h"
#include "dma.h"
#include "led.h"
#include "tmr.h"
#include "pb.h"
#include "cnn.h"
#include "weights.h"
#include "mxc_delay.h"
#include "camera.h"
#include "softmax.h"  
#include "example_config.h"
#include "tft_utils.h"
#include "tft_ili9341.h"

#define USE_CAMERA_INPUT 0
#include "sample_data.h" 

#define CAMERA_TO_LCD (1)
#define IMAGE_SIZE_X (64)
#define IMAGE_SIZE_Y (64)
#define CAMERA_FREQ (10 * 1000 * 1000)

#define TFT_BUFF_SIZE 35 

// === MNIST ===
#define CNN_NUM_OUTPUTS 10      
#define MNIST_W 28
#define MNIST_H 28

const char classes[CNN_NUM_OUTPUTS][10] = { "0","1","2","3","4","5","6","7","8","9" };
char buff[TFT_BUFF_SIZE];

volatile uint32_t cnn_time; // Stopwatch
uint32_t input_0_camera[1024];
uint32_t input_1_camera[1024];
uint32_t input_2_camera[1024];

int font_1 = (int)&Liberation_Sans16x16[0];
int font_2 = (int)&Liberation_Sans16x16[0];

void fail(void)
{
    printf("\n*** FAIL ***\n\n");
    while (1) {}
}

// ---- 0 = use sample bytes, 1 = use cam ----
#define USE_CAMERA_INPUT 0

static uint8_t in784[784]; 
static const uint32_t input_0[] = SAMPLE_INPUT_0; // its a 7

void load_input(void)
{
#if USE_CAMERA_INPUT == 0
    memcpy32((uint32_t *)0x50400000, input_0, 196);
#else
    uint8_t *buf; uint32_t len, w, h;
    camera_get_image(&buf, &len, &w, &h);  

    const float sx = (float)w / 28.0f;
    const float sy = (float)h / 28.0f;

    for (uint32_t yo = 0; yo < 28; ++yo) {
        uint32_t yi = (uint32_t)(yo * sy);
        uint32_t row_off = yi * w * 4; // RGB888 (+pad) = 4 bytes/pixel
        for (uint32_t xo = 0; xo < 28; ++xo) {
            uint32_t xi = (uint32_t)(xo * sx);
            uint32_t idx = row_off + xi * 4;

            uint8_t r = buf[idx + 0];
            uint8_t g = buf[idx + 1];
            uint8_t b = buf[idx + 2];

            // luma 0..255 (white-on-black)
            uint8_t gray = (uint8_t)((77u * r + 150u * g + 29u * b) >> 8);

            in784[yo * 28 + xo] = gray;
        }
    }

    memcpy((uint32_t *) 0x50400000, in784, 784); // TODO should be 196
#endif
}
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

void softmax_layer(void)
{
    cnn_unload((uint32_t *) ml_data);
    softmax_q17p14_q15((const q31_t *) ml_data, CNN_NUM_OUTPUTS, ml_softmax);
}

/* **************************************************************************** */
static uint8_t signed_to_unsigned(int8_t val)
{
    uint8_t value;
    if (val < 0) {
        value = ~val + 1;
        return (128 - value);
    }
    return val + 128;
}

/* **************************************************************************** */
int8_t unsigned_to_signed(uint8_t val)
{
    return val - 128;
}

/* **************************************************************************** */
void lcd_show_sampledata(uint32_t *data0, uint32_t *data1, uint32_t *data2, int xcord, int ycord,
                         int length)
{
    int i;
    int j;
    int x;
    int y;
    int r;
    int g;
    int b;
    int scale = 1.2;

    uint32_t color;
    uint8_t *ptr0;
    uint8_t *ptr1;
    uint8_t *ptr2;

    x = xcord;
    y = ycord;
    for (i = 0; i < length; i++) {
        ptr0 = (uint8_t *)&data0[i];
        ptr1 = (uint8_t *)&data1[i];
        ptr2 = (uint8_t *)&data2[i];
        for (j = 0; j < 4; j++) {
            r = ptr0[j];
            g = ptr1[j];
            b = ptr2[j];
            color = RGB(r, g, b); // convert to RGB565
            MXC_TFT_WritePixel(x * scale, y * scale, scale, scale, color);
            x += 1;
            if (x >= (IMAGE_SIZE_X + xcord)) {
                x = xcord;
                y += 1;
                if ((y + 6) >= (IMAGE_SIZE_Y + ycord))
                    return;
            }
        }
    }
}

/* **************************************************************************** */
void process_camera_img(uint32_t *data0, uint32_t *data1, uint32_t *data2)
{
    uint8_t *frame_buffer;
    uint32_t imgLen;
    uint32_t w, h, x, y;
    uint8_t *ptr0;
    uint8_t *ptr1;
    uint8_t *ptr2;
    uint8_t *buffer;

    camera_get_image(&frame_buffer, &imgLen, &w, &h);
    ptr0 = (uint8_t *)data0;
    ptr1 = (uint8_t *)data1;
    ptr2 = (uint8_t *)data2;
    buffer = frame_buffer;
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++, ptr0++, ptr1++, ptr2++) {
            *ptr0 = (*buffer);
            buffer++;
            *ptr1 = (*buffer);
            buffer++;
            *ptr2 = (*buffer);
            buffer++;

            buffer++; //MSB is zero
        }
    }
}

/* **************************************************************************** */
void capture_camera_img(void)
{
    camera_start_capture_image();
    while (1) {
        if (camera_is_image_rcv()) {
            return;
        }
    }
}

/* **************************************************************************** */
void convert_img_unsigned_to_signed(uint32_t *data0, uint32_t *data1, uint32_t *data2)
{
    uint8_t *ptr0;
    uint8_t *ptr1;
    uint8_t *ptr2;
    ptr0 = (uint8_t *)data0;
    ptr1 = (uint8_t *)data1;
    ptr2 = (uint8_t *)data2;
    for (int i = 0; i < 4096; i++) {
        *ptr0 = unsigned_to_signed(*ptr0);
        ptr0++;
        *ptr1 = unsigned_to_signed(*ptr1);
        ptr1++;
        *ptr2 = unsigned_to_signed(*ptr2);
        ptr2++;
    }
}

/* **************************************************************************** */
void convert_img_signed_to_unsigned(uint32_t *data0, uint32_t *data1, uint32_t *data2)
{
    uint8_t *ptr0;
    uint8_t *ptr1;
    uint8_t *ptr2;
    ptr0 = (uint8_t *)data0;
    ptr1 = (uint8_t *)data1;
    ptr2 = (uint8_t *)data2;
    for (int i = 0; i < 4096; i++) {
        *ptr0 = signed_to_unsigned(*ptr0);
        ptr0++;
        *ptr1 = signed_to_unsigned(*ptr1);
        ptr1++;
        *ptr2 = signed_to_unsigned(*ptr2);
        ptr2++;
    }
}

int main(void)
{
    int i, dma_channel;
    int digs, tens;
    int ret = 0;

    MXC_Delay(200000);
    Camera_Power(POWER_ON);
    printf("\n\nMNIST Demo\n");

    MXC_ICC_Enable(MXC_ICC0);

    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    printf("Waiting...\n");

    // do not remove!!!
    MXC_Delay(SEC(2)); 

    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);

    // CNN boost (P2.5)
    mxc_gpio_cfg_t gpio_out;
    gpio_out.port = MXC_GPIO2;
    gpio_out.mask = MXC_GPIO_PIN_5;
    gpio_out.pad = MXC_GPIO_PAD_NONE;
    gpio_out.func = MXC_GPIO_FUNC_OUT;
    gpio_out.vssel = MXC_GPIO_VSSEL_VDDIO;
    gpio_out.drvstr = MXC_GPIO_DRVSTR_0;
    MXC_GPIO_Config(&gpio_out);
    MXC_GPIO_OutSet(gpio_out.port, gpio_out.mask);

#ifdef TFT_ENABLE
    printf("Init LCD.\n");
#endif
    MXC_TFT_Init(MXC_SPI0, 1, NULL, NULL);
    MXC_TFT_SetRotation(ROTATE_270);
    MXC_TFT_SetForeGroundColor(WHITE);
    MXC_Delay(1000000);

    MXC_DMA_Init();
    dma_channel = MXC_DMA_AcquireChannel();

    camera_init(CAMERA_FREQ);

    ret = camera_setup(IMAGE_SIZE_X, IMAGE_SIZE_Y, PIXFORMAT_RGB888, FIFO_THREE_BYTE, USE_DMA,
                       dma_channel);
    if (ret != STATUS_OK) {
        printf("Error returned from setting up camera. Error %d\n", ret);
        return -1;
    }

#ifdef TFT_ENABLE
    MXC_TFT_SetBackGroundColor(4);
    memset(buff, ' ', TFT_BUFF_SIZE);
#endif

    while (1) {
        printf("********** PB1(SW1) to capture an image **********\r\n");
        while (!PB_Get(0)) {}

#ifdef TFT_ENABLE
        MXC_TFT_ClearScreen();
#endif

        capture_camera_img();

        process_camera_img(input_0_camera, input_1_camera, input_2_camera);

#ifdef TFT_ENABLE
        MXC_TFT_ClearScreen();
        TFT_Print(buff, 10, 30, font_2, snprintf(buff, sizeof(buff), "Camera (64x64)"));
        lcd_show_sampledata(input_0_camera, input_1_camera, input_2_camera, 25, 85, 1024);
#endif

        cnn_init();
        cnn_load_weights();
        cnn_load_bias();
        cnn_configure();

        #if USE_CAMERA_INPUT
        capture_camera_img();   // get a fresh frame first
        #endif

        load_input();    
        cnn_start();          
        SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk; // SLEEPDEEP=0
        while (cnn_time == 0) __WFI();
        softmax_layer();

        printf("Approx CNN time: %d us\n\n", cnn_time);

        cnn_disable();

        int best_i = 0;
        q15_t best_v = ml_softmax[0];
        printf("Outs (w softmax):\n");
        for (i = 0; i < CNN_NUM_OUTPUTS; i++) {
            digs = (1000 * ml_softmax[i] + 0x4000) >> 15;
            tens = digs % 10;
            digs = digs / 10;
            printf("[%7d] -> Class %d: %d.%d%%\n", ml_data[i], i, digs, tens);
        }

#ifdef TFT_ENABLE
        TFT_Print(buff, 5, 5, font_2, snprintf(buff, sizeof(buff), "Pred: %s", classes[best_i]));
        TFT_Print(buff, 5, 210, font_2, snprintf(buff, sizeof(buff), "PRESS PB1(SW1) TO CAPTURE"));
#endif
    }

    return 0;
}
