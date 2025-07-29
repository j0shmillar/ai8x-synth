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
#include "spi.h"
#include "mxc.h"
#include "mxc_device.h"
#include "board.h"
#include "rtc.h"
#include "uart.h"
#include "tft_utils.h"
#include "softmax.h"

#define TFT_X_OFFSET 50

#define IMAGE_SIZE_X 74
#define IMAGE_SIZE_Y 74
#define IMAGE_SCALE 3
#define CAMERA_SIZE_X (IMAGE_SCALE * IMAGE_SIZE_X)
#define CAMERA_SIZE_Y (IMAGE_SCALE * IMAGE_SIZE_Y)

// TODO - rm?
#ifdef BOARD_EVKIT_V1
#include "bitmap.h"
#include "tft_ssd2119.h"
#endif
#ifdef BOARD_FTHR_REVA
#include "tft_ili9341.h"
#endif
#include "example_config.h"

#ifdef BOARD_EVKIT_V1 // TODO - rm?
int font = urw_gothic_12_grey_bg_white;
#endif
#ifdef BOARD_FTHR_REVA
int font = (int)&Liberation_Sans16x16[0];
#endif
volatile uint32_t cnn_time; // Stopwatch

#if defined(RGB565) && defined(BOARD_EVKIT_V1)
uint8_t data565[CAMERA_SIZE_X * 2];
#endif

extern int g_dma_channel_tft;
void fail(void)
{
    printf("\n*** FAIL ***\n\n");

    while (1) {}
}

static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

void softmax_layer(void)
{
    printf("and here\n");
    cnn_unload((uint32_t *)ml_data);
    printf("and here...?\n");
    // TODO - print shape (should be 10 & not 810)
    softmax_q17p14_q15((const q31_t *)ml_data, CNN_NUM_OUTPUTS, ml_softmax);
}

#ifdef RGB565
#define RESIZED_WIDTH 28
#define RESIZED_HEIGHT 28

void load_input_RGB565(void)
{
    static stream_stat_t *stat;
    uint8_t *buffer;
    uint32_t imgLen, w, h;
    uint32_t *cnn_mem = (uint32_t *)0x50402000;

    union {
        uint32_t w;
        uint8_t b[4];
    } m;

    camera_start_capture_image();
    camera_get_image(&buffer, &imgLen, &w, &h);

    float scale_x = (float)w / RESIZED_WIDTH;
    float scale_y = (float)h / RESIZED_HEIGHT;

    for (uint32_t y_out = 0; y_out < RESIZED_HEIGHT; y_out++) {
        uint32_t y_in = (uint32_t)(y_out * scale_y);
        for (uint32_t x_out = 0; x_out < RESIZED_WIDTH; x_out++) {
            uint32_t x_in = (uint32_t)(x_out * scale_x);

            uint32_t index = (y_in * w + x_in) * 2;
            uint8_t byte1 = buffer[index];
            uint8_t byte2 = buffer[index + 1];

            // RGB565 to 24-bit RGB
            m.b[0] = byte1 & 0xF8;                            // R (5 bits)
            m.b[1] = ((byte1 & 0x07) << 5) | ((byte2 & 0xE0) >> 3); // G (6 bits)
            m.b[2] = (byte2 & 0x1F) << 3;                     // B (5 bits)

            // pack and norm
            *cnn_mem++ = m.w ^ 0x00808080U;
        }
    }

    stat = get_camera_stream_statistic();
    if (stat->overflow_count > 0) {
        printf("OVERFLOW CNN = %d\n", stat->overflow_count);
        LED_On(LED2); // red LED if overflow
        while (1) {}
    }
}
#endif

void display_camera_RGB565(void)
{
    static stream_stat_t *stat;

    uint8_t *frame_buffer = NULL;
    uint8_t *buffer;
    uint32_t imgLen;
    uint32_t w, h, y;

    // display
    camera_start_capture_image();

    camera_get_image(&buffer, &imgLen, &w, &h);

    printf("W:%d H:%d L:%d \n", w, h, imgLen);

#ifdef BOARD_FTHR_REVA
    // init FTHR TFT for DMA streaming
    MXC_TFT_Stream(TFT_X_OFFSET, 0, w, h);
#endif

    for (y = 0; y < h; y++) {
        while ((frame_buffer = get_camera_stream_buffer()) == NULL) {
            if (camera_is_image_rcv()) {
                break;
            }
        };

#ifdef BOARD_EVKIT_V1
        int j = 0;
        for (int k = 2 * w - 1; k > 0; k -= 2) { // flip order to display

            data565[j++] = frame_buffer[k + 1];
            data565[j++] = frame_buffer[k];
        }

        MXC_TFT_ShowImageCameraRGB565(TFT_X_OFFSET, y, data565, w, 1);
#endif
#ifdef BOARD_FTHR_REVA
        tft_dma_display(TFT_X_OFFSET, y, w, 1, (uint32_t *)frame_buffer);
#endif

        release_camera_stream_buffer();
    }

    stat = get_camera_stream_statistic();
    if (stat->overflow_count > 0) {
        printf("OVERFLOW DISP = %d\n", stat->overflow_count);
        LED_On(LED2); // red LED if overflow
        while (1) {}
    }
}

void cnn_wait(void)
{
    while ((*((volatile uint32_t *)0x50100000) & (1 << 12)) != 1 << 12) {}

    CNN_COMPLETE;
    cnn_time = MXC_TMR_SW_Stop(MXC_TMR0);
}

uint32_t utils_get_time_ms(void)
{
    uint32_t sec, ssec;
    double subsec;
    uint32_t ms;
    MXC_RTC_GetSubSeconds(&ssec);
    subsec = ssec / 4096.0;
    MXC_RTC_GetSeconds(&sec);

    ms = (sec * 1000) + (int)(subsec * 1000);

    return ms;
}

int main(void)
{
#ifdef TFT_ENABLE
    char buff[TFT_BUFF_SIZE];
#endif
    static uint32_t t1, t2, t3, t4, t5, t6;

#if defined(BOARD_FTHR_REVA)
    // wait for pmic 1.8V to become available, approx 180ms after power up
    MXC_Delay(200000);
    Camera_Power(POWER_ON);
#endif

    MXC_ICC_Enable(MXC_ICC0);

    // switch to 100 MHz clock
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    printf("Waiting...\n");

    // init RTC
    MXC_RTC_Init(0, 0);
    MXC_RTC_Start();

    // DO NOT RM THIS LINE:
    MXC_Delay(SEC(2)); 

    // config P2.5, BOOST
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
    printf("LCD init...");

#ifdef BOARD_EVKIT_V1
    MXC_TFT_Init();
#endif

#ifdef BOARD_FTHR_REVA
    MXC_TFT_Init(MXC_SPI0, 1, NULL, NULL);
    MXC_TFT_SetRotation(ROTATE_270);
    MXC_TFT_SetForeGroundColor(WHITE); 
#endif

    memset(buff, 32, TFT_BUFF_SIZE);
    TFT_Print(buff, 80, 30, font, snprintf(buff, sizeof(buff), "ADI             "));
    TFT_Print(buff, 55, 50, font, snprintf(buff, sizeof(buff), "Demo      "));
    TFT_Print(buff, 120, 90, font, snprintf(buff, sizeof(buff), "Ver. 1.1.0                   "));
    MXC_Delay(SEC(2));

#ifdef BOARD_EVKIT_V1
    MXC_TFT_SetBackGroundColor(255);
#endif

#endif 

    int xx = IMAGE_SIZE_X;
    int yy = IMAGE_SIZE_Y;
    printf("x %d  y %d\n", xx, yy);

    int dma_channel;
    printf("Init Camera...\n");

    MXC_DMA_Init();
    dma_channel = MXC_DMA_AcquireChannel();

    camera_init(CAMERA_FREQ);

#ifndef RGB565
    int ret = camera_setup(IMAGE_SIZE_X, IMAGE_SIZE_Y, PIXFORMAT_RGB888, FIFO_THREE_BYTE, USE_DMA,
                           dma_channel);
#else
    int ret = camera_setup(CAMERA_SIZE_X, CAMERA_SIZE_Y, PIXFORMAT_RGB565, FIFO_FOUR_BYTE,
                           STREAMING_DMA, dma_channel);
    // set camera clock prescaler to prevent streaming overflow due to TFT display latency

#ifdef BOARD_EVKIT_V1
    camera_write_reg(0x11, 0x3);
#endif

#ifdef BOARD_FTHR_REVA
    camera_write_reg(0x11, 0x0);
#endif

#endif

    if (ret != STATUS_OK) {
        printf("\tError in camera set up: %d\n", ret);
        return -1;
    }

    // NN clock: 50 MHz div 1
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_init();
    cnn_load_weights();

#ifdef TFT_ENABLE
    MXC_TFT_ClearScreen();
#endif

    while (1) {

        printf("Proc...\n");
        t1 = utils_get_time_ms();

        cnn_init(); 
        cnn_load_bias();
        cnn_configure(); 

        load_input_RGB565();

        t2 = utils_get_time_ms();

        LED_On(LED1);

        cnn_start(); 

#if defined(TFT_ENABLE) && defined(RGB565)
        display_camera_RGB565();
#endif

        t3 = utils_get_time_ms();

        while (cnn_time == 0) {
            __WFI(); 
        }

        t4 = utils_get_time_ms();

        LED_Off(LED1);

        t5 = utils_get_time_ms();

        printf("here\n");
        softmax_layer();
        printf("but not here\n");

        cnn_disable();

        t6 = utils_get_time_ms();

        printf("CNN time: %d us\n", cnn_time);

        printf("load:%d TFT:%d cnn_wait:%d cnn_unload:%d pproc:%d Total:%dms\n", t2 - t1,
               t3 - t2, t4 - t3, t5 - t4, t6 - t5, t6 - t1);
        MXC_Delay(SEC(1));

        printf("Out:\n");
        int digs, tens;
        for (int i = 0; i < CNN_NUM_OUTPUTS; i++) {
            digs = (1000 * ml_softmax[i] + 0x4000) >> 15;
            tens = digs % 10;
            digs = digs / 10;
            printf("[%7d] -> class %d: %d.%d%%\n", ml_data[i], i, digs, tens);
        }

    }

    return 0;
}
