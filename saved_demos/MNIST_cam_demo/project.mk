# https://analogdevicesinc.github.io/msdk/USERGUIDE/#build-system

#MXC_OPTIMIZE_CFLAGS = -Og

# **********************************************************

BOARD=FTHR_RevA

$(info Note: This project is designed and tested for the OV7692 only.)
override CAMERA=OV7692

MXC_OPTIMIZE_CFLAGS = -O2

ifeq "$(BOARD)" "EvKit_V1"
VPATH += TFT/evkit/resources
endif
ifeq "$(BOARD)" "FTHR_RevA"
FONTS = LiberationSans16x16
endif

IPATH += TFT/evkit/resources

ifeq ($(BOARD),Aud01_RevA)
$(error ERR_NOTSUPPORTED: This project is not supported for the Audio board)
endif

ifeq ($(BOARD),CAM01_RevA)
$(error ERR_NOTSUPPORTED: This project is not supported for the CAM01 board)
endif

ifeq ($(BOARD),CAM02_RevA)
$(error ERR_NOTSUPPORTED: This project is not supported for the CAM02 board)
endif


