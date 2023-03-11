# Main makefile for VLC 3rd party libraries ("contrib")
# Copyright (C) 2003-2011 the VideoLAN team
#
# This file is under the same license as the vlc package.

all: .ffmpeg

# bootstrap configuration
include config.mak

PATH :=$(abspath ../../extras/tools/build/bin):$(PATH)
export PATH

#
# Machine-dependent variables
#

PREFIX ?= ./$(HOST)
PREFIX := $(abspath $(PREFIX))
ifneq ($(HOST),$(BUILD))
HAVE_CROSS_COMPILE = 1
endif
ARCH := $(shell ./get-arch.sh $(HOST))

ifeq ($(ARCH)-$(HAVE_WIN32),x86_64-1)
HAVE_WIN64 := 1
endif

need_pkg = 1

#
# Default values for tools
#
ifndef HAVE_CROSS_COMPILE
ifneq ($(findstring $(origin CC),undefined default),)
CC := gcc
endif
ifneq ($(findstring $(origin CXX),undefined default),)
CXX := g++
endif
ifneq ($(findstring $(origin LD),undefined default),)
LD := ld
endif
ifneq ($(findstring $(origin AR),undefined default),)
AR := ar
endif
ifneq ($(findstring $(origin RANLIB),undefined default),)
RANLIB := ranlib
endif
ifneq ($(findstring $(origin STRIP),undefined default),)
STRIP := strip
endif
else
ifneq ($(findstring $(origin CC),undefined default),)
CC := $(HOST)-gcc
endif
ifneq ($(findstring $(origin CXX),undefined default),)
CXX := $(HOST)-g++
endif
ifneq ($(findstring $(origin LD),undefined default),)
LD := $(HOST)-ld
endif
ifneq ($(findstring $(origin AR),undefined default),)
AR := $(HOST)-ar
endif
ifneq ($(findstring $(origin RANLIB),undefined default),)
RANLIB := $(HOST)-ranlib
endif
ifneq ($(findstring $(origin STRIP),undefined default),)
STRIP := $(HOST)-strip
endif
endif

ifdef HAVE_ANDROID
CC :=  $(HOST)-gcc --sysroot=$(ANDROID_NDK)/platforms/android-9/arch-$(PLATFORM_SHORT_ARCH)
CXX := $(HOST)-g++ --sysroot=$(ANDROID_NDK)/platforms/android-9/arch-$(PLATFORM_SHORT_ARCH)
endif

ifdef HAVE_MACOSX
MIN_OSX_VERSION=10.6
CC=xcrun llvm-gcc-4.2
CXX=xcrun llvm-g++-4.2
AR=xcrun ar
LD=xcrun ld
STRIP=xcrun strip
RANLIB=xcrun ranlib
EXTRA_CFLAGS += -isysroot $(MACOSX_SDK) -mmacosx-version-min=$(MIN_OSX_VERSION) -DMACOSX_DEPLOYMENT_TARGET=$(MIN_OSX_VERSION)
EXTRA_LDFLAGS += -Wl,-syslibroot,$(MACOSX_SDK) -mmacosx-version-min=$(MIN_OSX_VERSION) -isysroot $(MACOSX_SDK) -DMACOSX_DEPLOYMENT_TARGET=$(MIN_OSX_VERSION)
ifeq ($(ARCH),x86_64)
EXTRA_CFLAGS += -m64
EXTRA_LDFLAGS += -m64
else
EXTRA_CFLAGS += -m32
EXTRA_LDFLAGS += -m32
endif

XCODE_FLAGS = -sdk macosx$(OSX_VERSION)
ifeq ($(shell xcodebuild -version 2>/dev/null | tee /dev/null|head -1|cut -d\  -f2|cut -d. -f1),3)
XCODE_FLAGS += ARCHS=$(ARCH)
# XCode 3 doesn't support -arch
else
XCODE_FLAGS += -arch $(ARCH)
endif

endif

ifdef HAVE_IOS
CC=xcrun clang
CXX=xcrun clang++
ifeq ($(ARCH), arm)
AS=perl $(abspath ../../extras/tools/build/bin/gas-preprocessor.pl) $(CC)
else
AS=xcrun as
endif
AR=xcrun ar
LD=xcrun ld
STRIP=xcrun strip
RANLIB=xcrun ranlib
EXTRA_CFLAGS += -isysroot $(SDKROOT)  -miphoneos-version-min=5.0
EXTRA_LDFLAGS += -Wl,-syslibroot,$(SDKROOT) -isysroot $(SDKROOT) -miphoneos-version-min=5.0
endif

ifdef HAVE_WIN32
ifneq ($(shell $(CC) $(CFLAGS) -E -dM -include _mingw.h - < /dev/null | grep -E __MINGW64_VERSION_MAJOR),)
HAVE_MINGW_W64 := 1
endif
endif

cppcheck = $(shell $(CC) $(CFLAGS) -E -dM - < /dev/null | grep -E $(1))

EXTRA_CFLAGS += -I$(PREFIX)/include
CPPFLAGS := $(CPPFLAGS) $(EXTRA_CFLAGS)
CFLAGS := $(CFLAGS) $(EXTRA_CFLAGS) -g
CXXFLAGS := $(CXXFLAGS) $(EXTRA_CFLAGS) -g
EXTRA_LDFLAGS += -L$(PREFIX)/lib
LDFLAGS := $(LDFLAGS) $(EXTRA_LDFLAGS)
# Do not export those! Use HOSTVARS.

# Do the FPU detection, after we have figured out our compilers and flags.
ifneq ($(findstring $(ARCH),aarch64 i386 ppc ppc64 sparc sparc64 x86_64),)
# This should be consistent with include/vlc_cpu.h
HAVE_FPU = 1
else ifneq ($(findstring $(ARCH),arm),)
ifneq ($(call cppcheck, __VFP_FP__)),)
ifeq ($(call cppcheck, __SOFTFP__),)
HAVE_FPU = 1
endif
endif
else ifneq ($(call cppcheck, __mips_hard_float),)
HAVE_FPU = 1
endif

ACLOCAL_AMFLAGS += -I$(PREFIX)/share/aclocal
export ACLOCAL_AMFLAGS

ifdef HAVE_CROSS_COMPILE
# This inhibits .pc file from within the cross-compilation toolchain sysroot.
# Hopefully, nobody ever needs that.
PKG_CONFIG_PATH := /usr/share/pkgconfig
PKG_CONFIG_LIBDIR := /usr/$(HOST)/lib/pkgconfig
export PKG_CONFIG_LIBDIR
endif
PKG_CONFIG_PATH := $(PKG_CONFIG_PATH):$(PREFIX)/lib/pkgconfig
export PKG_CONFIG_PATH

ifndef GIT
ifeq ($(shell git --version >/dev/null 2>&1 || echo FAIL),)
GIT = git
endif
endif
GIT ?= $(error git not found!)

ifndef SVN
ifeq ($(shell svn --version >/dev/null 2>&1 || echo FAIL),)
SVN = svn
endif
endif
SVN ?= $(error subversion client (svn) not found!)

ifeq ($(shell curl --version >/dev/null 2>&1 || echo FAIL),)
download = curl -f -L -- "$(1)" > "$@"
else ifeq ($(shell wget --version >/dev/null 2>&1 || echo FAIL),)
download = rm -f $@.tmp && \
	wget --passive -c -p -O $@.tmp "$(1)" && \
	touch $@.tmp && \
	mv $@.tmp $@
else ifeq ($(which fetch >/dev/null 2>&1 || echo FAIL),)
download = rm -f $@.tmp && \
	fetch -p -o $@.tmp "$(1)" && \
	touch $@.tmp && \
	mv $@.tmp $@
else
download = $(error Neither curl nor wget found!)
endif

ifeq ($(shell gzcat --version >/dev/null 2>&1 || echo FAIL),)
ZCAT = gzcat
else ifeq ($(shell zcat --version >/dev/null 2>&1 || echo FAIL),)
ZCAT = zcat
else
ZCAT ?= $(error Gunzip client (zcat) not found!)
endif

#
# Common helpers
#
HOSTCONF := --prefix="$(PREFIX)"
HOSTCONF += --build="$(BUILD)" --host="$(HOST)" --target="$(HOST)"
HOSTCONF += --program-prefix=""
# libtool stuff:
HOSTCONF += --enable-static --disable-shared --disable-dependency-tracking
ifdef HAVE_WIN32
HOSTCONF += --without-pic
PIC :=
else
HOSTCONF += --with-pic
PIC := -fPIC
endif

HOSTTOOLS := \
	CC="$(CC)" CXX="$(CXX)" LD="$(LD)" \
	AR="$(AR)" RANLIB="$(RANLIB)" STRIP="$(STRIP)" \
	PATH="$(PREFIX)/bin:$(PATH)"
HOSTVARS := $(HOSTTOOLS) \
	CPPFLAGS="$(CPPFLAGS)" \
	CFLAGS="$(CFLAGS)" \
	CXXFLAGS="$(CXXFLAGS)" \
	LDFLAGS="$(LDFLAGS)"
HOSTVARS_PIC := $(HOSTTOOLS) \
	CPPFLAGS="$(CPPFLAGS) $(PIC)" \
	CFLAGS="$(CFLAGS) $(PIC)" \
	CXXFLAGS="$(CXXFLAGS) $(PIC)" \
	LDFLAGS="$(LDFLAGS)"

download_git = \
	rm -Rf $(@:.tar.xz=) && \
	$(GIT) clone $(2:%=--branch %) $(1) $(@:.tar.xz=) && \
	(cd $(@:.tar.xz=) && $(GIT) checkout $(3:%= %)) && \
	rm -Rf $(@:%.tar.xz=%)/.git && \
	(cd $(dir $@) && \
	tar cvJ $(notdir $(@:.tar.xz=))) > $@ && \
	rm -Rf $(@:.tar.xz=)
UNPACK = $(RM) -R $@ \
	$(foreach f,$(filter %.tar.gz %.tgz,$^), && tar xvzf $(f)) \
	$(foreach f,$(filter %.tar.bz2,$^), && tar xvjf $(f)) \
	$(foreach f,$(filter %.tar.xz,$^), && tar xvJf $(f)) \
	$(foreach f,$(filter %.zip,$^), && unzip $(f))
UNPACK_DIR = $(basename $(basename $(notdir $<)))
MOVE = mv $(UNPACK_DIR) $@ && touch $@

fetch: ffmpeg

clean:
	-$(RM) -R "$(PREFIX)"
	-$(RM) -R ffmpeg/

distclean: clean
	$(RM) config.mak

.PHONY: all fetch clean distclean

.SECONDEXPANSION:

.DELETE_ON_ERROR:

# FFmpeg

HASH=HEAD

#FFMPEG_SNAPURL := http://git.videolan.org/?p=ffmpeg.git;a=snapshot;h=$(HASH);sf=tgz
FFMPEG_SNAPURL := https://git.libav.org/?p=libav.git;a=snapshot;h=$(HASH);sf=tgz

FFMPEGCONF = \
	--cc="$(CC)" \
	--disable-doc \
	--disable-debug \
	--disable-avdevice \
	--disable-devices \
	--disable-avfilter \
	--disable-filters \
	--disable-bsfs \
	--disable-bzlib \
    --disable-avresample \
    --disable-network \
    --disable-encoders \
    --disable-muxers \

# Those tools are named differently in FFmpeg and Libav
#	--disable-ffserver \
#	--disable-ffplay \
#	--disable-ffprobe

ifdef HAVE_CROSS_COMPILE
FFMPEGCONF += --enable-cross-compile
ifndef HAVE_DARWIN_OS
FFMPEGCONF += --cross-prefix=$(HOST)-
endif
endif

# x86 stuff
ifeq ($(ARCH),i386)
ifndef HAVE_DARWIN_OS
FFMPEGCONF += --arch=x86
endif
endif

# Darwin
ifdef HAVE_DARWIN_OS
FFMPEGCONF += --arch=$(ARCH) --target-os=darwin
ifneq ($(findstring $(ARCH),i386 x86_64),)
FFMPEGCONF += --enable-memalign-hack
endif
ifeq ($(ARCH),x86_64)
FFMPEGCONF += --cpu=core2
endif
endif
ifdef HAVE_MACOSX
FFMPEGCONF += --enable-vda
endif

# Linux
ifdef HAVE_LINUX
FFMPEGCONF += --target-os=linux --enable-pic

endif

# Windows
ifdef HAVE_WIN32
ifndef HAVE_MINGW_W64
endif
FFMPEGCONF += --target-os=mingw32 --enable-memalign-hack
FFMPEGCONF += --enable-w32threads

ifdef HAVE_WIN64
FFMPEGCONF += --cpu=athlon64 --arch=x86_64
else # !WIN64
FFMPEGCONF+= --cpu=i686 --arch=x86
endif

else # !Windows
FFMPEGCONF += --enable-pthreads
endif

ffmpeg-$(HASH).tar.gz:
	$(call download,$(FFMPEG_SNAPURL))

ffmpeg: ffmpeg-$(HASH).tar.gz
	rm -Rf $@ $@-$(HASH)
	mkdir -p $@-$(HASH)
	$(ZCAT) "$<" | (cd $@-$(HASH) && tar xvf - --strip-components=1)
	$(MOVE)

.ffmpeg: ffmpeg
	cd $< && $(HOSTVARS) ./configure \
		--extra-ldflags="$(LDFLAGS)" $(FFMPEGCONF) \
		--prefix="$(PREFIX)" --enable-static --disable-shared
	cd $< && $(MAKE) install-libs install-headers
	touch $@
