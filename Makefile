.DEFAULT_GOAL := help
.PHONY: help cpu gpu cuda dump
METHOD ?= vanilla
SEARCH ?= all
RUN ?= $(if $(filter dump,$(MAKECMDGOALS)),1,0)
ICP_DUMP := $(if $(filter dump,$(MAKECMDGOALS)),1,0)
SOURCE ?= data/000001.bin
TARGET ?= data/000000.bin
export METHOD SEARCH RUN SOURCE TARGET ICP_DUMP

help:
	@echo 'make cpu RUN=1                         CPU vanilla'
	@echo 'make gpu SEARCH=brute-force RUN=1      CUDA naive brute force'
	@echo 'make gpu SEARCH=tiled RUN=1            CUDA shared-memory tiled search'
	@echo 'make gpu SEARCH=all RUN=1              Run both CUDA searches'
	@echo 'make gpu SEARCH=kd-tree RUN=1          FILL ME (not implemented)'
	@echo 'make gpu SEARCH=voxel-grid RUN=1       FILL ME (not implemented)'
	@echo 'make kdtree-test [STRIDE=1]            Host KD-tree vs brute force (no GPU)'
	@echo 'make cpu dump RUN=1                    CPU iteration point CSV dumps'
	@echo 'Without RUN=1, only compile. Override SOURCE and TARGET as needed.'

cpu:
	bash tests/run_cpu.sh

gpu cuda:
	bash tests/run_gpu.sh

# `make dump` runs CPU vanilla; combined goals add dumping to that target.
dump:
ifeq ($(filter cpu gpu cuda,$(MAKECMDGOALS)),)
	bash tests/run_cpu.sh
else
	@:
endif

KDTREE_BIN := build-kdtree/icp_kdtree_test
KDTREE_SRCS := src_gpu/vanilla/icp_kdtree_test.cpp \
	src_gpu/vanilla/icp_kdtree_build.cpp \
	src_gpu/common/kitti_parser.cpp src_gpu/common/io.cpp \
	src_gpu/common/pose.cpp
KDTREE_HDRS := src_gpu/vanilla/icp_kdtree.h src_gpu/common/point_cloud.h \
	src_gpu/common/kitti_parser.h

kdtree-test: $(KDTREE_BIN)
	./$(KDTREE_BIN) $(SOURCE) $(TARGET) $(STRIDE)

$(KDTREE_BIN): $(KDTREE_SRCS) $(KDTREE_HDRS)
	mkdir -p $(@D)
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Wpedantic -ffp-contract=off \
		-Isrc_gpu/common -Isrc_gpu/vanilla $(KDTREE_SRCS) -o $@