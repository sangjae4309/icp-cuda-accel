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
	@echo 'make kdtree-gpu-test                   GPU step 2 only: KD-tree kernel vs naive kernel'

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

NVCC ?= nvcc
CUDA_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '. ')
KDGPU_BIN := build-kdtree-gpu/icp_kdtree_gpu_test
KDGPU_SRCS := src_gpu/vanilla/icp_kdtree_gpu_test.cpp \
	src_gpu/vanilla/icp_kdtree_build.cpp \
	src_gpu/common/kitti_parser.cpp src_gpu/common/io.cpp \
	src_gpu/common/pose.cpp \
	src_gpu/vanilla/icp_utils.cu \
	src_gpu/vanilla/icp_step2_correspondence.cu \
	src_gpu/vanilla/icp_step2_kdtree.cu \
	src_gpu/vanilla/icp_step2_voxel_grid.cu
KDGPU_HDRS := $(KDTREE_HDRS) src_gpu/vanilla/icp.h src_gpu/vanilla/icp_utils.h

.PHONY: kdtree-test kdtree-gpu-test
kdtree-gpu-test: $(KDGPU_BIN)
	./$(KDGPU_BIN) $(SOURCE) $(TARGET) $(KDGPU_ARGS)

$(KDGPU_BIN): $(KDGPU_SRCS) $(KDGPU_HDRS)
	@test -n "$(CUDA_ARCH)" || { echo 'Set CUDA_ARCH, e.g. make kdtree-gpu-test CUDA_ARCH=86' >&2; exit 1; }
	mkdir -p $(@D)
	$(NVCC) -O3 -std=c++17 -fmad=false -Xcompiler=-ffp-contract=off,-Wall,-Wextra \
		-gencode=arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
		-Isrc_gpu/common -Isrc_gpu/vanilla $(KDGPU_SRCS) -o $@