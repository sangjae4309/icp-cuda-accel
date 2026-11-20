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
	@echo 'make gpu SEARCH=all RUN=1              Run naive and tiled searches'
	@echo 'make gpu SEARCH=kd-tree RUN=1          FILL ME (not implemented)'
	@echo 'make gpu SEARCH=voxel-grid RUN=1       CUDA voxel-grid nearest neighbor'
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
