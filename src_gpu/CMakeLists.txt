cmake_minimum_required(VERSION 3.18)
project(icp_cuda LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../third_party/eigen"
    "${CMAKE_CURRENT_BINARY_DIR}/eigen")
add_library(icp_core
    common/io.cpp
    common/kitti_parser.cpp
    common/pose.cpp
    vanilla/icp_step4_solve.cpp
    vanilla/icp_step5_update.cpp
    vanilla/icp_log.cpp
    vanilla/icp_kdtree_build.cpp
    vanilla/icp_run.cpp)
target_link_libraries(icp_core PRIVATE Eigen3::Eigen)
target_include_directories(icp_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common ${CMAKE_CURRENT_SOURCE_DIR}/vanilla)
target_compile_options(icp_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
# CPU/GPU parity: both compilers fuse multiply-add by default, which changes results by a
# ULP of the accumulation type. The transform accumulates in double and rounds once to
# float, so contraction is invisible there (verified both ways). It is NOT invisible where
# the accumulation type is the output type: correspondence (NN-1) computes dist2 in float
# and must match bit-for-bit, ties included. Pin contraction off for both languages so that
# parity is a property of the build rather than of how much slack the rounding happens to
# leave. Drop these two lines only together with the bit-exact assertions in the tests.
target_compile_options(icp_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
  $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "Pass -DCMAKE_CUDA_ARCHITECTURES=<actual GPU architecture>")
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set_target_properties(icp_core PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_sources(icp_core PRIVATE
    vanilla/icp_utils.cu
    vanilla/icp_step1_transform.cu
    vanilla/icp_step2_correspondence.cu
    vanilla/icp_step2_kdtree.cu
    vanilla/icp_step2_voxel_grid.cu
    vanilla/icp_step3_statistics.cu)
target_compile_definitions(icp_core PUBLIC ICP_HAS_CUDA=1)
target_link_libraries(icp_core PUBLIC CUDA::cudart)
add_executable(icp_run main.cpp)
target_link_libraries(icp_run PRIVATE icp_core)
