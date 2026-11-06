cmake_minimum_required(VERSION 3.18)
project(icp_draft LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_library(icp_core io.cpp icp_gold.cpp icp_solver.cpp icp_run.cpp)
target_include_directories(icp_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
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
target_sources(icp_core PRIVATE cuda_memory.cu transform_kernel.cu correspondence_kernel.cu statistics_kernel.cu)
target_compile_definitions(icp_core PUBLIC ICP_HAS_CUDA=1)
target_link_libraries(icp_core PUBLIC CUDA::cudart)
add_executable(icp_run main.cpp)
target_link_libraries(icp_run PRIVATE icp_core)
enable_testing()
add_executable(icp_tests tests/test_draft.cpp)
target_link_libraries(icp_tests PRIVATE icp_core)
add_test(NAME draft_host COMMAND icp_tests)
add_test(NAME draft_cuda_memory COMMAND icp_tests --gpu)
set_tests_properties(draft_cuda_memory PROPERTIES SKIP_RETURN_CODE 77)
