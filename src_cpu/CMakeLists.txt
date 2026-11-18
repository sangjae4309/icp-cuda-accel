cmake_minimum_required(VERSION 3.18)
project(icp_cpu LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
add_library(icp_core
    common/io.cpp
    common/kitti_parser.cpp
    common/pose.cpp
    vanilla/icp_utils.cpp
    vanilla/icp_step1_transform.cpp
    vanilla/icp_step2_correspondence.cpp
    vanilla/icp_step3_statistics.cpp
    vanilla/icp_step4_solve.cpp
    vanilla/icp_step5_update.cpp
    vanilla/icp_log.cpp
    vanilla/icp_run.cpp)
target_link_libraries(icp_core PRIVATE Eigen3::Eigen)
target_include_directories(icp_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common ${CMAKE_CURRENT_SOURCE_DIR}/vanilla)
target_compile_options(icp_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
# Must match src_gpu: the host gold has to produce bit-identical results in both projects,
# and the CUDA side pins -fmad=false for the same reason. See src_gpu/CMakeLists.txt.
target_compile_options(icp_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
add_library(icp_feature feature/preprocessing.cpp feature/correspondence.cpp feature/residual.cpp
    feature/optimizer.cpp feature/icp.cpp)
target_link_libraries(icp_feature PUBLIC icp_core Eigen3::Eigen)
target_compile_options(icp_feature PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
add_executable(icp_run main.cpp)
target_link_libraries(icp_run PRIVATE icp_feature)
