# Project-owned wiring for the unmodified Eigen 3.4.0 headers.
add_library(icp_eigen INTERFACE)
target_include_directories(icp_eigen SYSTEM INTERFACE
    "${CMAKE_CURRENT_LIST_DIR}")
add_library(Eigen3::Eigen ALIAS icp_eigen)
