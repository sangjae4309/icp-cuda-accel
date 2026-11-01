#include "icp/icp.hpp"
#include <stdexcept>

namespace icp {

bool cuda_available() {
    return false;
}

std::unique_ptr<Backend> make_cuda(const Cloud&, const Cloud&, double, bool) {
    throw std::runtime_error("CUDA backend not built. Configure with -DICP_ENABLE_CUDA=ON and an explicit architecture.");
}
Result run_cuda(const Cloud&, const Cloud&, const Options&, bool) {
    throw std::runtime_error("CUDA backend not built. Configure with -DICP_ENABLE_CUDA=ON and an explicit architecture.");
}
}
