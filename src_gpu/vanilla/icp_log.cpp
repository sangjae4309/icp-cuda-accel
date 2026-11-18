#include "icp_log.h"
#include <cerrno>
#include <cstdlib>
#include <sys/stat.h>

IcpStatus openIterationLog(const char* label, IterationLog* log)
{
    if (mkdir("scripts/result", 0755) != 0 && errno != EEXIST) {
        return ICP_IO_ERROR;
    }
    std::snprintf(log->directory, sizeof(log->directory),
        "scripts/result/%s-XXXXXX", label);
    if (!mkdtemp(log->directory)) {
        return ICP_IO_ERROR;
    }

    char path[320];
    std::snprintf(path, sizeof(path), "%s/iterations.csv", log->directory);
    log->timings = std::fopen(path, "w");
    if (!log->timings) {
        return ICP_IO_ERROR;
    }
    std::fprintf(log->timings,
        "iteration,time_ms,converged,"
        "r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz\n");
    std::printf("iteration_results=%s\n", log->directory);
    std::fflush(stdout);
    return ICP_OK;
}

IcpStatus recordIteration(IterationLog* log, int iteration, double time_ms,
    const Pose* pose, bool converged)
{
    std::printf("iteration=%d time_ms=%.3f converged=%d\n", iteration, time_ms,
        int(converged));
    std::fflush(stdout);
    std::fprintf(
        log->timings, "%d,%.9f,%d", iteration, time_ms, int(converged));
    for (int i = 0; i < 9; ++i) {
        std::fprintf(log->timings, ",%.17g", pose->R[i]);
    }
    for (int i = 0; i < 3; ++i) {
        std::fprintf(log->timings, ",%.17g", pose->t[i]);
    }
    std::fprintf(log->timings, "\n");
    return std::fflush(log->timings) == 0 ? ICP_OK : ICP_IO_ERROR;
}

IcpStatus closeIterationLog(IterationLog* log)
{
    return std::fclose(log->timings) == 0 ? ICP_OK : ICP_IO_ERROR;
}
