#pragma once
#include "../common/point_cloud.h"
#include <cstdio>

struct IterationLog
{
    char directory[256];
    FILE* timings;
};

IcpStatus openIterationLog(const char* label, IterationLog* log);
IcpStatus recordIteration(IterationLog* log, int iteration, double time_ms,
    const Pose* pose, bool converged);
IcpStatus closeIterationLog(IterationLog* log);

IcpStatus dumpIterationPoints(const IterationLog* log, const PointCloud* cloud,
    const Pose* pose, const char* name);
