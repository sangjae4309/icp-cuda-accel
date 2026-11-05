#include "icp.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <climits>
#include <cstdint>

void identityPose(Pose* p) { if (p) { *p = {}; p->R[0] = p->R[4] = p->R[8] = 1; } }
void defaultOptions(IcpOptions* o) {
    if (!o) return;
    *o = {}; o->backend = BACKEND_CPU; o->max_iterations = 20;
    o->max_distance = 1; o->translation_tolerance = o->rotation_tolerance = 1e-6;
    identityPose(&o->initial_pose);
}
const char* statusName(IcpStatus s) {
    switch (s) {
#define STATUS(x) case x: return #x
    STATUS(ICP_OK); STATUS(ICP_NOT_IMPLEMENTED); STATUS(ICP_INVALID_INPUT);
    STATUS(ICP_IO_ERROR); STATUS(ICP_CUDA_ERROR); STATUS(ICP_INSUFFICIENT_MATCHES);
    STATUS(ICP_DEGENERATE); STATUS(ICP_UNAVAILABLE); STATUS(ICP_ALLOCATION_ERROR);
#undef STATUS
    default: return "UNKNOWN_STATUS";
    }
}
IcpStatus parseBackend(const char* name, IcpBackend* b) {
    if (!name || !b) return ICP_INVALID_INPUT;
    if (!std::strcmp(name, "cpu")) *b = BACKEND_CPU;
    else if (!std::strcmp(name, "cuda-naive")) *b = BACKEND_CUDA_NAIVE;
    else if (!std::strcmp(name, "cuda-tiled")) *b = BACKEND_CUDA_TILED;
    else return ICP_INVALID_INPUT;
    return ICP_OK;
}
IcpStatus validateCloud(const PointCloud* c) {
    if (!c || !c->points_h || c->count <= 0) return ICP_INVALID_INPUT;
    for (int i = 0; i < c->count; ++i)
        if (!std::isfinite(c->points_h[i].x) || !std::isfinite(c->points_h[i].y) ||
            !std::isfinite(c->points_h[i].z)) return ICP_INVALID_INPUT;
    return ICP_OK;
}
void freePointCloud(PointCloud* c) { if (c) { std::free(c->points_h); *c = {}; } }
IcpStatus loadXYZ(const char* path, PointCloud* c) {
    if (!path || !c || c->points_h || c->count) return ICP_INVALID_INPUT;
    FILE* f = std::fopen(path, "r");
    if (!f) return ICP_IO_ERROR;
    char* line = nullptr; size_t line_capacity = 0, capacity = 0;
    IcpStatus s = ICP_OK;
    while (getline(&line, &line_capacity, f) >= 0) {
        char* p = line;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
        if (!*p || *p == '#') continue;
        Point v{};
        if (std::sscanf(p, "%f %f %f", &v.x, &v.y, &v.z) != 3 ||
            !std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
            s = ICP_INVALID_INPUT; break;
        }
        if (c->count == INT_MAX) { s = ICP_INVALID_INPUT; break; }
        if (size_t(c->count) == capacity) {
            size_t next = capacity ? capacity * 2 : 256;
            if (next > SIZE_MAX / sizeof(Point)) { s = ICP_ALLOCATION_ERROR; break; }
            void* allocation = std::realloc(c->points_h, next * sizeof(Point));
            if (!allocation) { s = ICP_ALLOCATION_ERROR; break; }
            c->points_h = static_cast<Point*>(allocation); capacity = next;
        }
        c->points_h[c->count++] = v;
    }
    if (std::ferror(f)) s = ICP_IO_ERROR;
    std::free(line);
    if (std::fclose(f) != 0) s = ICP_IO_ERROR;
    if (s == ICP_OK) s = validateCloud(c);
    if (s != ICP_OK) freePointCloud(c);
    return s;
}
