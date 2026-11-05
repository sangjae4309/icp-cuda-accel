#include "icp.h"
IcpStatus cudaAvailability() { return ICP_UNAVAILABLE; }
IcpStatus preallocDeviceWorkspace(DeviceWorkspace*, int, int) { return ICP_UNAVAILABLE; }
void deallocDeviceWorkspace(DeviceWorkspace* w) { if(w) *w={}; }
IcpStatus uploadPointClouds(DeviceWorkspace*, const Point*, const Point*) { return ICP_UNAVAILABLE; }
IcpStatus uploadPose(DeviceWorkspace*, const Pose*) { return ICP_UNAVAILABLE; }
IcpStatus downloadStatistics(const DeviceWorkspace*, IcpStatistics*) { return ICP_UNAVAILABLE; }
IcpStatus transformPointsCUDA(DeviceWorkspace*) { return ICP_UNAVAILABLE; }
IcpStatus findCorrespondencesCUDA(DeviceWorkspace*, IcpBackend, float) { return ICP_UNAVAILABLE; }
IcpStatus computeStatisticsCUDA(DeviceWorkspace*) { return ICP_UNAVAILABLE; }
