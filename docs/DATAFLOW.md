# Draft dataflow and ownership

All paths below refer to `src_cpu/` and `src_gpu/`. N is source count, M is target count,
B=(size_t(N)+BLOCK_SIZE-1)/BLOCK_SIZE. No N*M matrix is allocated.

| Buffer | Location / size | Owner | Writer | Reader | Allocation / free | H2D | D2H |
|---|---|---|---|---|---|---|---|
| source.points_h | Host, N Points | main | XYZ loader | CPU transform / uploader | loadXYZ / freePointCloud | Once before loop | None |
| target.points_h | Host, M Points | main | XYZ loader | CPU matching/statistics / uploader | loadXYZ / freePointCloud | Once before loop | None |
| transformed_h | Host, N Points | HostWorkspace | CPU transform TODO | CPU matching/statistics TODO | preallocHostWorkspace / deallocHostWorkspace | None | None |
| match_index_h | Host, N ints | HostWorkspace | CPU matching TODO | CPU statistics TODO | Same host workspace lifetime | None | None |
| match_dist2_h | Host, N floats | HostWorkspace | CPU matching TODO | CPU statistics TODO | Same host workspace lifetime | None | None |
| source_d | Device, N Points | DeviceWorkspace | uploadPointClouds | CUDA transform TODO | preallocDeviceWorkspace / deallocDeviceWorkspace | Once before loop | Tests only |
| target_d | Device, M Points | DeviceWorkspace | uploadPointClouds | CUDA matching/statistics TODO | Same device workspace lifetime | Once before loop | Tests only |
| transformed_d | Device, N Points | DeviceWorkspace | CUDA transform TODO | CUDA matching/statistics TODO | Same device workspace lifetime | None | None |
| match_index_d | Device, N ints | DeviceWorkspace | CUDA matching TODO | CUDA statistics TODO | Same device workspace lifetime | None | None |
| match_dist2_d | Device, N floats | DeviceWorkspace | CUDA matching TODO | CUDA statistics TODO | Same device workspace lifetime | None | None |
| pose_d | Device, 1 Pose | DeviceWorkspace | uploadPose | CUDA transform TODO | Same device workspace lifetime | Every iteration | Tests only |
| stats_d | Device, 1 IcpStatistics | DeviceWorkspace | CUDA statistics TODO | downloadStatistics | Same device workspace lifetime | Test sentinel only | Every iteration |
| centroid_partial_d | Device, B CentroidPartials | DeviceWorkspace | Pass A TODO | Centroid finalization TODO | Same device workspace lifetime | None | None |
| covariance_partial_d | Device, B CovariancePartials | DeviceWorkspace | Pass B TODO | Covariance finalization TODO | Same device workspace lifetime | None | None |
| current | Host, 1 Pose | runICP stack | Initial options / composition TODO | uploadPose / CPU transform TODO | Function entry / exit | Every GPU iteration | None |
| delta, next | Host, 2 Poses | runICP stack | Solver / composition TODO | Composition / convergence TODO | Iteration entry / exit | None | None |
| stats | Host, 1 IcpStatistics | runICP stack | CPU statistics TODO or download | Solver TODO | Iteration entry / exit | None | Every GPU iteration |

Original source is immutable. Transformed source is recomputed from the original
under the current accumulated pose, never by applying the accumulated pose again
to already-transformed points. Original source and target remain device-resident.

The accumulated pose maps original source to target. Incremental delta maps the
currently transformed source toward matched targets. Composition is LEFT:
`R_next=delta_R*R_current`, `t_next=delta_R*t_current+delta_t`.
Current and next occupy separate objects.

Planned GPU loop:

1. Upload current pose; transform original source.
2. Find matches against all M target points, with many-to-one matches allowed.
3. Pass A: reduce valid count, source/target sums and SSE; finalize centroids.
4. Pass B: reduce centered cross-covariance H using those centroids.
5. Download only stats; solve delta on CPU; compose and check convergence.
6. Repeat; then re-evaluate at the final pose before publishing results.

Every TODO wrapper stops execution with `ICP_NOT_IMPLEMENTED`; later stages and
final evaluation are connected but not yet executable. Test-only D2H copies do not
belong in the normal loop. Matched-target sums count duplicate selections, and
threshold rejection uses index=-1. Exact float distance ties choose the lowest index.
