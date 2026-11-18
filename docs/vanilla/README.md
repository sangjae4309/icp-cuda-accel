# Vanilla point-to-point ICP

CPU source is under `src_cpu/vanilla/`, CUDA source under `src_gpu/vanilla/`.
Common types, XYZ I/O and pose operations live under each implementation's common/.
CPU stages have separate numbered files; CUDA retains its existing source names.
Contributor function bodies were preserved during the CPU split. CPU/CUDA trees
remain independent.

See [five-step design](../VANILLA_DESIGN.md),
[Smith's CUDA optimizations](../VANILLA_OPTIMIZATIONS.md), and
[CPU pair validation](../CPU_PAIR_VALIDATION.md). Historical source paths in old
validation records describe their original revisions. The layout change reproduced
the 21-iteration CPU result exactly; CUDA has not been rebuilt after relocation.


CPU is a fixed target: its options/header contain no backend selector, CUDA
workspace or CUDA function declarations. CPU invocation is
`build-cpu/icp_run source.xyz target.xyz [vanilla|feature]`.
The CPU CUDA-unavailable stub and its CMake source entry were removed.
The CUDA tree retains its own backend options; CPU/GPU arithmetic contracts
remain the same, but their host structs are no longer interchangeable.
Rebuild CPU libraries and executable together after this interface cleanup;
no build or registration was run during cleanup.


CPU intermediate-buffer type and allocation/free helpers are declared in
`src_cpu/vanilla/icp_utils.h` and implemented in `icp_utils.cpp`.
`icp_step1_transform.cpp`, `icp_step2_correspondence.cpp` and
`icp_step3_statistics.cpp` contain the first three stages;
`icp_run.cpp` includes the utility header directly. CMake includes the new source.


## Five CPU block interfaces

`src_cpu/vanilla/icp.h` declares one function boundary per PDF stage:

| PDF stage | Function | Output |
| --- | --- | --- |
| 1. Transform | transformPointsCPU | transformed source points |
| 2. Correspondence | findCorrespondencesCPU | target indices and squared distances |
| 3. Statistics | computeStatisticsCPU | IcpStatistics |
| 4. Rigid solve | solveRigidUpdateCPU | incremental Pose |
| 5. Update | updatePoseCPU | next Pose and convergence flag |

The runner invokes these in order and feeds the updated pose into the next
iteration. Step 5 groups left composition and incremental-motion convergence;
its implementation is in icp_step5_update.cpp and reuses common pose math.
The SVD implementation is in icp_step4_solve.cpp. Steps 1-3
are reevaluated after the loop so the published error uses the final pose.
Buffer lifetime helpers remain in icp_utils.h/.cpp. These are software function
boundaries for block reasoning; they do not claim synthesized hardware blocks.


```text
src_cpu/vanilla/
    icp.h                          # five stage interfaces
    icp_run.cpp                    # iteration orchestration
    icp_step1_transform.cpp         # transformPointsCPU
    icp_step2_correspondence.cpp    # findCorrespondencesCPU
    icp_step3_statistics.cpp        # computeStatisticsCPU
    icp_step4_solve.cpp             # solveRigidUpdateCPU
    icp_step5_update.cpp            # updatePoseCPU
    icp_utils.h / icp_utils.cpp     # intermediate buffer lifetime
```

The old CPU icp_gold.cpp and icp_solver.cpp were removed after exact function-body
relocation. Their paths in historical validation describe the earlier layout.

## CUDA stage files and iteration recording

```text
src_gpu/vanilla/
    icp.h                         # five stage interfaces
    icp_run.cpp                   # CUDA-only iteration orchestration
    icp_step1_transform.cu         # original transform kernel and launch
    icp_step2_correspondence.cu    # original naive/tiled kernels and launches
    icp_step3_statistics.cu        # two-pass statistics kernels and launch
    icp_step4_solve.cpp            # host SVD
    icp_step5_update.cpp           # host composition and convergence
    icp_utils.h / icp_utils.cu     # device storage, transfers, CUDA events
    icp_log.h / icp_log.cpp        # host-side iteration CSV logging
```

CPU retains the corresponding numbered .cpp stages and its own icp_log files.
The GPU uses CudaIcpBuffers with allocateIcpBuffers/freeIcpBuffers; its selectable
search method is independent of execution target. There is no CPU fallback or
duplicate CPU gold source in the CUDA library. Device arithmetic, barriers,
launch geometry, precision flags and error checks were preserved. Renaming these
files does not create a different GPU architecture or demonstrate a speedup.

The host interval from upload through kernels and statistics download is bounded
by CUDA events in the default stream. The statistics download completes device
work before host SVD. One step-3 interface still launches four kernels; the file
boundary does not imply a single kernel. No separable device compilation is
needed because device helpers and their callers remain in the same .cu file.

Both vanilla runners log every completed iteration into a unique directory under
scripts/result/. CPU optionally dumps a fixed local set of point coordinates;
GPU logs CUDA-event times and poses only. See the root README for make dump
commands, the exact timing scopes and 3 m selection. The new source/ABI/layout
needs rebuilding; no CPU/GPU build or run was performed for this change.


## Subsequent execution validation

The refactored CPU/CUDA libraries, binary loaders and iteration timing paths were
subsequently built and run on an L40S. See
[CPU/CUDA accuracy and latency](../CPU_CUDA_COMPARISON.md) for matched 20-iteration
conditions, measured results and evidence. CPU point dumps were disabled and
remain runtime unverified. Earlier "not built/run" statements above describe
the source-only cleanup at the time, not this later experiment.
