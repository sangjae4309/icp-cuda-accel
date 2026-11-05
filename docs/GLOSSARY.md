# ICP Skeleton

Vocabulary for registration of two point clouds.

## Language

**Original source**:
The unchanged point cloud whose coordinates are mapped into the target frame.
_Avoid_: Updated source

**Transformed source**:
The original source expressed under the current accumulated pose.

**Accumulated pose**:
The transformation from the original source frame into the target frame.
_Avoid_: Delta pose

**Incremental delta pose**:
The rigid update from the currently transformed source to its matched targets, composed on the left of the accumulated pose.

**Correspondence**:
A nearest-target association for one transformed source point. Multiple source points may share the same target point.

**Matched-target centroid**:
The mean of target points selected by valid correspondences, counting repeated selections separately.

**Final-pose evaluation**:
Evaluation of correspondences and residuals at the final accumulated pose.

**Dry run**:
An infrastructure check that produces no registration estimate.
