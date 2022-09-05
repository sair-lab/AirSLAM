#ifndef G2O_OPTIMIZATION_H_
#define G2O_OPTIMIZATION_H_

#include <vector>

#include "read_configs.h"
#include "camera.h"
#include "frame.h"
#include "mappoint.h"
#include "g2o_optimization/types.h"

// void LocalmapOptimization(MapOfPoses& poses, MapOfPoints3d& points, std::vector<CameraPtr>& camera_list, 
    // VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints);

void LocalmapOptimization(MapOfPoses& poses, MapOfPoints3d& points, MapOfLine3d& lines, std::vector<CameraPtr>& camera_list, 
    VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints, 
    VectorOfMonoLineConstraints& mono_line_constraints, VectorOfStereoLineConstraints& stereo_line_constraints,
    const OptimizationConfig& cfg);

int FrameOptimization(MapOfPoses& poses, MapOfPoints3d& points, std::vector<CameraPtr>& camera_list, 
    VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints,
    const OptimizationConfig& cfg);

int SolvePnPWithCV(FramePtr frame, std::vector<MappointPtr>& mappoints, Eigen::Matrix4d& pose, std::vector<int>& inliers);

#endif  // G2O_OPTIMIZATION_H_