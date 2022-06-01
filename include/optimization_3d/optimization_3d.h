#ifndef OPTIMIZATION_3D_H_
#define OPTIMIZATION_3D_H_

#include <vector>
#include <ceres/ceres.h>

#include "camera.h"
#include "optimization_3d/types.h"

int AddVisualErrorTerm(MapOfPoses& poses, MapOfPoints3d& points, std::vector<CameraPtr>& camera_list, 
    VectorOfPointConstraints& visual_constraints, ceres::Problem* problem,
    ceres::LocalParameterization* quaternion_local_parameterization);
void BuildOptimizationProblem(MapOfPoses& poses, MapOfPoints3d& points, std::vector<CameraPtr>& camera_list,
    VectorOfPointConstraints& visual_constraints, std::vector<int>& fixed_poses, 
    std::vector<int>& fixed_points, ceres::Problem* problem);
bool SolveOptimizationProblem(ceres::Problem* problem);
int Optimize(MapOfPoses& poses, MapOfPoints3d& points, std::vector<CameraPtr> camera_list,
    VectorOfPointConstraints& visual_constraints, std::vector<int>& fixed_poses, 
    std::vector<int>& fixed_points, std::vector<int>& inliers);

int SolvePnPWithCV();

#endif  // OPTIMIZATION_3D_H_