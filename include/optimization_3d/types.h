
#ifndef OPTIMIZATION_3D_TYPES_H_
#define OPTIMIZATION_3D_TYPES_H_

#include <istream>
#include <map>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"

struct Pose3d {
  Eigen::Vector3d p;
  Eigen::Quaterniond q;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef std::map<int,
                 Pose3d,
                 std::less<int>,
                 Eigen::aligned_allocator<std::pair<const int, Pose3d>>>
    MapOfPoses;

struct Position3d{
  Eigen::Vector3d p;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef std::map<int,
                 Position3d,
                 std::less<int>,
                 Eigen::aligned_allocator<std::pair<const int, Pose3d>>>
    MapOfPoints3d;

struct PointConstraint {
  int id_pose;
  int id_point;
  int id_camera;

  // x_left, y_left, x_right
  Eigen::Vector3d keypoint;  
  double pixel_sigma;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef std::vector<PointConstraint, Eigen::aligned_allocator<PointConstraint>>
    VectorOfPointConstraints;

#endif  // OPTIMIZATION_3D_TYPES_H_
