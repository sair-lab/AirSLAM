#ifndef EDGE_PROJECT_POINT_H_
#define EDGE_PROJECT_POINT_H_

#include <Eigen/Core> 
#include <Eigen/Geometry> 
#include <opencv2/core/eigen.hpp>
#include <g2o/core/base_binary_edge.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/sba/g2o_types_sba_api.h>
#include <g2o/types/sba/vertex_se3_expmap.h>
#include <g2o/types/slam3d_addons/line3d.h>

#include "utils.h"
#include "g2o_optimization/vertex_vi_pose.h"

class EdgeSE3ProjectPoint
    : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ, VertexVIPose> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectPoint();

  bool read(std::istream &is);
  bool write(std::ostream &os) const;
  void computeError();
  bool isDepthPositive();

  Eigen::Vector2d cam_project(const Eigen::Vector3d& point) const;
  // Eigen::Matrix<double, 2, 3> cam_projection_jacobian(const Eigen::Vector3d &point) const;
  // virtual void linearizeOplus();

  double fx, fy, cx, cy;
};

class EdgeSE3ProjectStereoPoint
    : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, g2o::VertexPointXYZ, VertexVIPose> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectStereoPoint();

  bool read(std::istream &is);
  bool write(std::ostream &os) const;
  void computeError();
  bool isDepthPositive();

  Eigen::Vector3d cam_project(const Eigen::Vector3d& point) const;
  // Eigen::Matrix<double, 2, 3> cam_projection_jacobian(const Eigen::Vector3d &point) const;
  // virtual void linearizeOplus();
  
  double fx, fy, cx, cy, bf;
};

#endif  // EDGE_PROJECT_POINT_H_
