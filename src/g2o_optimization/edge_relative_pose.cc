#include <Eigen/Core> 
#include <Eigen/Geometry> 
#include <opencv2/core/eigen.hpp>
#include <g2o/types/slam3d/isometry3d_mappings.h>

#include "utils.h"
#include "imu.h"
#include "g2o_optimization/vertex_vi_pose.h"
#include "g2o_optimization/edge_relative_pose.h"

// monocular point
EdgeRelativePose::EdgeRelativePose()
    : g2o::BaseBinaryEdge<6, Vector6d, VertexVIPose, VertexVIPose>() {
}

void EdgeRelativePose::computeError() {
  const VertexVIPose *v1 = static_cast<const VertexVIPose *>(_vertices[0]);
  const VertexVIPose *v2 = static_cast<const VertexVIPose *>(_vertices[1]);

  Eigen::Vector3d er;
  SO3Log(Rc1c2 * v2->estimate().Rcw * v1->estimate().Rcw.transpose(), er);
  Eigen::Vector3d ep = (v1->estimate().tcw - v1->estimate().Rcw * v2->estimate().Rcw.transpose() * v2->estimate().tcw) - tc1c2;

  _error << er, ep;
}