#include <Eigen/Core> 
#include <Eigen/Geometry> 
#include <opencv2/core/eigen.hpp>
#include <g2o/types/slam3d/isometry3d_mappings.h>

#include "g2o_optimization/edge_project_line.h"

EdgeSE3ProjectLine::EdgeSE3ProjectLine()
    : g2o::BaseBinaryEdge<2, Eigen::Vector4d, VertexLine3D, g2o::VertexSE3Expmap>() {}

bool EdgeSE3ProjectLine::read(std::istream &is) {
  g2o::internal::readVector(is, _measurement);
  return readInformationMatrix(is);
}

bool EdgeSE3ProjectLine::write(std::ostream &os) const {
  g2o::internal::writeVector(os, measurement());
  return writeInformationMatrix(os);
}

void EdgeSE3ProjectLine::computeError() {
  const g2o::VertexSE3Expmap *v1 =
      static_cast<const g2o::VertexSE3Expmap *>(_vertices[1]);
  const VertexLine3D *v2 = static_cast<const VertexLine3D *>(_vertices[0]);
  Eigen::Vector4d obs(_measurement);
  Eigen::Vector3d line_2d = cam_project(g2o::internal::fromSE3Quat(v1->estimate()) * v2->estimate());

  double line_2d_norm = line_2d.head(2).norm();
  Eigen::Vector2d error;
  error(0) = obs(0) * line_2d(0) + obs(1) * line_2d(1) + line_2d(2);
  error(1) = obs(2) * line_2d(0) + obs(3) * line_2d(1) + line_2d(2);
  _error = error / line_2d_norm;
}

Eigen::Vector3d EdgeSE3ProjectLine::cam_project(const g2o::Line3D& line) const {
  Eigen::Vector3d w = line.w();
  Eigen::Vector3d line_2d;
  line_2d(0) = fy * w(0);
  line_2d(1) = fx * w(1);
  line_2d(2) = Kv.transpose() * w;
  return line_2d;
}