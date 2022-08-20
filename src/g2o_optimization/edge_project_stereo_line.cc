#include "g2o_optimization/edge_project_stereo_line.h"
#include "utils.h"

EdgeStereoSE3ProjectLine::EdgeStereoSE3ProjectLine()
    : BaseBinaryEdge<4, Vector8d, g2o::VertexLine3D, g2o::VertexSE3Expmap>() {}

bool EdgeStereoSE3ProjectLine::read(std::istream &is) {
  internal::readVector(is, _measurement);
  return readInformationMatrix(is);
}

bool EdgeStereoSE3ProjectLine::write(std::ostream &os) const {
  internal::writeVector(os, measurement());
  return writeInformationMatrix(os);
}

void EdgeStereoSE3ProjectLine::computeError() {
  const VertexSE3Expmap *v1 =
      static_cast<const VertexSE3Expmap *>(_vertices[1]);
  const VertexLine3D *v2 = static_cast<const VertexLine3D *>(_vertices[0]);
  Vector8d obs(_measurement);
  Eigen::Vector4d error;

  g2o::Isometry3 T_left = v1->estimate().Isometry3();
  Eigen::Vector3d line_2d_left = cam_project(T_left * v2->estimate());
  double line_2d_norm_left = line_2d_left.head(2).norm();
  error(0) = (obs(0) * line_2d_left(0) + obs(1) * line_2d_left(1) + line_2d_left(2)) / line_2d_norm_left;
  error(1) = (obs(2) * line_2d_left(0) + obs(3) * line_2d_left(1) + line_2d_left(2)) / line_2d_norm_left;

  g2o::Isometry3 T_right(T_left);
  T_right(0, 3) -= b;
  Eigen::Vector3d line_2d_right = cam_project(T_right * v2->estimate());
  double line_2d_norm_right = line_2d_right.head(2).norm();
  error(2) = (obs(5) * line_2d_right(0) + obs(6) * line_2d_right(1) + line_2d_right(2)) / line_2d_norm_right;
  error(3) = (obs(7) * line_2d_right(0) + obs(8) * line_2d_right(1) + line_2d_right(2)) / line_2d_norm_right;
  _error = error;
}

Eigen::Vector6d EdgeStereoSE3ProjectLine::cam_project(const g2o::Line3D& line) const {
  Eigen::vector3d w = line.w();
  Eigen::Vector3d line_2d;
  line_2d(0) = fy * w(0);
  line_2d(1) = fx * w(1);
  line_2d(2) = Kv.transpose() * w;
  return line_2d;
}