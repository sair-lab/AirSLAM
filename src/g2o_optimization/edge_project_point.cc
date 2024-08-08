#include <Eigen/Core> 
#include <Eigen/Geometry> 
#include <opencv2/core/eigen.hpp>
#include <g2o/types/slam3d/isometry3d_mappings.h>

#include "utils.h"
#include "g2o_optimization/edge_project_point.h"

// monocular point
EdgeSE3ProjectPoint::EdgeSE3ProjectPoint()
    : g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexPointXYZ, VertexVIPose>() {}

bool EdgeSE3ProjectPoint::read(std::istream &is) {
  g2o::internal::readVector(is, _measurement);
  return readInformationMatrix(is);
}

bool EdgeSE3ProjectPoint::write(std::ostream &os) const {
  g2o::internal::writeVector(os, measurement());
  return writeInformationMatrix(os);
}

void EdgeSE3ProjectPoint::computeError() {
  const g2o::VertexPointXYZ *v1 =
      static_cast<const g2o::VertexPointXYZ *>(_vertices[0]);
  const VertexVIPose *v2 = static_cast<const VertexVIPose *>(_vertices[1]);
  Eigen::Vector2d obs(_measurement);
  _error = obs - cam_project(v2->estimate().Rcw * v1->estimate() + v2->estimate().tcw);
}

bool EdgeSE3ProjectPoint::isDepthPositive(){
  const g2o::VertexPointXYZ *v1 =
      static_cast<const g2o::VertexPointXYZ *>(_vertices[0]);
  const VertexVIPose *v2 = static_cast<const VertexVIPose *>(_vertices[1]);
  return (v2->estimate().Rcw * v1->estimate() + v2->estimate().tcw)(2) > 0;
}

Eigen::Vector2d EdgeSE3ProjectPoint::cam_project(const Eigen::Vector3d& point) const {
  double z_inv = 1.0 / point(2);
  Eigen::Vector2d point_2d;
  point_2d(0) = point(0) * z_inv * fx + cx;
  point_2d(1) = point(1) * z_inv * fy + cy;
  return point_2d;
}

// Eigen::Matrix<double, 2, 3> EdgeSE3ProjectPoint::cam_projection_jacobian(const Eigen::Vector3d &point) const {
//   Eigen::Matrix<double, 2, 3> jac;
//   jac(0, 0) = fx / point[2];
//   jac(0, 1) = 0.;
//   jac(0, 2) = -fx * point[0] / (point[2] * point[2]);
//   jac(1, 0) = 0.;
//   jac(1, 1) = fy / point[2];
//   jac(1, 2) = -fy * point[1] / (point[2] * point[2]);
//   return jac;
// }

// void EdgeSE3ProjectPoint::linearizeOplus(){

//   const g2o::VertexPointXYZ *v1 =
//       static_cast<const g2o::VertexPointXYZ *>(_vertices[0]);
//   const VertexVIPose *v2 = static_cast<const VertexVIPose *>(_vertices[1]);

//   const Eigen::Matrix3d& Rcw = v2->estimate().Rcw;
//   const Eigen::Matrix3d& Rcb = v2->estimate().Rcb;
//   const Eigen::Vector3d& tcw = v2->estimate().tcw;

//   const Eigen::Vector3d Xc = Rcw * v1->estimate() + tcw;
//   const Eigen::Vector3d Xb = v2->estimate().Rbc * Xc + v2->estimate().tbc;

//   const Eigen::Matrix<double,2,3> projection_jacobian = cam_projection_jacobian(Xc);
//   _jacobianOplusXi = -projection_jacobian * Rcw;

//   Eigen::Matrix<double,3,6> SE3deriv;
//   double x = Xb(0);
//   double y = Xb(1);
//   double z = Xb(2);

//   SE3deriv <<  0.0,   z,  -y, 1.0, 0.0, 0.0,
//                 -z , 0.0,   x, 0.0, 1.0, 0.0,
//                 y ,  -x, 0.0, 0.0, 0.0, 1.0;

//   _jacobianOplusXj = projection_jacobian * Rcb * SE3deriv;
// }


// stereo point
EdgeSE3ProjectStereoPoint::EdgeSE3ProjectStereoPoint()
    : g2o::BaseBinaryEdge<3, Eigen::Vector3d, g2o::VertexPointXYZ, VertexVIPose>() {}

bool EdgeSE3ProjectStereoPoint::read(std::istream &is) {
  g2o::internal::readVector(is, _measurement);
  return readInformationMatrix(is);
}

bool EdgeSE3ProjectStereoPoint::write(std::ostream &os) const {
  g2o::internal::writeVector(os, measurement());
  return writeInformationMatrix(os);
}

void EdgeSE3ProjectStereoPoint::computeError() {
  const g2o::VertexPointXYZ *v1 =
      static_cast<const g2o::VertexPointXYZ *>(_vertices[0]);
  const VertexVIPose *v2 = static_cast<const VertexVIPose *>(_vertices[1]);
  Eigen::Vector3d obs(_measurement);
  _error = obs - cam_project(v2->estimate().Rcw * v1->estimate() + v2->estimate().tcw);
}

bool EdgeSE3ProjectStereoPoint::isDepthPositive(){
  const g2o::VertexPointXYZ *v1 =
      static_cast<const g2o::VertexPointXYZ *>(_vertices[0]);
  const VertexVIPose *v2 = static_cast<const VertexVIPose *>(_vertices[1]);
  return (v2->estimate().Rcw * v1->estimate() + v2->estimate().tcw)(2) > 0;
}

Eigen::Vector3d EdgeSE3ProjectStereoPoint::cam_project(const Eigen::Vector3d& point) const {
  double z_inv = 1.0 / point(2);
  Eigen::Vector3d point_2d;
  point_2d(0) = point(0) * z_inv * fx + cx;
  point_2d(1) = point(1) * z_inv * fy + cy;
  point_2d(2) = point_2d(0) - bf * z_inv;
  return point_2d;
}

// Eigen::Matrix<double, 2, 3> EdgeSE3ProjectStereoPoint::cam_projection_jacobian(const Eigen::Vector3d &point) const {
//   Eigen::Matrix<double, 2, 3> jac;
//   jac(0, 0) = fx / point[2];
//   jac(0, 1) = 0.;
//   jac(0, 2) = -fx * point[0] / (point[2] * point[2]);
//   jac(1, 0) = 0.;
//   jac(1, 1) = fy / point[2];
//   jac(1, 2) = -fy * point[1] / (point[2] * point[2]);
//   return jac;
// }

// void EdgeSE3ProjectStereoPoint::linearizeOplus(){

//   const g2o::VertexPointXYZ *v1 =
//       static_cast<const g2o::VertexPointXYZ *>(_vertices[0]);
//   const VertexVIPose *v2 = static_cast<const VertexVIPose *>(_vertices[1]);

//   const Eigen::Matrix3d& Rcw = v2->estimate().Rcw;
//   const Eigen::Matrix3d& Rcb = v2->estimate().Rcb;
//   const Eigen::Vector3d& tcw = v2->estimate().tcw;

//   const Eigen::Vector3d Xc = Rcw * v1->estimate() + tcw;
//   const Eigen::Vector3d Xb = v2->estimate().Rbc * Xc + v2->estimate().tbc;
//   const double inv_z2 = 1.0 / (Xc(2) * Xc(2));


//   Eigen::Matrix<double,3,3> projection_jacobian;
//   projection_jacobian.block<2,3>(0,0) = cam_projection_jacobian(Xc);
//   projection_jacobian.block<1,3>(2,0) = projection_jacobian.block<1,3>(0,0);
//   projection_jacobian(2,2) += bf * inv_z2;
//   _jacobianOplusXi = -projection_jacobian * Rcw;

//   Eigen::Matrix<double,3,6> SE3deriv;
//   double x = Xb(0);
//   double y = Xb(1);
//   double z = Xb(2);

//   SE3deriv <<  0.0,   z,  -y, 1.0, 0.0, 0.0,
//                 -z , 0.0,   x, 0.0, 1.0, 0.0,
//                 y ,  -x, 0.0, 0.0, 0.0, 1.0;

//   _jacobianOplusXj = projection_jacobian * Rcb * SE3deriv;
// }