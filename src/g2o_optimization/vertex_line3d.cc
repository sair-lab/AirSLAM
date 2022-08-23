#include <Eigen/Core> 
#include <Eigen/Geometry> 
#include <opencv2/core/eigen.hpp>
#include <g2o/core/base_binary_edge.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/sba/g2o_types_sba_api.h>
#include <g2o/types/sba/vertex_se3_expmap.h>
#include <g2o/types/slam3d_addons/line3d.h>

#include "utils.h"
#include "g2o_optimization/vertex_line3d.h"

VertexLine3D::VertexLine3D() : g2o::BaseVertex<4, g2o::Line3D>(), color(1., 0.5, 0.) {}

bool VertexLine3D::read(std::istream& is) {
  Vector6d lv;
  bool state = g2o::internal::readVector(is, lv);
  setEstimate(g2o::Line3D(lv));
  return state;
}

bool VertexLine3D::write(std::ostream& os) const {
  return g2o::internal::writeVector(os, _estimate);
}