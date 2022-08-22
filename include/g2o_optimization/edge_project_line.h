#ifndef EDGE_PROJECT_LINE_H_
#define EDGE_PROJECT_LINE_H_

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

class EdgeSE3ProjectLine
    : public g2o::BaseBinaryEdge<2, Eigen::Vector4d, VertexLine3D, g2o::VertexSE3Expmap> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectLine();

  bool read(std::istream &is);
  bool write(std::ostream &os) const;
  void computeError();

  Eigen::Vector3d cam_project(const g2o::Line3D &line) const;

  double fx, fy;
  Eigen::Vector3d Kv; // [-cx*fy, -fx*cy, fx*fy]
};

#endif  // EDGE_PROJECT_LINE_H_
