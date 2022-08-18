#ifndef EDGE_PROJECT_STEREO_LINE_H_
#define EDGE_PROJECT_STEREO_LINE_H_

#include <Eigen/Core> 
#include <Eigen/Geometry> 
#include <opencv2/core/eigen.hpp>
#include <g2o/core/base_binary_edge.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/sba/g2o_types_sba_api.h>
#include <g2o/types/sba/vertex_se3_expmap.h>
#include <g2o/types/slam3d_addons/line3d.h>

#include "utils.h"


class EdgeStereoSE3ProjectLine
    : public BaseBinaryEdge<4, Vector8d, VertexLine3D, VertexSE3Expmap> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeStereoSE3ProjectLine();

  bool read(std::istream &is);
  bool write(std::ostream &os) const;
  void computeError();

  Eigen::Vector6d cam_project(const g2o::Line3D &line) const;

  double fx, fy, b;
  Eigen::Vector3d Kv; // [-cx*fy, -fx*cy, fx*fy]
};

#endif  // EDGE_PROJECT_STEREO_LINE_H_
