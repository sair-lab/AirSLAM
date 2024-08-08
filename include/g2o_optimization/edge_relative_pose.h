#ifndef EDGE_RELATIVE_POSE_H_
#define EDGE_RELATIVE_POSE_H_

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

class EdgeRelativePose
    : public g2o::BaseBinaryEdge<6, Vector6d, VertexVIPose, VertexVIPose> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeRelativePose();

  virtual bool read(std::istream& is){return false;}
  virtual bool write(std::ostream& os) const{return false;}
  void computeError();

  Eigen::Matrix3d Rc1c2;
  Eigen::Vector3d tc1c2;
};

#endif  // EDGE_RELATIVE_POSE_H_
