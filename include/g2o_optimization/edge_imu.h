#ifndef EDGE_IMU_H_
#define EDGE_IMU_H_

#include <Eigen/Core> 
#include <Eigen/Geometry> 
#include <opencv2/core/eigen.hpp>
#include <g2o/core/base_binary_edge.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/sba/g2o_types_sba_api.h>
#include <g2o/types/sba/vertex_se3_expmap.h>
#include <g2o/types/slam3d_addons/line3d.h>
#include <g2o/core/base_multi_edge.h>

#include "utils.h"
#include "imu.h"
#include "g2o_optimization/vertex_imu.h"
#include "g2o_optimization/vertex_vi_pose.h"

class EdgeIMU : public g2o::BaseMultiEdge<9, Vector9d> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeIMU(PreinterationPtr preinteration_); 

  virtual bool read(std::istream& is){return false;}
  virtual bool write(std::ostream& os) const{return false;}
  void computeError();
  void PrintError();

  Eigen::Vector3d g;
  PreinterationPtr preinteration;
};

class EdgeGyr : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, VertexGyrBias, VertexGyrBias>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeGyr(){}

  virtual bool read(std::istream& is){return false;}
  virtual bool write(std::ostream& os) const{return false;}

  void computeError();
};

class EdgeAcc : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, VertexAccBias, VertexAccBias>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeAcc(){}

  virtual bool read(std::istream& is){return false;}
  virtual bool write(std::ostream& os) const{return false;}

  void computeError();
};

#endif  // EDGE_IMU_H_
