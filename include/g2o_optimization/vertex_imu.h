#ifndef VERTEX_IMU_H_
#define VERTEX_IMU_H_

#include <Eigen/Core> 
#include <Eigen/Geometry> 
#include <opencv2/core/eigen.hpp>
#include <g2o/core/base_binary_edge.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/sba/g2o_types_sba_api.h>
#include <g2o/types/sba/vertex_se3_expmap.h>
#include <g2o/types/slam3d_addons/line3d.h>


class VertexVelocity : public g2o::BaseVertex<3, Eigen::Vector3d> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexVelocity();
  VertexVelocity(const Eigen::Vector3d& velocity);
  virtual bool read(std::istream& is);
  virtual bool write(std::ostream& os) const;

  virtual void setToOriginImpl() { _estimate.setZero(); }
  virtual void oplusImpl(const double* update_);
  virtual int estimateDimension() const;
};


class VertexGyrBias : public g2o::BaseVertex<3, Eigen::Vector3d> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexGyrBias();
  VertexGyrBias(const Eigen::Vector3d& gyr_bias);
  virtual bool read(std::istream& is);
  virtual bool write(std::ostream& os) const;

  virtual void setToOriginImpl();
  virtual void oplusImpl(const double* update_);
  virtual int estimateDimension() const;
};


class VertexAccBias : public g2o::BaseVertex<3, Eigen::Vector3d> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexAccBias();
  VertexAccBias(const Eigen::Vector3d& acc_bias);
  virtual bool read(std::istream& is);
  virtual bool write(std::ostream& os) const;

  virtual void setToOriginImpl();
  virtual void oplusImpl(const double* update_);
  virtual int estimateDimension() const;
};

class GDirection{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  GDirection();
  GDirection(const Eigen::Matrix3d& Rwg_);

  void Update(const double *update_);

  Eigen::Matrix3d Rwg;
  int its;
};

class VertexGDirection : public g2o::BaseVertex<2, GDirection>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VertexGDirection();
  VertexGDirection(const Eigen::Matrix3d& Rwg);

  virtual bool read(std::istream &is);
  virtual bool write(std::ostream &os) const;

  virtual void setToOriginImpl();
  virtual void oplusImpl(const double *update_);
};


#endif  // VERTEX_IMU_H_