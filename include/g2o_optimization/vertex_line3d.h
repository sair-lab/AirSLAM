#ifndef VERTEX_LINE_3D_H_
#define VERTEX_LINE_3D_H_

#include <Eigen/Core> 
#include <Eigen/Geometry> 
#include <opencv2/core/eigen.hpp>
#include <g2o/core/base_binary_edge.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/sba/g2o_types_sba_api.h>
#include <g2o/types/sba/vertex_se3_expmap.h>
#include <g2o/types/slam3d_addons/line3d.h>

class VertexLine3D : public g2o::BaseVertex<4, g2o::Line3D> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexLine3D();
  virtual bool read(std::istream& is);
  virtual bool write(std::ostream& os) const;

  virtual void setToOriginImpl() { _estimate = g2o::Line3D(); }

  virtual void oplusImpl(const double* update_) {
    Eigen::Map<const Eigen::Vector4d> update(update_);
    _estimate.oplus(update);
  }

  virtual bool setEstimateDataImpl(const double* est) {
    Eigen::Map<const Vector6d> _est(est);
    _estimate = g2o::Line3D(_est);
    return true;
  }

  virtual bool getEstimateData(double* est) const {
    Eigen::Map<Vector6d> _est(est);
    _est = _estimate;
    return true;
  }

  virtual int estimateDimension() const { return 6; }

  Eigen::Vector3d color;
};

#endif  // VERTEX_LINE_3D_H_