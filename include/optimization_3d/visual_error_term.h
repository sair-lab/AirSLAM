#ifndef Visual_ERROR_TERM_H_
#define Visual_ERROR_TERM_H_

#include <Eigen/Core>
#include <ceres/autodiff_cost_function.h>
#include "optimization_3d/types.h"

#include <typeinfo>

template <typename CameraType>
class MonoPointReprojectionErrorTerm{
public:
  MonoPointReprojectionErrorTerm(
      const Eigen::Vector2d& measurement, double pixel_sigma, const CameraType* camera)
      : _measurement(measurement), _pixel_sigma_inverse(1.0/pixel_sigma), _camera_ptr(camera) {
    CHECK(camera);
  }

  template <typename T>
  bool operator()(const T* const pose_p,
                  const T* const pose_q,
                  const T* const position,
                  T* residuals_ptr) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> vertex_p(pose_p);
    Eigen::Map<const Eigen::Quaternion<T>> vertex_q(pose_q);

    Eigen::Map<const Eigen::Matrix<T, 3, 1>> point_p(position);

    Eigen::Matrix<T, 3, 3> Rwb = vertex_q.matrix();
    Eigen::Matrix<T, 3, 1> Pb = Rwb.transpose() * (point_p - vertex_p);
    Eigen::Matrix<T, 2, 1> reproject_position;
    _camera_ptr->Project(reproject_position, Pb);

    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(residuals_ptr, 2, 1);
    residuals = (reproject_position - _measurement) * static_cast<T>(_pixel_sigma_inverse);

    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector2d& measurement, double pixel_sigma, const CameraType* camera) {
    return new ceres::AutoDiffCostFunction<MonoPointReprojectionErrorTerm, 2, 3, 4, 3>(
        new MonoPointReprojectionErrorTerm(measurement, pixel_sigma, camera));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  Eigen::Vector2d _measurement;
  const double _pixel_sigma_inverse;
  const CameraType* _camera_ptr;
};


template <typename CameraType>
class StereoPointReprojectionErrorTerm{
public:
  StereoPointReprojectionErrorTerm(
      const Eigen::Vector3d& measurement, double pixel_sigma, const CameraType* camera)
      : _measurement(measurement), _pixel_sigma_inverse(1.0/pixel_sigma), _camera_ptr(camera) {
    CHECK(camera);
  }

  template <typename T>
  bool operator()(const T* const pose_p,
                  const T* const pose_q,
                  const T* const position,
                  T* residuals_ptr) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> vertex_p(pose_p);
    Eigen::Map<const Eigen::Quaternion<T>> vertex_q(pose_q);

    Eigen::Map<const Eigen::Matrix<T, 3, 1>> point_p(position);

    Eigen::Matrix<T, 3, 3> Rwb = vertex_q.matrix();
    Eigen::Matrix<T, 3, 1> Pb = Rwb.transpose() * (point_p - vertex_p);
    Eigen::Matrix<T, 3, 1> reproject_position;
    _camera_ptr->StereoProject(reproject_position, Pb);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr, 3, 1);
    residuals = (reproject_position - _measurement) * static_cast<T>(_pixel_sigma_inverse);

    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector3d& measurement, double pixel_sigma, const CameraType* camera) {
    return new ceres::AutoDiffCostFunction<StereoPointReprojectionErrorTerm, 3, 3, 4, 3>(
        new StereoPointReprojectionErrorTerm(measurement, pixel_sigma, camera));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  Eigen::Vector3d _measurement;
  const double _pixel_sigma_inverse;
  const CameraType* _camera_ptr;
};


#endif  // Visual_ERROR_TERM_H_