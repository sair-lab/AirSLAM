#ifndef CAMERA_H_
#define CAMERA_H_

#include <limits>
#include <string>
#include <memory>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <yaml-cpp/yaml.h>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>

class Camera{
public:
  Camera();
  Camera(const std::string& camera_file);
  Camera& operator=(const Camera& camera); // deep copy
  
  void ReadCameraNode(YAML::Node& cam_node, cv::Mat& K, cv::Mat& D, Eigen::Matrix4d& Tbc);
  void UndistortImage(cv::Mat& image_left, cv::Mat& image_left_rect);
  void UndistortImage(
      cv::Mat& image_left, cv::Mat& image_right, cv::Mat& image_left_rect, cv::Mat& image_right_rect);
  double ImageHeight();
  double ImageWidth();
  bool UseIMU();
  double BF();
  double Fx();
  double Fy();
  double Cx();
  double Cy();
  double DepthLowerThr();
  double DepthUpperThr();
  double MaxXDiff();
  double MinXDiff();
  double MaxYDiff();
  Eigen::Matrix4d CameraToBody();
  Eigen::Matrix4d BodyToCamera();
  double GyrNoise();
  double AccNoise();
  double GyrWalk();
  double AccWalk();
  void GetCamerMatrix(cv::Mat& camera_matrix);
  void GetDistCoeffs(cv::Mat& dist_coeffs);

  bool BackProjectMono(const Eigen::Vector2d& keypoint, Eigen::Vector3d& output);
  bool BackProjectStereo(const Eigen::Vector3d& keypoint, Eigen::Vector3d& output);

  template<typename T> bool Project(Eigen::Matrix<T, 2, 1>& point2d, Eigen::Matrix<T, 3, 1>& point3d) const{
    if(point3d(2) <= static_cast<T>(0)) return false;

    T z_inv = static_cast<T>(1.0) / point3d(2);

    T x_normal = point3d(0) * z_inv;
    T y_normal = point3d(1) * z_inv;

    point2d(0) = x_normal * static_cast<T>(_fx) + static_cast<T>(_cx);
    point2d(1) = y_normal * static_cast<T>(_fy) + static_cast<T>(_cy);

    bool x_ok = (point2d(0) >= static_cast<T>(0)) && (point2d(0) < static_cast<T>(_image_width));
    bool y_ok = (point2d(1) >= static_cast<T>(0)) && (point2d(1) < static_cast<T>(_image_height));

    return x_ok && y_ok;
  }

  template<typename T> bool StereoProject(Eigen::Matrix<T, 3, 1>& point2d, Eigen::Matrix<T, 3, 1>& point3d) const{
    if(point3d(2) <= static_cast<T>(0)) return false;

    T z_inv = static_cast<T>(1.0) / point3d(2);

    T x_normal = point3d(0) * z_inv;
    T y_normal = point3d(1) * z_inv;

    point2d(0) = x_normal * static_cast<T>(_fx) + static_cast<T>(_cx);
    point2d(1) = y_normal * static_cast<T>(_fy) + static_cast<T>(_cy);
    point2d(2) = point2d(0) - static_cast<T>(_bf) * z_inv;

    bool x_left_ok = (point2d(0) >= static_cast<T>(0)) && (point2d(0) < static_cast<T>(_image_width));
    bool y_left_ok = (point2d(1) >= static_cast<T>(0)) && (point2d(1) < static_cast<T>(_image_height));
    bool x_right_ok = (point2d(2) >= static_cast<T>(0)) && (point2d(2) < point2d(0));

    return x_left_ok && y_left_ok && x_right_ok;
  }

public:
  static double IMU_G_VALUE;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version){
    ar & _image_height;
    ar & _image_width;

    ar & _bf;
    ar & _depth_lower_thr;
    ar & _depth_upper_thr;
    ar & _min_x_diff;
    ar & _max_x_diff;
    ar & _max_y_diff;

    ar & _fx;
    ar & _fy;
    ar & _cx;
    ar & _cy;
    ar & _fx_inv;
    ar & _fy_inv;
    SerializeCVMat(ar, _mapl1, version);
    SerializeCVMat(ar, _mapl2, version);
    SerializeCVMat(ar, _mapr1, version);
    SerializeCVMat(ar, _mapr2, version);

    ar & _use_imu;
    ar & boost::serialization::make_array(_Tbc.data(), _Tbc.size());
    ar & boost::serialization::make_array(_Tcb.data(), _Tcb.size());
    ar & _gyr_noise;
    ar & _acc_noise;
    ar & _gyr_walk;
    ar & _acc_walk;
    ar & _imu_frequency;
  }

private:
  int _image_height;
  int _image_width;

  double _bf;
  double _depth_lower_thr;
  double _depth_upper_thr;
  double _min_x_diff;
  double _max_x_diff;
  double _max_y_diff;

  double _fx;
  double _fy;
  double _cx;
  double _cy;
  double _fx_inv;
  double _fy_inv;
  cv::Mat _mapl1;
  cv::Mat _mapl2;
  cv::Mat _mapr1;
  cv::Mat _mapr2;

  // IMU related configs
  int _use_imu;
  Eigen::Matrix4d _Tbc, _Tcb;
  double _gyr_noise, _acc_noise, _gyr_walk, _acc_walk;
  double _imu_frequency;
};

typedef std::shared_ptr<Camera> CameraPtr;

#endif  // CAMERA_H_