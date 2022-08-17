#include <yaml-cpp/yaml.h>

#include "camera.h"
#include "utils.h"

Camera::Camera(){
}

Camera::Camera(const std::string& camera_file){
  cv::FileStorage camera_configs(camera_file, cv::FileStorage::READ);
  if(!camera_configs.isOpened()){
      std::cerr << "ERROR: Wrong path to settings" << std::endl;
      exit(-1);
  }

  _image_height = camera_configs["image_height"];
  _image_width = camera_configs["image_width"];
  _bf = camera_configs["bf"];
  _depth_lower_thr = camera_configs["depth_lower_thr"];
  _depth_upper_thr = camera_configs["depth_upper_thr"];
  _max_x_diff = _bf / _depth_lower_thr;
  _min_x_diff = _bf / _depth_upper_thr;

  cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
  camera_configs["LEFT.K"] >> K_l;
  camera_configs["RIGHT.K"] >> K_r;

  camera_configs["LEFT.P"] >> P_l;
  camera_configs["RIGHT.P"] >> P_r;

  camera_configs["LEFT.R"] >> R_l;
  camera_configs["RIGHT.R"] >> R_r;

  camera_configs["LEFT.D"] >> D_l;
  camera_configs["RIGHT.D"] >> D_r;

  if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || 
     R_r.empty() || D_l.empty() || D_r.empty() || _image_height == 0 || _image_width == 0){
    std::cout << "ERROR: Calibration parameters to rectify stereo are missing!" << std::endl;
    exit(0);
  }

  _fx = P_l.at<double>(0, 0);
  _fy = P_l.at<double>(1, 1);
  _cx = P_l.at<double>(0, 2);
  _cy = P_l.at<double>(1, 2);
  _fx_inv = 1.0 / _fx;
  _fy_inv = 1.0 / _fy;

  cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0,3).colRange(0,3), 
      cv::Size(_image_width, _image_height), CV_32F, _mapl1, _mapl2);

  cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0,3).colRange(0,3),
      cv::Size(_image_width, _image_height), CV_32F, _mapr1, _mapr2);
}

Camera& Camera::operator=(const Camera& camera){
  _image_height = camera._image_height;
  _image_width = camera._image_width;
  _bf = camera._bf;
  _depth_lower_thr = camera._depth_lower_thr;
  _depth_upper_thr = camera._depth_upper_thr;
  _max_x_diff = _bf / _depth_lower_thr;
  _min_x_diff = _bf / _depth_upper_thr;
  _fx = camera._fx;
  _fy = camera._fy;
  _cx = camera._cx;
  _cy = camera._cy;
  _fx_inv = camera._fx_inv;
  _fy_inv = camera._fy_inv;
  _mapl1 = camera._mapl1.clone();
  _mapl2 = camera._mapl2.clone();
  _mapr1 = camera._mapr1.clone();
  _mapr2 = camera._mapr2.clone();
  return *this;
}

void Camera::UndistortImage(
    cv::Mat& image_left, cv::Mat& image_right, cv::Mat& image_left_rect, cv::Mat& image_right_rect){
  cv::remap(image_left, image_left_rect, _mapl1, _mapl2, cv::INTER_LINEAR);
  cv::remap(image_right, image_right_rect, _mapr1, _mapr2, cv::INTER_LINEAR);
}

double Camera::ImageHeight(){
  return _image_height;
}

double Camera::ImageWidth(){
  return _image_width;
}

double Camera::BF(){
  return _bf;
}

double Camera::Fx(){
  return _fx;
}

double Camera::Fy(){
  return _fy;
}

double Camera::Cx(){
  return _cx;
}

double Camera::Cy(){
  return _cy;
}

double Camera::DepthLowerThr(){
  return _depth_lower_thr;
}

double Camera::DepthUpperThr(){
  return _depth_upper_thr;
}

double Camera::MaxXDiff(){
  return _max_x_diff;
}

double Camera::MinXDiff(){
  return _min_x_diff;
}

void Camera::GetCamerMatrix(cv::Mat& camera_matrix){
  camera_matrix = (cv::Mat_<double>(3, 3) << _fx, 0.0, _cx, 0.0, _fy, _cy, 0.0, 0.0, 1.0);
}

void Camera::GetDistCoeffs(cv::Mat& dist_coeffs){
  dist_coeffs = (cv::Mat_<double>(5, 1) << 0.0, 0.0, 0.0, 0.0, 0.0);
}

bool Camera::BackProjectMono(const Eigen::Vector2d& keypoint, Eigen::Vector3d& output){
  output(0) = (keypoint(0) - _cx) * _fx_inv;
  output(1) = (keypoint(1) - _cy) * _fy_inv;
  output(2) = 1.0;
  return true;
}

bool Camera::BackProjectStereo(const Eigen::Vector3d& keypoint, Eigen::Vector3d& output){
  BackProjectMono(keypoint.head(2), output);
  double d = _bf / (keypoint(0) - keypoint(2));
  output = output * d;
  return true;
}