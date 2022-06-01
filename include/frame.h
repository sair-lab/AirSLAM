#ifndef FRAME_H_
#define FRAME_H_

#include <string>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "mappoint.h"
#include "camera.h"

class Frame{
public:
  Frame();
  Frame(int frame_id, bool pose_fixed, CameraPtr camera);
  Frame& operator=(const Frame& other);

  void SetFrameId(int frame_id);
  int GetFrameId();
  void SetPoseFixed(bool pose_fixed);
  bool PoseFixed();
  void SetPose(Eigen::Matrix4d& pose);
  Eigen::Matrix4d& GetPose();

  void AddFeatures(Eigen::Matrix<double, 259, Eigen::Dynamic>& features_left, 
      Eigen::Matrix<double, 259, Eigen::Dynamic>& features_right, std::vector<cv::DMatch>& stereo_matches);
  Eigen::Matrix<double, 259, Eigen::Dynamic>& GetAllFeatures();

  size_t FeatureNum();

  bool GetKeypointPosition(size_t keypoint_id, Eigen::Vector3d& keypoint_pos);
  std::vector<cv::KeyPoint>& GetAllKeypoints();
  cv::KeyPoint& GetKeypoint(size_t idx);

  double GetRightPosition(size_t idx);
  std::vector<double>& GetAllRightPosition(); 

  double GetDepth(size_t idx);
  std::vector<double>& GetAllDepth();
  void SetDepth(size_t idx, double depth);

  void SetTrackIds(std::vector<int>& track_ids);
  std::vector<int>& GetAllTrackIds();
  void SetTrackId(size_t idx, int track_id);
  int GetTrackId(size_t idx);

  MappointPtr GetMappoint(size_t idx);
  std::vector<MappointPtr>& GetAllMappoints();
  void InsertMappoint(size_t idx, MappointPtr mappoint);

  bool BackProjectPoint(size_t idx, Eigen::Vector3d& p3D);

private:
  int _frame_id;
  bool _pose_fixed;
  Eigen::Matrix4d _pose;

  Eigen::Matrix<double, 259, Eigen::Dynamic> _features;
  std::vector<cv::KeyPoint> _keypoints;
  std::vector<double> _u_right;
  std::vector<double> _depth;
  std::vector<int> _track_ids;
  std::vector<MappointPtr> _mappoints;
  CameraPtr _camera;
};

typedef std::shared_ptr<Frame> FramePtr;

#endif  // FRAME_H_