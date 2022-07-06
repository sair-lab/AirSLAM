#ifndef MAP_BUILDER_H_
#define MAP_BUILDER_H_

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "super_point.h"
#include "super_glue.h"
#include "read_configs.h"
#include "dataset.h"
#include "camera.h"
#include "frame.h"
#include "point_matching.h"
#include "map.h"
#include "ros_publisher.h"
#include "g2o_optimization/types.h"

class MapBuilder{
public:
  MapBuilder(Configs& configs);
  void AddInput(int frame_id, cv::Mat& image_left, cv::Mat& image_right, double timestamp);
  void StereoMatch(Eigen::Matrix<double, 259, Eigen::Dynamic>& features_left, 
      Eigen::Matrix<double, 259, Eigen::Dynamic>& features_right, std::vector<cv::DMatch>& matches);
  bool Init(FramePtr frame);
  int TrackFrame(FramePtr frame0, FramePtr frame1, std::vector<cv::DMatch>& matches);
  int FramePoseOptimization(FramePtr frame, std::vector<MappointPtr>& mappoints, std::vector<int>& inliers);
  void InsertKeyframe(FramePtr frame);
  void SaveTrajectory();
  void SaveMap(const std::string& map_root);

private:
  // tmp 
  bool _init;
  int _track_id;
  FramePtr _last_frame;
  FramePtr _last_keyframe;
  int _num_since_last_keyframe;

  cv::Mat _last_image;
  cv::Mat _last_keyimage;

  Pose3d _last_pose;

  // class
  Configs _configs;
  RosPublisherPtr _ros_publisher;
  CameraPtr _camera;
  SuperPointPtr _superpoint;
  PointMatchingPtr _point_matching;
  MapPtr _map;
};

#endif  // MAP_BUILDER_H_