#ifndef MAP_USER_H_
#define MAP_USER_H_

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "feature_detector.h"
#include "super_glue.h"
#include "read_configs.h"
#include "imu.h"
#include "dataset.h"
#include "camera.h"
#include "frame.h"
#include "point_matcher.h"
#include "line_processor.h"
#include "map.h"
#include "ros_publisher.h"
#include "g2o_optimization/types.h"
#include "bow/database.h"

struct RelocalizationGroupCandidate{
  RelocalizationGroupCandidate(): group_score(0) {}

  std::set<FramePtr> group_frames;
  double group_score;
};

class MapUser{
public:
  MapUser();
  MapUser(RelocalizationConfigs& configs, ros::NodeHandle nh);

  void LoadMap(const std::string& map_root);
  void LoadVocabulary(const std::string voc_path);
  bool Relocalization(cv::Mat& image, Eigen::Matrix4d& pose);

  Eigen::Matrix4d GetBaseFramePose();
  double GetBaseFrameTimestamp();

  // Visualization
  void PubMap();
  void StopVisualization();

public:
  int new_frame_id;

private:
  // class
  RelocalizationConfigs _configs;
  FeatureDetectorPtr _feature_detector;
  PointMatcherPtr _point_matcher;
  MapPtr _map;
  CameraPtr _camera;

  // for relocalization
  DatabasePtr _database;
  DatabasePtr _junction_database;

  // for visualization
  RosPublisherPtr _ros_publisher;
  bool _stop;
  std::thread _visualization_thread;

  RelocMessagePtr _reloc_message;
};

#endif  // MAP_USER_H_