#ifndef MAP_REFINER_H_
#define MAP_REFINER_H_

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "super_point.h"
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

struct LoopGroupCandidate{
  LoopGroupCandidate(): group_score(0) {}

  std::set<FramePtr> group_frames;
  double group_score;
};

struct LoopFramePair{
  LoopFramePair() {}

  FramePtr query_frame;
  FramePtr loop_frame;
  Eigen::Matrix3d Rlq;
  Eigen::Vector3d tlq;
};


class MapRefiner{
public:
  MapRefiner();
  MapRefiner(MapRefinementConfigs& configs, ros::NodeHandle nh);

  void LoadMap(const std::string& map_root);

  // for loop closure
  void LoadVocabulary(const std::string voc_path);

  void UpdateCovisibilityGraph();

  int LoopDetection();
  void LoopDetection(FramePtr frame, DBoW2::WordIdToFeatures& word_features, DBoW2::BowVector& bow_vector);
  void RelativatePoseEstimation(FramePtr frame, DBoW2::WordIdToFeatures& word_features, 
      FramePtr loop_frame, std::vector<cv::DMatch>& loop_matches, std::map<FramePtr, LoopGroupCandidate>& group_candidates);

  void PoseGraphRefinement();

  void MergeMap();
  void MergeMappoints();
  void MergeMappointGroup(const std::set<int>& mappoint_group);
  void MergeMaplines();
  void MergeMaplineGroup(const std::set<int>& mapline_group);

  void GlobalMapOptimization();

  // for junction database
  void BuildJunctionDatabase();

  // for saving
  void SaveTrajectory(std::string save_path);
  void SaveFinalMap(std::string map_root);

  // for visualization
  void PubMap();
  void StopVisualization();
  void Wait(int breakpoint);

private:
  // tmp
  double odometry_length;
  std::vector<LoopFramePair> loop_frame_pairs;
  std::map<MappointPtr, std::set<MappointPtr>> merged_mappoints;


private:
  // class
  MapRefinementConfigs _configs;
  PointMatcherPtr _point_matcher;
  MapPtr _map;
  CameraPtr _camera;

  // for loop closure
  DatabasePtr _database;

  // for visualization
  RosPublisherPtr _ros_publisher;
  std::mutex _map_mutex;
  bool _stop;
  bool _stopped;
  bool _map_ready;
  std::thread _visualization_thread;
};

#endif  // MAP_REFINER_H_