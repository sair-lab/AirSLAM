#ifndef MAP_BUILDER_H_
#define MAP_BUILDER_H_

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "super_glue.h"
#include "read_configs.h"
#include "imu.h"
#include "dataset.h"
#include "camera.h"
#include "frame.h"
#include "point_matcher.h"
#include "line_processor.h"
#include "feature_detector.h"
#include "map.h"
#include "ros_publisher.h"
#include "g2o_optimization/types.h"

struct InputData{
  size_t index;
  double time;
  cv::Mat image_left;
  cv::Mat image_right;
  ImuDataList batch_imu_data;

  InputData() {}
  InputData& operator =(InputData& other){
		index = other.index;
		time = other.time;
		image_left = other.image_left.clone();
		image_right = other.image_right.clone();
		return *this;
	}
};
typedef std::shared_ptr<InputData> InputDataPtr;

enum FrameType {
  NormalFrame = 0,
  KeyFrame = 1,
  InitializationFrame = 2,
};

struct TrackingData{
  FramePtr frame;
  FrameType frame_type;
  FramePtr ref_keyframe;
  std::vector<cv::DMatch> matches;
  InputDataPtr input_data;

  TrackingData() {}
  TrackingData& operator =(TrackingData& other){
		frame = other.frame;
		ref_keyframe = other.ref_keyframe;
		matches = other.matches;
		input_data = other.input_data;
		return *this;
	}
};
typedef std::shared_ptr<TrackingData> TrackingDataPtr;

class MapBuilder{
public:
  MapBuilder(VisualOdometryConfigs& configs, ros::NodeHandle nh);
  bool UseIMU();
  void AddInput(InputDataPtr data);
  void ExtractFeatureThread();
  void TrackingThread();

  int TrackFrame(FramePtr ref_frame, FramePtr current_frame, std::vector<cv::DMatch>& matches, Preinteration& _preinteration);

  int FramePoseOptimization(FramePtr frame0, FramePtr frame, std::vector<MappointPtr>& mappoints, std::vector<int>& inliers, 
      Preinteration& preinteration);
  int AddKeyframeCheck(FramePtr ref_keyframe, FramePtr current_frame, const std::vector<cv::DMatch>&);
  void InsertKeyframe(FramePtr frame);

  void PublishFrame(FramePtr frame, cv::Mat& image, FrameType frame_type, std::vector<cv::DMatch>& matches);
  void SaveTrajectory();
  void SaveTrajectory(std::string file_path);
  void SaveMap(const std::string& map_root);

  void Stop();
  bool IsStopped();


private:
  // left feature extraction and tracking thread
  std::mutex _buffer_mutex;
  std::queue<InputDataPtr> _data_buffer;
  std::thread _feature_thread;

  // pose estimation thread
  std::mutex _tracking_mutex;
  std::queue<TrackingDataPtr> _tracking_data_buffer;
  std::thread _tracking_thread;

  std::mutex _stop_mutex;
  bool _shutdown;
  bool _feature_thread_stop;
  bool _tracking_trhead_stop;

  // tmp 
  bool _init;
  bool _insert_next_keyframe;
  int _track_id;
  int _line_track_id;
  FramePtr _last_keyframe_feature;
  FramePtr _last_keyframe_tracking;
  FramePtr _last_tracked_frame;
  cv::Mat _last_keyimage;

  cv::Mat key_image_pub;
  int key_image_id_pub;
  std::vector<cv::KeyPoint> keyframe_keypoints_pub;

  // for imu
  Preinteration _preinteration_keyframe;

private:
  // class
  VisualOdometryConfigs _configs;
  CameraPtr _camera;
  PointMatcherPtr _point_matcher;
  FeatureDetectorPtr _feature_detector;
  RosPublisherPtr _ros_publisher;
  MapPtr _map;
};

#endif  // MAP_BUILDER_H_