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
#include "line_processor.h"
#include "map.h"
#include "ros_publisher.h"
#include "g2o_optimization/types.h"

struct TrackingData{
  FramePtr frame;
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
  MapBuilder(Configs& configs);
  void AddInput(InputDataPtr data);
  void ExtractFeatureThread();
  void TrackingThread();
  void Process();

  void ExtractFeatrue(const cv::Mat& image, Eigen::Matrix<double, 259, Eigen::Dynamic>& points, std::vector<Eigen::Vector4d>& lines);
  void ExtractFeatureAndMatch(const cv::Mat& image, const Eigen::Matrix<double, 259, Eigen::Dynamic>& points0, 
      Eigen::Matrix<double, 259, Eigen::Dynamic>& points1, std::vector<Eigen::Vector4d>& lines, std::vector<cv::DMatch>& matches);
  bool Init(FramePtr frame, cv::Mat& image_left, cv::Mat& image_right);
  int TrackFrame(FramePtr frame0, FramePtr frame1, std::vector<cv::DMatch>& matches);

  // pose_init = 0 : opencv pnp, pose_init = 1 : last frame pose, pose_init = 2 : original pose
  int FramePoseOptimization(FramePtr frame, std::vector<MappointPtr>& mappoints, std::vector<int>& inliers, int pose_init = 0);
  bool AddKeyframe(FramePtr last_keyframe, FramePtr current_frame, int num_match);
  void InsertKeyframe(FramePtr frame, const cv::Mat& image_right);
  void InsertKeyframe(FramePtr frame);

  // for tracking local map
  void UpdateReferenceFrame(FramePtr frame);
  void UpdateLocalKeyframes(FramePtr frame);
  void UpdateLocalMappoints(FramePtr frame);
  void SearchLocalPoints(FramePtr frame, std::vector<std::pair<int, MappointPtr>>& good_projections);
  int TrackLocalMap(FramePtr frame, int num_inlier_thr);

  void PublishFrame(FramePtr frame, cv::Mat& image);

  void SaveTrajectory();
  void SaveTrajectory(std::string file_path);
  void SaveMap(const std::string& map_root);

  void ShutDown();

private:
  // left feature extraction and tracking thread
  std::mutex _buffer_mutex;
  std::queue<InputDataPtr> _data_buffer;
  std::thread _feature_thread;

  // pose estimation thread
  std::mutex _tracking_mutex;
  std::queue<TrackingDataPtr> _tracking_data_buffer;
  std::thread _tracking_thread;

  // gpu mutex
  std::mutex _gpu_mutex;

  bool _shutdown;

  // tmp 
  bool _init;
  int _track_id;
  int _line_track_id;
  FramePtr _last_frame;
  FramePtr _last_keyframe;
  int _num_since_last_keyframe;
  bool _last_frame_track_well;

  cv::Mat _last_image;
  cv::Mat _last_right_image;
  cv::Mat _last_keyimage;

  Pose3d _last_pose; 

  // for tracking local map
  bool _to_update_local_map;
  FramePtr _ref_keyframe;
  std::vector<MappointPtr> _local_mappoints;
  std::vector<FramePtr> _local_keyframes;

  // class
  Configs _configs;
  CameraPtr _camera;
  SuperPointPtr _superpoint;
  PointMatchingPtr _point_matching;
  LineDetectorPtr _line_detector;
  RosPublisherPtr _ros_publisher;
  MapPtr _map;
};

#endif  // MAP_BUILDER_H_