#ifndef ROS_PUBLISHER_H_
#define ROS_PUBLISHER_H_

#include <map>
#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud.h>
#include <visualization_msgs/Marker.h>

#include "utils.h"
#include "read_configs.h"
#include "thread_publisher.h"

struct FeatureMessgae{
  double time;
  cv::Mat image;
  std::vector<bool> inliers;
  std::vector<cv::KeyPoint> keypoints;
};
typedef std::shared_ptr<FeatureMessgae> FeatureMessgaePtr;
typedef std::shared_ptr<const FeatureMessgae> FeatureMessgaeConstPtr;

struct FramePoseMessage{
  double time;
  Eigen::Matrix4d pose;
};
typedef std::shared_ptr<FramePoseMessage> FramePoseMessagePtr;
typedef std::shared_ptr<const FramePoseMessage> FramePoseMessageConstPtr;

struct KeyframeMessage{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double time;
  std::vector<int> ids;
  std::vector<Eigen::Matrix4d> poses;
  // Eigen::Matrix4Xd poses;
};
typedef std::shared_ptr<KeyframeMessage> KeyframeMessagePtr;
typedef std::shared_ptr<const KeyframeMessage> KeyframeMessageConstPtr;

struct MapMessage{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  double time;
  bool reset;
  std::vector<int> ids;
  std::vector<Eigen::Vector3d> points;
  // Eigen::Matrix3Xd points;
};
typedef std::shared_ptr<MapMessage> MapMessagePtr;
typedef std::shared_ptr<const MapMessage> MapMessageConstPtr;

struct MapLineMessage{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  double time;
  bool reset;
  std::vector<int> ids;
  std::vector<Vector6d> lines;
};
typedef std::shared_ptr<MapLineMessage> MapLineMessagePtr;
typedef std::shared_ptr<const MapLineMessage> MapLineMessageConstPtr;


double GetCurrentTime();

class RosPublisher{
public:
  RosPublisher(const RosPublisherConfig& ros_publisher_config);

  void PublishFeature(FeatureMessgaePtr feature_message);
  void PublishFramePose(FramePoseMessagePtr frame_pose_message);
  void PublisheKeyframe(KeyframeMessagePtr keyframe_message);
  void PublishMap(MapMessagePtr map_message);
  void PublishMapLine(MapLineMessagePtr mapline_message);

  void ShutDown();

private:
  RosPublisherConfig _config;
  ros::NodeHandle nh;

  // for publishing features
  ros::Publisher _ros_feature_pub;
  ThreadPublisher<FeatureMessgae> _feature_publisher;

  // for publishing frame
  ros::Publisher _ros_frame_pose_pub;
  ThreadPublisher<FramePoseMessage> _frame_pose_publisher;

  // for publishing keyframes
  ros::Publisher _ros_keyframe_pub;
  ros::Publisher _ros_path_pub;
  std::map<int, int> _keyframe_id_to_index;
  geometry_msgs::PoseArray  _ros_keyframe_array;
  nav_msgs::Path _ros_path;
  ThreadPublisher<KeyframeMessage> _keyframe_publisher;

  // for publishing mappoints
  ros::Publisher _ros_map_pub;
  std::unordered_map<int, int> _mappoint_id_to_index;
  sensor_msgs::PointCloud _ros_mappoints;
  ThreadPublisher<MapMessage> _map_publisher;

  // for publishing maplines
  ros::Publisher _ros_mapline_pub;
  std::unordered_map<int, int> _mapline_id_to_index;
  visualization_msgs::Marker _ros_maplines;
  ThreadPublisher<MapLineMessage> _mapline_publisher;
};
typedef std::shared_ptr<RosPublisher> RosPublisherPtr;

#endif  // ROS_PUBLISHER_H_