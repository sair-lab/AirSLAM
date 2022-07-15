#ifndef MAP_H_
#define MAP_H_

// #include <ros/ros.h>
// #include <ros/package.h>
// #include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

#include "camera.h"
#include "mappoint.h"
#include "frame.h"
#include "ros_publisher.h"

class Map{
public:
  Map(CameraPtr camera, RosPublisherPtr ros_publisher);
  void InsertKeyframe(FramePtr frame);
  void InsertMappoint(MappointPtr mappoint);

  FramePtr GetFramePtr(int frame_id);
  MappointPtr GetMappointPtr(int mappoint_id);

  bool TriangulateMappoint(MappointPtr mappoint);
  bool UpdateMappointDescriptor(MappointPtr mappoint);
  void SlidingWindowOptimization();
  void SearchNeighborFrames(FramePtr frame, std::vector<FramePtr>& neighbor_frames);
  void AddFrameVertex(FramePtr frame, MapOfPoses& poses, bool fix_this_frame);
  void LocalMapOptimization(FramePtr frame);
  void SaveMap(const std::string& map_root);
  std::pair<FramePtr, FramePtr> MakeFramePair(FramePtr frame0, FramePtr frame1);
  void RemoveOutliers(const std::vector<std::pair<FramePtr, MappointPtr>>& outliers);
  void UpdateFrameConnection(FramePtr frame);
  void PrintConnection();
  void SearchByProjection(FramePtr frame, std::vector<MappointPtr>& mappoints, 
      int thr, std::vector<std::pair<int, MappointPtr>>& good_projections);
  void SaveKeyframeTrajectory(std::string save_root);

private:
  CameraPtr _camera;
  std::map<int, MappointPtr> _mappoints;
  std::map<int, FramePtr> _keyframes;
  std::vector<int> _keyframe_ids;
  RosPublisherPtr _ros_publisher;
};

typedef std::shared_ptr<Map> MapPtr;

#endif // MAP_H_