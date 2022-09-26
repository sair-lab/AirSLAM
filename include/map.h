#ifndef MAP_H_
#define MAP_H_

#include <opencv2/highgui/highgui.hpp>

#include "read_configs.h"
#include "camera.h"
#include "mappoint.h"
#include "mapline.h"
#include "frame.h"
#include "g2o_optimization/types.h"
#include "ros_publisher.h"

class Map{
public:
  Map(OptimizationConfig& backend_optimization_config, CameraPtr camera, RosPublisherPtr ros_publisher);
  void InsertKeyframe(FramePtr frame);
  void InsertMappoint(MappointPtr mappoint);
  void InsertMapline(MaplinePtr mapline);
  bool UppdateMapline(MaplinePtr mapline);
  void UpdateMaplineEndpoints(MaplinePtr mapline);

  FramePtr GetFramePtr(int frame_id);
  MappointPtr GetMappointPtr(int mappoint_id);
  MaplinePtr GetMaplinePtr(int mapline_id);

  bool TriangulateMappoint(MappointPtr mappoint);
  bool TriangulateMaplineByMappoints(MaplinePtr mapline);
  bool UpdateMappointDescriptor(MappointPtr mappoint);
  void SearchNeighborFrames(FramePtr frame, std::vector<FramePtr>& neighbor_frames);
  void AddFrameVertex(FramePtr frame, MapOfPoses& poses, bool fix_this_frame);
  void LocalMapOptimization(FramePtr new_frame);
  void SaveMap(const std::string& map_root);
  std::pair<FramePtr, FramePtr> MakeFramePair(FramePtr frame0, FramePtr frame1);
  void RemoveOutliers(const std::vector<std::pair<FramePtr, MappointPtr>>& outliers);
  void RemoveLineOutliers(const std::vector<std::pair<FramePtr, MaplinePtr>>& line_outliers);
  void UpdateFrameConnection(FramePtr frame);
  void PrintConnection();
  void SearchByProjection(FramePtr frame, std::vector<MappointPtr>& mappoints, 
      int thr, std::vector<std::pair<int, MappointPtr>>& good_projections);
  void SaveKeyframeTrajectory(std::string save_root);

private:
  OptimizationConfig _backend_optimization_config;
  CameraPtr _camera;
  std::map<int, MappointPtr> _mappoints;
  std::map<int, MaplinePtr> _maplines;
  std::map<int, FramePtr> _keyframes;
  std::vector<int> _keyframe_ids;
  RosPublisherPtr _ros_publisher;
};

typedef std::shared_ptr<Map> MapPtr;

#endif // MAP_H_