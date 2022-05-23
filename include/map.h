#ifndef MAP_H_
#define MAP_H_

// #include <ros/ros.h>
// #include <ros/package.h>
// #include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

#include "camera.h"
#include "mappoint.h"
#include "frame.h"
#include "optimization_3d/optimization_3d.h"

class Map{
public:
  Map(CameraPtr camera);
  void InsertKeyframe(FramePtr frame);
  void InsertMappoint(MappointPtr mappoint);

  FramePtr GetFramePtr(int frame_id);
  MappointPtr GetMappointPtr(int mappoint_id);

  bool TriangulateMappoint(MappointPtr mappoint);
  void SlidingWindowOptimization();
  void GlobalBundleAdjust();
  void SaveMap(const std::string& map_root);

private:
  CameraPtr _camera;
  std::map<int, MappointPtr> _mappoints;
  std::map<int, FramePtr> _keyframes;
  std::vector<int> _keyframe_ids;
};

typedef std::shared_ptr<Map> MapPtr;

#endif // MAP_H_