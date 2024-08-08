#ifndef MAP_H_
#define MAP_H_

#include <opencv2/highgui/highgui.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>

#include "read_configs.h"
#include "camera.h"
#include "mappoint.h"
#include "mapline.h"
#include "frame.h"
#include "g2o_optimization/types.h"
#include "ros_publisher.h"
#include "bow/database.h"

class MapRefiner;

class Map{
public:
  Map();
  Map(OptimizationConfig& backend_optimization_config, CameraPtr camera, RosPublisherPtr ros_publisher);
  void InsertKeyframe(FramePtr frame);
  void InsertMappoint(MappointPtr mappoint);
  void InsertMapline(MaplinePtr mapline);
  bool UppdateMapline(MaplinePtr mapline);
  void UpdateMaplineEndpoints(MaplinePtr mapline);

  void CheckAndDeleteMappoint(MappointPtr mpt);
  void CheckAndDeleteMapline(MaplinePtr mpl);
  void DeleteKeyframe(FramePtr frame);

  CameraPtr GetCameraPtr();
  FramePtr GetFramePtr(int frame_id);
  MappointPtr GetMappointPtr(int mappoint_id);
  MaplinePtr GetMaplinePtr(int mapline_id);

  bool TriangulateMappoint(MappointPtr mappoint);
  bool TriangulateMaplineByMappoints(MaplinePtr mapline);
  bool UpdateMappointDescriptor(MappointPtr mappoint);
  void LocalMapOptimization(FramePtr new_frame);
  std::pair<FramePtr, FramePtr> MakeFramePair(FramePtr frame0, FramePtr frame1);
  void RemoveOutliers(const std::vector<std::pair<FramePtr, MappointPtr>>& outliers);
  void RemoveLineOutliers(const std::vector<std::pair<FramePtr, MaplinePtr>>& line_outliers);
  int UpdateFrameTrackIds(int track_id);
  int UpdateFrameLineTrackIds(int line_track_id);

  void SearchByProjection(FramePtr frame, std::vector<MappointPtr>& mappoints, 
      int thr, std::vector<std::pair<int, MappointPtr>>& good_projections);
  void SaveKeyframeTrajectory(std::string save_root);

  bool InitializeIMU(FramePtr frame);
  void SetRwg(const Eigen::Matrix3d& Rwg);
  Eigen::Matrix3d GetRwg();
  void SetIMUInit(bool imu_init);
  bool IMUInit();

  void SaveMap(const std::string& map_root);

  void SetRosPublisher(RosPublisherPtr ros_publisher);
  void Publish(double time, bool clear_old_message = false);


  // for offline optimization
  std::map<int, MappointPtr>& GetAllMappoints();
  std::map<int, MaplinePtr>& GetAllMaplines();
  std::map<int, FramePtr>& GetAllKeyframes();
  int RemoveInValidMappoints();
  int RemoveInValidMaplines();

  void UpdateCovisibilityGraph();
  void UpdateFrameCovisibility(FramePtr frame);

  void GetConnectedFrames(FramePtr frame, std::map<FramePtr, int>& covi_frames);

  // visualization
  double MapScale();

  // debug
  void CheckMap();

public:
  // tmp parameters
  std::vector<std::pair<FramePtr, int>> to_update_track_id;
  std::vector<std::pair<FramePtr, int>> to_update_line_track_id;

  // for imu
  FramePtr last_keyframe;

  double imu_init_time;
  FramePtr imu_init_frame;
  int imu_init_stage;  


private:
  friend class MapRefiner;
  friend class MapUser;
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version){
    ar & _camera;
    ar & _mappoints;
    ar & _maplines;
    ar & _keyframes;
    // ar & _keyframe_ids;
    ar & _imu_init;
    ar & boost::serialization::make_array(_Rwg.data(), _Rwg.size());

    ar & _covisibile_frames;
    ar & _database;
    ar & _junction_database;
    ar & _junction_voc;
  }

private:
  OptimizationConfig _backend_optimization_config;
  CameraPtr _camera;
  std::map<int, MappointPtr> _mappoints;
  std::map<int, MaplinePtr> _maplines;
  std::map<int, FramePtr> _keyframes;
  std::vector<int> _keyframe_ids;
  RosPublisherPtr _ros_publisher;

  // for imu
  bool _imu_init;
  Eigen::Matrix3d _Rwg;

  // for loop detection adn relocalization
  std::map<FramePtr, std::map<FramePtr, int>> _covisibile_frames;
  DatabasePtr _database;

  DatabasePtr _junction_database;
  SuperpointVocabularyPtr _junction_voc;
};

typedef std::shared_ptr<Map> MapPtr;

#endif // MAP_H_