#include <cmath> 
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "map.h"
#include "utils.h"
#include "line_processor.h"
#include "frame.h"
#include "g2o_optimization/g2o_optimization.h"
#include "timer.h"

// INITIALIZE_TIMER;

Map::Map(CameraPtr camera, RosPublisherPtr ros_publisher): _camera(camera), _ros_publisher(ros_publisher){
}

void Map::InsertKeyframe(FramePtr frame){
  // insert keyframe to map
  int frame_id = frame->GetFrameId();
  _keyframes[frame_id] = frame;
  _keyframe_ids.push_back(frame_id);
  if(_keyframes.size() < 2) return;

  // START_TIMER;
  // update mappoints
  std::vector<MappointPtr> new_mappoints;
  std::vector<int>& track_ids = frame->GetAllTrackIds();
  std::vector<cv::KeyPoint>& keypoints = frame->GetAllKeypoints();
  std::vector<double>& depth = frame->GetAllDepth();
  std::vector<MappointPtr>& mappoints = frame->GetAllMappoints();
  Eigen::Matrix4d& Twf = frame->GetPose();
  Eigen::Matrix3d Rwf = Twf.block<3, 3>(0, 0);
  Eigen::Vector3d twf = Twf.block<3, 1>(0, 3);
  for(size_t i = 0; i < frame->FeatureNum(); i++){
    MappointPtr mpt = mappoints[i];
    if(mpt == nullptr){
      if(track_ids[i] < 0) continue;  // would not happen normally
      mpt = std::shared_ptr<Mappoint>(new Mappoint(track_ids[i]));
      Eigen::Matrix<double, 256, 1> descriptor;
      if(!frame->GetDescriptor(i, descriptor)) continue;
      mpt->SetDescriptor(descriptor);
      Eigen::Vector3d pf;
      if(frame->BackProjectPoint(i, pf)){
        Eigen::Vector3d pw = Rwf * pf + twf;
        mpt->SetPosition(pw);
      }
      frame->InsertMappoint(i, mpt);
      new_mappoints.push_back(mpt);
    }
    mpt->AddObverser(frame_id, i);
    if(mpt->GetType() == Mappoint::Type::UnTriangulated && mpt->ObverserNum() > 2){
      TriangulateMappoint(mpt);
    }
  }

  // add new mappoints to map
  for(MappointPtr mpt:new_mappoints){
    InsertMappoint(mpt);
  }
  // STOP_TIMER("Insert to map Time");

  // update mapline
  std::vector<MaplinePtr> new_maplines;
  const std::vector<int>& line_track_ids = frame->GetAllTrackIds();
  const std::vector<Eigen::Vector4d>& lines = frame->GatAllLines();
  const std::vector<Eigen::Vector4d>& lines_right = frame->GatAllRightLines();
  const std::vector<bool>& lines_right_valid = frame->GetAllRightLineStatus()
  std::vector<MaplinePtr>& maplines = frame->GetAllMappoints();
  for(size_t i = 0; i < frame->LineNum(); i++){
    MaplinePtr mpl = maplines[i];
    if(mpl == nullptr){
      if(line_track_ids[i] < 0) continue; // would not happen normally
      mpl = std::shared_ptr<Mapline>(new Mapline(line_track_ids[i]));
      if(lines_right_valid[i]){
        Vector6d endpoints;
        if(frame->TriangleStereoLine(i, endpoints)){
          mpl->SetEndpoints(endpoints);
        }
        frame->InsertMapline(i, mpl);
        new_maplines.push_back(mpl);
      }
    }
    mpl->AddObverser(frame_id, i);
    if(mpt->GetType() == Mappoint::Type::UnTriangulated && mpt->ObverserNum() > 2){
      const std::map<int, int>& mpl_obversers = mpl->GetAllObversers();
      int obverser_frame_id = mpl_obversers.begin->first;
      int obverser_line_idx = mpl_obversers.begin->second;

      FramePtr obverser_frame = GetFramePtr(obverser_frame_id);
      if(!obverser_frame) continue;
      Eigen::Vector4d obverser_line;
      if(!GetLine(i, obverser_line)) continue;
      Line3DPtr line_3d = std::shared_ptr<g2o::Line3D>(new g2o::Line3D());
      Eigen::Matrix4d obverser_pose = obverser_frame->GetPose();
      if(TriangleByTwoFrames(lines[i], Twf, obverser_line, obverser_pose, line_3d)) continue;
      mpl->SetLine3DPtr(line_3d);
    }
  }

  // add new maplines to map
  for(MaplinePtr mpl:new_maplines){
    InsertMapline(mpl);
  }

  // optimization
  if(_keyframes.size() >= 2){
    // SlidingWindowOptimization(frame);
    LocalMapOptimization(frame);
  }
}

void Map::InsertMappoint(MappointPtr mappoint){
  int mappoint_id = mappoint->GetId();
  _mappoints[mappoint_id] = mappoint;
}

void Map::InsertMapline(MaplinePtr mapline){
  int mapline_id = mapline->GetId();
  _maplines[mapline_id] = mapline;
}

void Map::UpdateMaplineEndpoints(MaplinePtr mapline){
  if(!mappline->ToUpdateEndpoints()) return;
  ConstLine3DPtr line_3d = mapline->GetLine3DPtr();
  Vector6 line_cart = line_3d->toCartesian();
  Eigen::Vector3d lo = line_cart.head(3);
  size_t md = 
}

FramePtr Map::GetFramePtr(int frame_id){
  if(_keyframes.count(frame_id) == 0){
    return nullptr;
  }
  return _keyframes[frame_id];
}

MappointPtr Map::GetMappointPtr(int mappoint_id){
  if(_mappoints.count(mappoint_id) == 0){
    return nullptr;
  }
  return _mappoints[mappoint_id];
}

MaplinePtr GetMaplinePtr(int mapline_id){
  if(_maplines.count(mapline_id) == 0){
    return nullptr;
  }
  return _maplines[mapline_id];
}

bool Map::TriangulateMappoint(MappointPtr mappoint){
  const std::map<int, int> obversers = mappoint->GetAllObversers();
  Eigen::Matrix3Xd G_bearing_vectors;
  Eigen::Matrix3Xd p_G_C_vector;
  G_bearing_vectors.resize(Eigen::NoChange, obversers.size());
  p_G_C_vector.resize(Eigen::NoChange, obversers.size());
  int num_valid_obversers = 0;
  for(const auto kv : obversers){
    int frame_id = kv.first;
    int keypoint_id = kv.second;
    if(_keyframes.count(frame_id) == 0) continue;
    if(keypoint_id < 0) continue;
    // if(!_keyframes[frame_id]->IsValid()) continue;
    Eigen::Vector3d keypoint_pos;
    if(!_keyframes[frame_id]->GetKeypointPosition(keypoint_id, keypoint_pos)) continue;

    Eigen::Vector3d backprojected_pos;
    _camera->BackProjectMono(keypoint_pos.head(2), backprojected_pos);
    Eigen::Matrix4d frame_pose = _keyframes[frame_id]->GetPose();
    Eigen::Matrix3d frame_R = frame_pose.block<3, 3>(0, 0);
    Eigen::Vector3d frame_p = frame_pose.block<3, 1>(0, 3);

    p_G_C_vector.col(num_valid_obversers) = frame_p;
    G_bearing_vectors.col(num_valid_obversers) = frame_R * backprojected_pos;
    num_valid_obversers++;
  }
  // std::cout << "num_valid_obversers = " << num_valid_obversers << std::endl;
  if(num_valid_obversers < 2) return false;

  Eigen::Matrix3Xd t_G_bv = G_bearing_vectors.leftCols(num_valid_obversers);
  Eigen::Matrix3Xd p_G_C = p_G_C_vector.leftCols(num_valid_obversers);

  const Eigen::MatrixXd BiD = t_G_bv *
      t_G_bv.colwise().squaredNorm().asDiagonal().inverse();
  const Eigen::Matrix3d AxtAx = num_valid_obversers * Eigen::Matrix3d::Identity() -
      BiD * t_G_bv.transpose();
  const Eigen::Vector3d Axtbx = p_G_C.rowwise().sum() - BiD *
      t_G_bv.cwiseProduct(p_G_C).colwise().sum().transpose();

  Eigen::ColPivHouseholderQR<Eigen::Matrix3d> qr = AxtAx.colPivHouseholderQr();
  static constexpr double kRankLossTolerance = 1e-5;
  qr.setThreshold(kRankLossTolerance);
  const size_t rank = qr.rank();
  if(rank < 3) return false;
  
  Eigen::Vector3d p_G_P = qr.solve(Axtbx);
  mappoint->SetPosition(p_G_P);
  return true;
}

bool TriangulateMapline(MaplinePtr mapline){

  
}

bool Map::UpdateMappointDescriptor(MappointPtr mappoint){
  const std::map<int, int> obversers = mappoint->GetAllObversers();
  typedef Eigen::Matrix<double, 256, 1> Descriptor;
  std::vector<Descriptor, Eigen::aligned_allocator<Descriptor> > descriptor_array;
  descriptor_array.resize(obversers.size());
  int num_valid_obversers = 0;
  for(const auto kv : obversers){
    int frame_id = kv.first;
    int keypoint_id = kv.second;
    if(_keyframes.count(frame_id) == 0 || keypoint_id < 0) continue;
    if(_keyframes[frame_id]->GetDescriptor(keypoint_id, descriptor_array[num_valid_obversers])){
      num_valid_obversers++;
    }
  }
  
  if(num_valid_obversers == 0){
    return false;
  }else if(num_valid_obversers <=2){
    mappoint->SetDescriptor(descriptor_array[0]);
    return true;
  }

  descriptor_array.resize(num_valid_obversers);
  // Eigen::Matrix<double, 256, Dynamic> descriptors = 
  //     Eigen::Map<Eigen::Matrix<double, 256, Dynamic>>(descriptor_array[0].data(), 3, descriptor_array.size());

  double distances[num_valid_obversers][num_valid_obversers];
  for(size_t i = 0; i < num_valid_obversers; i++){
    distances[i][i]=0;
    for(size_t j = i + 1; j < num_valid_obversers; j++){
      // double dij = (descriptor_array[i] - descriptor_array[j]).cwiseAbs2().sum();
      double dij = DescriptorDistance(descriptor_array[i], descriptor_array[j]);
      distances[i][j] = dij;
      distances[j][i] = dij;
    }
  }

  // Take the descriptor with least median distance to the rest
  double best_median = 4.0;
  size_t best_idx = 0;
  for(size_t i = 0; i < num_valid_obversers; i++){
    std::vector<double> di(distances[i], distances[i]+num_valid_obversers);
    sort(di.begin(), di.end());
    int median = di[(int)(0.5*(num_valid_obversers-1))];
    if(median < best_median){
      best_median = median;
      best_idx = i;
    }
  }

  mappoint->SetDescriptor(descriptor_array[best_idx]);
  return true;
}

void Map::SlidingWindowOptimization(FramePtr new_frame){
  const size_t WindowSize = 10;
  int new_frame_id = new_frame->GetFrameId();
  // START_TIMER;
  MapOfPoses poses;
  MapOfPoints3d points;
  std::vector<CameraPtr> camera_list;
  VectorOfMonoPointConstraints mono_point_constraints;
  VectorOfStereoPointConstraints stereo_point_constraints;

  // camera
  camera_list.emplace_back(_camera);

  // select frames to optimize
  std::vector<FramePtr> frames;
  size_t frame_num = std::min(WindowSize, _keyframe_ids.size());
  for(size_t i = _keyframe_ids.size() - frame_num; i < _keyframe_ids.size(); i++){
    frames.push_back(_keyframes[_keyframe_ids[i]]);
  }

  std::cout << "--------------SlidingWindowOptimization Begin------------------------" << std::endl;
  std::cout << "before optimization : " << std::endl;

  // point constrainnts
  for(size_t i = 0; i < frames.size(); i++){
    FramePtr frame = frames[i];
    bool fix_this_frame = ((i==0) || frame->PoseFixed());
    int frame_id = frame->GetFrameId();
    Eigen::Matrix4d& frame_pose = frame->GetPose();
    Pose3d pose;
    pose.q = frame_pose.block<3, 3>(0, 0);
    pose.p = frame_pose.block<3, 1>(0, 3);
    pose.fixed = fix_this_frame;
    poses.insert(std::pair<int, Pose3d>(frame_id, pose));  
    
    std::vector<MappointPtr>& mappoints = frame->GetAllMappoints();
    for(size_t j = 0; j < mappoints.size(); j++){
      // points
      MappointPtr mpt = mappoints[j];
      if(!mpt || !mpt->IsValid()) continue;
      Eigen::Vector3d keypoint; 
      if(!frame->GetKeypointPosition(j, keypoint)) continue;
      int mpt_id = mpt->GetId();

      if(mpt->local_map_optimization_frame_id != new_frame_id){
        Position3d point;
        point.p = mpt->GetPosition();
        point.fixed = false;
        points.insert(std::pair<int, Position3d>(mpt_id, point));
        mpt->local_map_optimization_frame_id = new_frame_id;
      }

      // visual constraint
      if(keypoint(2) > 0){
        StereoPointConstraintPtr stereo_constraint = std::shared_ptr<StereoPointConstraint>(new StereoPointConstraint()); 
        stereo_constraint->id_pose = frame_id;
        stereo_constraint->id_point = mpt_id;
        stereo_constraint->id_camera = 0;
        stereo_constraint->inlier = true;
        stereo_constraint->keypoint = keypoint;
        stereo_constraint->pixel_sigma = 0.8;
        stereo_point_constraints.push_back(stereo_constraint);
      }else{
        MonoPointConstraintPtr mono_constraint = std::shared_ptr<MonoPointConstraint>(new MonoPointConstraint()); 
        mono_constraint->id_pose = frame_id;
        mono_constraint->id_point = mpt_id;
        mono_constraint->id_camera = 0;
        mono_constraint->inlier = true;
        mono_constraint->keypoint = keypoint.head(2);
        mono_constraint->pixel_sigma = 0.8;
        mono_point_constraints.push_back(mono_constraint);
      }
    }
  }
  // STOP_TIMER("SlidingWindowOptimization Time1");
  // START_TIMER;
  LocalmapOptimization(poses, points, camera_list, mono_point_constraints, stereo_point_constraints);
  // STOP_TIMER("SlidingWindowOptimization Time2");
  // START_TIMER;

  // erase outliers
  std::vector<std::pair<FramePtr, MappointPtr>> outliers;
  for(auto& mono_point_constraint : mono_point_constraints){
    if(!mono_point_constraint->inlier){
      std::map<int, FramePtr>::iterator frame_it = _keyframes.find(mono_point_constraint->id_pose);
      std::map<int, MappointPtr>::iterator mpt_it = _mappoints.find(mono_point_constraint->id_point);
      if(frame_it != _keyframes.end() && mpt_it != _mappoints.end() && frame_it->second && mpt_it->second){
        outliers.emplace_back(frame_it->second, mpt_it->second);
      }
    }
  }

  for(auto& stereo_point_constraint : stereo_point_constraints){
    if(!stereo_point_constraint->inlier){
      std::map<int, FramePtr>::iterator frame_it = _keyframes.find(stereo_point_constraint->id_pose);
      std::map<int, MappointPtr>::iterator mpt_it = _mappoints.find(stereo_point_constraint->id_point);
      if(frame_it != _keyframes.end() && mpt_it != _mappoints.end() && frame_it->second && mpt_it->second){
        outliers.emplace_back(frame_it->second, mpt_it->second);
      }
    }
  }
  RemoveOutliers(outliers);
  // STOP_TIMER("RemoveOutliers Time2");
  // START_TIMER;
  UpdateFrameConnection(frames.back());
  // STOP_TIMER("UpdateFrameConnection Time2");
  // START_TIMER;
  // PrintConnection();
  // STOP_TIMER("PrintConnection Time2");

  std::cout << "after optimization : " << std::endl;

  // copy back to map
  KeyframeMessagePtr keyframe_message = std::shared_ptr<KeyframeMessage>(new KeyframeMessage);
  MapMessagePtr map_message = std::shared_ptr<MapMessage>(new MapMessage);

  for(auto& kv : poses){
    int frame_id = kv.first;
    Pose3d pose = kv.second;
    if(_keyframes.count(frame_id) == 0) continue;
    Eigen::Matrix4d pose_eigen;
    pose_eigen.block<3, 3>(0, 0) = pose.q.matrix();
    pose_eigen.block<3, 1>(0, 3) = pose.p;
    _keyframes[frame_id]->SetPose(pose_eigen);

    keyframe_message->ids.push_back(frame_id);
    keyframe_message->poses.push_back(pose_eigen);
  }

  for(auto& kv : points){
    int mpt_id = kv.first;
    Position3d position = kv.second;
    if(_mappoints.count(mpt_id) == 0) continue;
    _mappoints[mpt_id]->SetPosition(position.p);

    map_message->ids.push_back(mpt_id);
    map_message->points.push_back(position.p);
  }

  _ros_publisher->PublisheKeyframe(keyframe_message);
  _ros_publisher->PublishMap(map_message);
  std::cout << "--------------SlidingWindowOptimization Finish------------------------" << std::endl;
  // STOP_TIMER("SlidingWindowOptimization Time3");

}

void Map::SearchNeighborFrames(FramePtr frame, std::vector<FramePtr>& neighbor_frames){
  const int target_num = 9;
  int frame_id = frame->GetFrameId();
  neighbor_frames.clear();
  // 1. when keyframes are no more than target_num
  if(_keyframes.size() <= target_num){
    for(auto& kv : _keyframes){
      kv.second->local_map_optimization_frame_id = frame_id;
      neighbor_frames.push_back(kv.second);
    }
    return;
  }

  // 2. when keyframes are more than target_num
  neighbor_frames.push_back(frame);
  frame->local_map_optimization_frame_id = frame_id;
  std::vector<std::pair<int, FramePtr>> connections = frame->GetOrderedConnections(-1);
  int connection_num = connections.size();
  int added_first_layer_num = std::min(connection_num, target_num-1);
  for(int i = 0; i < added_first_layer_num; i++){
    connections[i].second->local_map_optimization_frame_id = frame_id;
    neighbor_frames.push_back(connections[i].second);
  }
  FramePtr parent = frame->GetParent();
  if(parent && parent->local_map_optimization_frame_id != frame_id){
    parent->local_map_optimization_frame_id = frame_id;
    neighbor_frames.push_back(parent);
  }

  // 3. if not enough, search deeper layers
  while(neighbor_frames.size() < target_num){
    std::map<FramePtr, int> deeper_layer;
    for(auto kf : neighbor_frames){
      std::vector<std::pair<int, FramePtr>> deeper_layer_connections = kf->GetOrderedConnections(-1);
      for(auto& kv : deeper_layer_connections){
        if(kv.second->local_map_optimization_frame_id != frame_id){
          deeper_layer[kv.second] += kv.first;
        }
      }
    }

    // std::map<int, FramePtr> ordered_deeper_layer;
    std::set<std::pair<int, FramePtr>> ordered_deeper_layer;
    for(auto& kv : deeper_layer){
      ordered_deeper_layer.insert(std::pair<int, FramePtr>(kv.second, kv.first));
    }

    int added_num = std::min(target_num - neighbor_frames.size(), ordered_deeper_layer.size());
    for(std::set<std::pair<int, FramePtr>>::reverse_iterator rit = ordered_deeper_layer.rbegin(); added_num > 0; added_num--, rit++){
      rit->second->local_map_optimization_frame_id = frame_id;
      neighbor_frames.push_back(rit->second);
    }
  }
}

void Map::AddFrameVertex(FramePtr frame, MapOfPoses& poses, bool fix_this_frame){
  int frame_id = frame->GetFrameId();
  Eigen::Matrix4d& frame_pose = frame->GetPose();
  Pose3d pose;
  pose.q = frame_pose.block<3, 3>(0, 0);
  pose.p = frame_pose.block<3, 1>(0, 3);
  pose.fixed = fix_this_frame;
  poses.insert(std::pair<int, Pose3d>(frame_id, pose)); 
}

void Map::LocalMapOptimization(FramePtr new_frame){
  UpdateFrameConnection(new_frame);
  int new_frame_id = new_frame->GetFrameId();  

  MapOfPoses poses;
  MapOfPoints3d points;
  std::vector<CameraPtr> camera_list;
  VectorOfMonoPointConstraints mono_point_constraints;
  VectorOfStereoPointConstraints stereo_point_constraints;

  // camera
  camera_list.emplace_back(_camera);

  // select frames
  size_t fixed_frame_num = 0;
  std::vector<FramePtr> neighbor_frames;
  SearchNeighborFrames(new_frame, neighbor_frames);

  for(auto& kf : neighbor_frames){
    bool fix_this_frame = (kf->GetFrameId() == 0);
    fixed_frame_num = fix_this_frame ? (fixed_frame_num + 1) : fixed_frame_num;
    AddFrameVertex(kf, poses, fix_this_frame);
  }

  // select fixed frames and mappoints
  std::map<FramePtr, int> fixed_frames;
  std::vector<MappointPtr> mappoints;
  for(auto neighbor_frame : neighbor_frames){
    std::vector<MappointPtr>& neighbor_mappoints = neighbor_frame->GetAllMappoints();
    for(MappointPtr mpt : neighbor_mappoints){
      if(!mpt || !mpt->IsValid() || mpt->local_map_optimization_frame_id == new_frame_id) continue;
      mpt->local_map_optimization_frame_id = new_frame_id;
      mappoints.push_back(mpt);

      const std::map<int, int> obversers = mpt->GetAllObversers();
      for(auto& kv : obversers){
        FramePtr kf = GetFramePtr(kv.first);
        if(!kf) continue;
        if(kf->local_map_optimization_frame_id != new_frame_id){
          fixed_frames[kf]++;
        }
      }
    }
  }


  const size_t max_fixed_frame_num = 1;
  if(fixed_frames.size() > 0 && max_fixed_frame_num > fixed_frame_num){
    std::set<std::pair<int, FramePtr>> ordered_fixed_frames;
    for(auto& kv : fixed_frames){
      ordered_fixed_frames.insert(std::pair<int, FramePtr>(kv.second, kv.first));
    }

    size_t to_add_fixed_num = std::min((max_fixed_frame_num-fixed_frame_num), ordered_fixed_frames.size());
    for(std::set<std::pair<int, FramePtr>>::reverse_iterator rit = ordered_fixed_frames.rbegin(); to_add_fixed_num > 0; to_add_fixed_num--, rit++){
      rit->second->local_map_optimization_fix_frame_id = new_frame_id;
      AddFrameVertex(rit->second, poses, true);
      std::cout << rit->second->GetFrameId() << " ";
    }
    fixed_frame_num += to_add_fixed_num;
  }

  std::cout << std::endl;

  // add point constraint
  for(auto& mpt : mappoints){
    if(!mpt || !mpt->IsValid()) continue;

    // add vertex
    int mpt_id = mpt->GetId();
    Position3d point;
    point.p = mpt->GetPosition();
    point.fixed = false;
    points.insert(std::pair<int, Position3d>(mpt_id, point));

    // add constraints
    const std::map<int, int> obversers = mpt->GetAllObversers();
    for(auto& kv : obversers){
      FramePtr kf = GetFramePtr(kv.first);
      if(!kf || (kf->local_map_optimization_frame_id != new_frame_id && kf->local_map_optimization_fix_frame_id != new_frame_id)) continue;

      Eigen::Vector3d keypoint; 
      if(!kf->GetKeypointPosition(kv.second, keypoint)) continue;
      // visual constraint
      if(keypoint(2) > 0){
        StereoPointConstraintPtr stereo_constraint = std::shared_ptr<StereoPointConstraint>(new StereoPointConstraint()); 
        stereo_constraint->id_pose = kv.first;
        stereo_constraint->id_point = mpt_id;
        stereo_constraint->id_camera = 0;
        stereo_constraint->inlier = true;
        stereo_constraint->keypoint = keypoint;
        stereo_constraint->pixel_sigma = 0.8;
        stereo_point_constraints.push_back(stereo_constraint);
      }else{
        MonoPointConstraintPtr mono_constraint = std::shared_ptr<MonoPointConstraint>(new MonoPointConstraint()); 
        mono_constraint->id_pose = kv.first;
        mono_constraint->id_point = mpt_id;
        mono_constraint->id_camera = 0;
        mono_constraint->inlier = true;
        mono_constraint->keypoint = keypoint.head(2);
        mono_constraint->pixel_sigma = 0.8;
        mono_point_constraints.push_back(mono_constraint);
      }
    }
  }

  // STOP_TIMER("SlidingWindowOptimization Time1");
  // START_TIMER;
  LocalmapOptimization(poses, points, camera_list, mono_point_constraints, stereo_point_constraints);
  // STOP_TIMER("SlidingWindowOptimization Time2");
  // START_TIMER;

  // erase outliers
  std::vector<std::pair<FramePtr, MappointPtr>> outliers;
  for(auto& mono_point_constraint : mono_point_constraints){
    if(!mono_point_constraint->inlier){
      std::map<int, FramePtr>::iterator frame_it = _keyframes.find(mono_point_constraint->id_pose);
      std::map<int, MappointPtr>::iterator mpt_it = _mappoints.find(mono_point_constraint->id_point);
      if(frame_it != _keyframes.end() && mpt_it != _mappoints.end() && frame_it->second && mpt_it->second){
        outliers.emplace_back(frame_it->second, mpt_it->second);
      }
    }
  }

  for(auto& stereo_point_constraint : stereo_point_constraints){
    if(!stereo_point_constraint->inlier){
      std::map<int, FramePtr>::iterator frame_it = _keyframes.find(stereo_point_constraint->id_pose);
      std::map<int, MappointPtr>::iterator mpt_it = _mappoints.find(stereo_point_constraint->id_point);
      if(frame_it != _keyframes.end() && mpt_it != _mappoints.end() && frame_it->second && mpt_it->second){
        outliers.emplace_back(frame_it->second, mpt_it->second);
      }
    }
  }
  RemoveOutliers(outliers);
  // STOP_TIMER("RemoveOutliers Time2");
  // START_TIMER;
  UpdateFrameConnection(new_frame);
  // STOP_TIMER("UpdateFrameConnection Time2");
  // START_TIMER;
  // PrintConnection();
  // STOP_TIMER("PrintConnection Time2");


  // copy back to map
  KeyframeMessagePtr keyframe_message = std::shared_ptr<KeyframeMessage>(new KeyframeMessage);
  MapMessagePtr map_message = std::shared_ptr<MapMessage>(new MapMessage);

  for(auto& kv : poses){
    int frame_id = kv.first;
    Pose3d pose = kv.second;
    if(_keyframes.count(frame_id) == 0) continue;
    Eigen::Matrix4d pose_eigen;
    pose_eigen.block<3, 3>(0, 0) = pose.q.matrix();
    pose_eigen.block<3, 1>(0, 3) = pose.p;
    _keyframes[frame_id]->SetPose(pose_eigen);

    keyframe_message->ids.push_back(frame_id);
    keyframe_message->poses.push_back(pose_eigen);
  }

  for(auto& kv : points){
    int mpt_id = kv.first;
    Position3d position = kv.second;
    if(_mappoints.count(mpt_id) == 0) continue;
    _mappoints[mpt_id]->SetPosition(position.p);

    map_message->ids.push_back(mpt_id);
    map_message->points.push_back(position.p);
  }

  _ros_publisher->PublisheKeyframe(keyframe_message);
  _ros_publisher->PublishMap(map_message);
  // STOP_TIMER("SlidingWindowOptimization Time3");
}

std::pair<FramePtr, FramePtr> Map::MakeFramePair(FramePtr frame0, FramePtr frame1){
  if(frame0->GetFrameId() > frame1->GetFrameId()){
    return std::pair<FramePtr, FramePtr>(frame0, frame1);
  }else{
    return std::pair<FramePtr, FramePtr>(frame1, frame0);
  }
}

void Map::RemoveOutliers(const std::vector<std::pair<FramePtr, MappointPtr>>& outliers){
  std::map<std::pair<FramePtr, FramePtr>, int> bad_connections; 
  for(auto& kv : outliers){
    FramePtr frame = kv.first;
    MappointPtr mpt = kv.second;
    if(!frame || !mpt || mpt->IsBad()) continue;

    // remove connection in mappoint
    mpt->RemoveObverser(frame->GetFrameId());
    std::map<int, int> obversers = mpt->GetAllObversers();
    for(auto& ob : obversers){
      std::map<int, FramePtr>::iterator obverser_it = _keyframes.find(ob.first);
      if(obverser_it != _keyframes.end()){
        bad_connections[MakeFramePair(frame, obverser_it->second)]++;
      }
    }

    if(mpt->ObverserNum() < 2 && !mpt->IsBad()){
      if(mpt->ObverserNum() > 0){
        std::map<int, FramePtr>::iterator obverser_it = _keyframes.find(obversers.begin()->first);
        if(obverser_it != _keyframes.end()){
          obverser_it->second->RemoveMappoint(obversers.begin()->second);
        }
      }
      mpt->SetBad();
    }

    // remove connection in frame
    frame->RemoveMappoint(mpt);
  }

  // update connections between frames
  for(auto& bad_connection : bad_connections){
    FramePtr frame0 = bad_connection.first.first;
    FramePtr frame1 = bad_connection.first.second;
    frame0->DecreaseWeight(frame1, bad_connection.second);
    frame1->DecreaseWeight(frame0, bad_connection.second);
  }
}

void Map::UpdateFrameConnection(FramePtr frame){
  int frame_id = frame->GetFrameId();
  std::vector<MappointPtr> mappoints = frame->GetAllMappoints();
  std::map<int, int> connections;
  for(MappointPtr mpt : mappoints){
    if(!mpt || mpt->IsBad()) continue;
    std::map<int, int> obversers = mpt->GetAllObversers();
    for(auto& kv : obversers){
      int observer_id = kv.first;
      if(observer_id == frame_id) continue;
      if(_keyframes.find(observer_id) == _keyframes.end()) continue;
      connections[observer_id]++;
    }
  }
  if(connections.empty()) return;

  std::set<std::pair<int, FramePtr>> good_connections;
  FramePtr best_connection;
  int best_weight = -1;
  const int MinWeight = 15;
  for(auto& kv : connections){
    FramePtr connected_frame = _keyframes[kv.first];
    assert(connected_frame != nullptr);
    int connected_weight = kv.second;
    if(connected_weight > best_weight){
      best_connection = connected_frame;
      best_weight = connected_weight;
    }

    if(connected_weight > MinWeight){
      good_connections.insert(std::pair<int, FramePtr>(connected_weight, connected_frame));
      connected_frame->AddConnection(frame, connected_weight);
    }
  }

  if(good_connections.empty()){
    good_connections.insert(std::pair<int, FramePtr>(best_weight, best_connection));
    best_connection->AddConnection(frame, best_weight);  
  }
 
  frame->AddConnection(good_connections);
}

void Map::PrintConnection(){
  for(auto& kv : _keyframes){
    FramePtr frame = kv.second;
    std::vector<std::pair<int, FramePtr>> connections = frame->GetOrderedConnections(-1);
    std::cout << "Connection of frame " << frame->GetFrameId() << " : ";
    for(auto& kv : connections){
      std::cout << kv.second->GetFrameId() << "--" << kv.first << ", ";
    }
    std::cout << std::endl;
  }
}

void Map::SearchByProjection(FramePtr frame, std::vector<MappointPtr>& mappoints, 
    int thr, std::vector<std::pair<int, MappointPtr>>& good_projections){
  int frame_id = frame->GetFrameId();
  Eigen::Matrix4d pose = frame->GetPose();
  Eigen::Matrix3d Rwc = pose.block<3, 3>(0, 0);
  Eigen::Vector3d twc = pose.block<3, 1>(0, 3);
  Eigen::Matrix<double, 259, Eigen::Dynamic>& features = frame->GetAllFeatures();
  CameraPtr camera = frame->GetCamera();
  double image_width = camera->ImageWidth();
  double image_height = camera->ImageHeight();
  const double r = 15.0 * thr;

  Eigen::VectorXi debug_vec = Eigen::VectorXi::Zero(6);
  for(auto& mpt : mappoints){
    // check whether mappoint is valid
    if(!mpt || !mpt->IsValid()) continue;

    // check whether mappoint is in the front of camera
    const Eigen::Vector3d& pw = mpt->GetPosition();
    Eigen::Vector3d pc = Rwc.transpose() * (pw - twc);
    if(pc(2) <= 0) continue;
    debug_vec(0) += 1;

    // check whether mappoint can project on the image
    Eigen::Vector3d p2D;
    camera->StereoProject(p2D, pc);
    if(p2D(0) <= 0 || p2D(0) >= image_width || p2D(1) <=0 || p2D(1) >= image_height) continue;
    debug_vec(1) += 1;

    // find neighbor features 
    std::vector<int> candidate_ids;
    frame->FindNeighborKeypoints(p2D, candidate_ids, r, true);
    // std::cout << "candidate_ids.size() = " << candidate_ids.size() << std::endl;
    if(candidate_ids.empty()) continue;
    debug_vec(2) += 1;

    Eigen::Matrix<double, 256, 1>& mpd_desc = mpt->GetDescriptor(); 
    double best_dist = 4.0;
    int best_idx = -1;
    double second_dist = 4.0;
    for(auto& idx : candidate_ids){
      double dist = DescriptorDistance(mpd_desc, features.block<256, 1>(3, idx));
      if(dist < best_dist){
        second_dist = best_dist;
        best_dist = dist;
        best_idx = idx;
      }else if(dist < second_dist){
        second_dist = dist;
      }
    }

    const double distance_threshold = 0.35;
    const double ratio_threshold = 0.6;
    if(best_dist < distance_threshold) debug_vec(3) += 1;
    if(best_dist < ratio_threshold * second_dist) debug_vec(4) += 1;
    if(best_dist < distance_threshold && best_dist < ratio_threshold * second_dist){
      // frame->InsertMappoint(best_idx, mpt);
      good_projections.emplace_back(best_idx, mpt);
      debug_vec(5) += 1;
    }
  }

  std::cout << "debug_vec = " << debug_vec.transpose() << std::endl;
}

void Map::SaveKeyframeTrajectory(std::string save_root){
  std::string file_path = ConcatenateFolderAndFileName(save_root, "keyframe_trajectory.txt");
  std::cout << "Save file to " << file_path << std::endl;
  std::ofstream f;
  f.open(file_path.c_str());
  f << std::fixed;
  std::cout << "_keyframe_ids.size = " << _keyframe_ids.size() << std::endl;
  for(auto& frame_id : _keyframe_ids){
    FramePtr kf = _keyframes[frame_id];
    Eigen::Matrix4d& pose = kf->GetPose();
    Eigen::Vector3d t = pose.block<3, 1>(0, 3);
    Eigen::Quaterniond q(pose.block<3, 3>(0, 0));

    f << std::setprecision(9) << kf->GetTimestamp() << " " 
      << std::setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " "
      << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
  f.close();
}

void Map::SaveMap(const std::string& map_root){
  // save keyframes
  std::string frame_root = ConcatenateFolderAndFileName(map_root, "frames");
  MakeDir(frame_root);
  for(auto& kv : _keyframes){
    FramePtr frame = kv.second;
    std::vector<std::vector<std::string> > frame_lines;
    std::vector<std::string> metadata;
    std::string frame_id = std::to_string(frame->GetFrameId());
    metadata.emplace_back(frame_id);
    Eigen::Matrix4d& pose = frame->GetPose();
    for(int i = 0; i < 3; i++){
      for(int j = 0; j < 4; j++){
        metadata.emplace_back(std::to_string(pose(i, j)));
      }
    }
    frame_lines.emplace_back(metadata);
    
    Eigen::Matrix<double, 259, Eigen::Dynamic>& features = frame->GetAllFeatures();
    std::vector<int>& track_ids = frame->GetAllTrackIds();
    assert(features.cols() == track_ids.size());
    for(size_t i = 0; i < track_ids.size(); i++){
      std::vector<std::string> feature_line;
      feature_line.emplace_back(std::to_string(track_ids[i]));
      for(size_t j = 0; j < features.rows(); j++){
        feature_line.emplace_back(std::to_string(features(j, i)));
      }
      frame_lines.emplace_back(feature_line);
    }
    
    std::string frame_file_name = frame_id + ".txt";
    std::string frame_file = ConcatenateFolderAndFileName(frame_root, frame_file_name);
    WriteTxt(frame_file, frame_lines, ",");
  }

  // save mappoints
  std::vector<std::vector<std::string> > mappoints_lines;
  for(auto& kv : _mappoints){
    MappointPtr mappoint = kv.second;
    if(!mappoint->IsValid()) continue;
    std::vector<std::string> mappoint_line;
    std::string mappoint_id = std::to_string(mappoint->GetId());
    mappoint_line.emplace_back(mappoint_id);
    Eigen::Vector3d& position = mappoint->GetPosition();
    for(size_t i = 0; i < 3; i++){
      mappoint_line.emplace_back(std::to_string(position(i)));
    }
    mappoints_lines.emplace_back(mappoint_line);
  }
  std::string mappoints_file = ConcatenateFolderAndFileName(map_root, "mappoints.txt");
  WriteTxt(mappoints_file, mappoints_lines, ",");
}




// ros::Time ConvertToRosTime(int64_t& t){
//   const uint32_t kNanosecondsToSecond = 1e9;
//   const uint64_t timestamp_u64 = static_cast<uint64_t>(t);
//   const uint32_t ros_timestamp_sec = timestamp_u64 / kNanosecondsToSecond;
//   const uint32_t ros_timestamp_nsec =
//       timestamp_u64 - (ros_timestamp_sec * kNanosecondsToSecond);
//   return ros::Time(ros_timestamp_sec, ros_timestamp_nsec);
// }

