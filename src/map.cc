#include <cmath> 
#include <math.h>
#include <numeric>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/slam3d_addons/types_slam3d_addons.h>

#include "map.h"
#include "utils.h"
#include "line_processor.h"
#include "frame.h"
#include "g2o_optimization/g2o_optimization.h"
#include "timer.h"

Map::Map(OptimizationConfig& backend_optimization_config, CameraPtr camera, RosPublisherPtr ros_publisher):
    _backend_optimization_config(backend_optimization_config), _camera(camera), _ros_publisher(ros_publisher){
}

void Map::InsertKeyframe(FramePtr frame){
  // insert keyframe to map
  int frame_id = frame->GetFrameId();
  _keyframes[frame_id] = frame;
  _keyframe_ids.push_back(frame_id);
  if(_keyframes.size() < 2) return;

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

  // update mapline
  std::vector<MaplinePtr> new_maplines;
  const std::vector<int>& line_track_ids = frame->GetAllTrackIds();
  const std::vector<Eigen::Vector4d>& lines = frame->GatAllLines();
  const std::vector<Eigen::Vector4d>& lines_right = frame->GatAllRightLines();
  const std::vector<bool>& lines_right_valid = frame->GetAllRightLineStatus();
  std::vector<MaplinePtr>& maplines = frame->GetAllMaplines();
  for(size_t i = 0; i < frame->LineNum(); i++){
    MaplinePtr mpl = maplines[i];
    if(mpl == nullptr){
      if(line_track_ids[i] < 0) continue; // would not happen normally
      mpl = std::shared_ptr<Mapline>(new Mapline(line_track_ids[i]));
      if(lines_right_valid[i]){
        Vector6d endpoints;
        if(frame->TrianguateStereoLine(i, endpoints)){
          mpl->SetEndpoints(endpoints);
          mpl->SetObverserEndpointStatus(frame_id, 1);
        }
      }
      frame->InsertMapline(i, mpl);
      new_maplines.push_back(mpl);
    }
    mpl->AddObverser(frame_id, i);
    if(mpl->GetObverserEndpointStatus(frame_id) < 0){
      mpl->SetObverserEndpointStatus(frame_id, 0);
    }
    if(mpl->GetType() == Mapline::Type::UnTriangulated && mpl->ObverserNum() >= 2){
      TriangulateMaplineByMappoints(mpl);
    }

  }

  // add new maplines to map
  for(MaplinePtr mpl:new_maplines){
    InsertMapline(mpl);
  }

  // optimization
  if(_keyframes.size() >= 2){
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

bool Map::UppdateMapline(MaplinePtr mapline){
  if(!mapline || !mapline->IsValid()) return false;

  // get associated mappoints
  std::vector<Eigen::Vector3d> points;
  const std::map<int, int>& obversers = mapline->GetAllObversers();
  if(obversers.empty()) return false;
  for(auto& kv : obversers){
    int frame_id = kv.first;
    FramePtr frame = GetFramePtr(frame_id);
    if(!frame) continue;
    std::map<int, double> points_on_line = frame->GetPointsOnLine(kv.second);
    for(auto& point : points_on_line){
      MappointPtr mpt = frame->GetMappoint(point.first);
      if(mpt && mpt->IsValid()){
        points.push_back(mpt->GetPosition());
      }
    }
  }

  // find endpoints
  std::vector<double> dist;
  Vector6d line_cart = mapline->GetLine3D().toCartesian();
  EigenPointLineDistance3D(points, line_cart, dist);
  assert(dist.size() == points.size());
  Eigen::Vector3d line_point = line_cart.head(3);
  Eigen::Vector3d line_direction = line_cart.tail(3);
  Eigen::Index max_index;
  line_direction.array().abs().maxCoeff(&max_index);
  size_t md = max_index;  // main direction
  double max_point_d = DBL_MIN, min_point_d = DBL_MAX;
  bool find_max = false, find_min = false;
  for(size_t i = 0; i < points.size(); i++){
    if(dist[i] > 0.2) continue;
    double di = points[i](md);
    if(di > max_point_d){
      max_point_d = di;
      find_max = true;
    }

    if(di < min_point_d){
      min_point_d = di;
      find_min = true;
    }
  }

  if(!find_max || !find_min) return false;

  double r1 = (max_point_d - line_point(md)) / line_direction(md);
  double r2 = (min_point_d - line_point(md)) / line_direction(md);
  Vector6d endpoints;
  endpoints.head(3) = line_point + r1 * line_direction;
  endpoints.tail(3) = line_point + r2 * line_direction;
  mapline->SetEndpoints(endpoints, false);
  mapline->SetEndpointsUpdateStatus(false);
  return true;
}

void Map::UpdateMaplineEndpoints(MaplinePtr mapline){
  if(!mapline || !mapline->IsValid() || !mapline->ToUpdateEndpoints()) return;
  ConstLine3DPtr line_3d = mapline->GetLine3DPtr();
  const std::map<int, int>& obversers = mapline->GetAllObversers();
  const std::map<int, int>& included_endpoints = mapline->GetAllObverserEndpointStatus();

  std::vector<Eigen::Vector3d> point_3d_vector;
  if(mapline->EndpointsValid()){
    const Vector6d& endpoints = mapline->GetEndpoints();
    Eigen::Vector3d endpoint1 = endpoints.head(3);
    Eigen::Vector3d endpoint2 = endpoints.tail(3);
    point_3d_vector.push_back(endpoint1);
    point_3d_vector.push_back(endpoint2);
  }

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  for(auto& kv : obversers){
    int frame_id = kv.first;
    FramePtr frame = GetFramePtr(frame_id);
    if(!frame || included_endpoints.at(frame_id) < 0) continue;
    Eigen::Vector4d line_measurement;
    if(!frame->GetLine(kv.second, line_measurement)) continue;
    const Eigen::Matrix4d& frame_pose = frame->GetPose();
    Eigen::Matrix3d Rwc = frame_pose.block<3, 3>(0, 0);
    Eigen::Vector3d twc = frame_pose.block<3, 1>(0, 3);
    Eigen::Matrix3d Rcw = Rwc.transpose();
    Eigen::Vector3d tcw = - Rcw * twc;
    T.rotate(Rcw);
    T.pretranslate(tcw);
    g2o::Line3D line_3d_c = T * (*line_3d);
    line_3d_c.normalize();
    Vector6d line_cart_c = line_3d_c.toCartesian();
    Eigen::Vector3d line_direction = line_cart_c.tail(3);
    Eigen::Vector3d anchor_point = line_cart_c.head(3);

    Eigen::Vector3d init_point_3d1, init_point_3d2;
    if(std::abs(line_direction(2)) < 0.1){
      Eigen::Index max_index;
      line_direction.array().abs().maxCoeff(&max_index);
      size_t md = max_index;  // main direction
      assert(md < 2);
      double op1 = -anchor_point(md) / line_direction(md);
      init_point_3d1 = anchor_point + op1 * line_direction;
      double p1p2 = 1.0 / line_direction(md);
      init_point_3d2 = init_point_3d1 + p1p2 * line_direction;
    }else{
      double op1 = (1.0 - anchor_point(2)) / line_direction(2);
      init_point_3d1 = anchor_point + op1 * line_direction;
      double op2 = (1.1 - anchor_point(2)) / line_direction(2);
      init_point_3d2 = anchor_point + op2 * line_direction;
    }
    assert(init_point_3d1(2) > 0);
    assert(init_point_3d2(2) > 0);

    CameraPtr camera = frame->GetCamera();
    Eigen::Vector2d init_point_2d1, init_point_2d2;
    camera->Project(init_point_2d1, init_point_3d1);
    camera->Project(init_point_2d2, init_point_3d2);

    Eigen::Vector3d endpoint1, endpoint2;
    Point2DTo3D(init_point_3d1, init_point_3d2, init_point_2d1, init_point_2d2, 
        line_measurement.head(2), endpoint1);
    Point2DTo3D(init_point_3d1, init_point_3d2, init_point_2d1, init_point_2d2, 
        line_measurement.tail(2), endpoint2);

    endpoint1 = Rwc * endpoint1 + twc;
    endpoint2 = Rwc * endpoint2 + twc;
    point_3d_vector.push_back(endpoint1);
    point_3d_vector.push_back(endpoint2);
    mapline->SetObverserEndpointStatus(frame_id, 1);
  }

  Eigen::Vector3d line_d = line_3d->d();
  Eigen::Index max_index;
  line_d.array().abs().maxCoeff(&max_index);
  size_t md = max_index;  // main direction
  size_t max_idx = 0;
  size_t min_idx = 0;
  double max_value = DBL_MIN;
  double min_value = DBL_MAX;
  for(size_t i = 0; i < point_3d_vector.size(); i++){
    double value = point_3d_vector[i](md);
    if(value > max_value) max_idx = i;
    if(value < min_value) min_idx = i;
  }

  Vector6d endpoints;
  endpoints << point_3d_vector[min_idx], point_3d_vector[max_idx];
  mapline->SetEndpoints(endpoints, false);
  mapline->SetEndpointsUpdateStatus(false);
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

MaplinePtr Map::GetMaplinePtr(int mapline_id){
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

bool Map::TriangulateMaplineByMappoints(MaplinePtr mapline){
  if(mapline->IsValid()) return true;
  const std::map<int, int>& obversers = mapline->GetAllObversers();
  if(obversers.size() < 2) return false;
  std::vector<cv::Point3f> points;
  for(const auto& kv : obversers){
    FramePtr frame = GetFramePtr(kv.first);
    if(!frame) continue;
    std::map<int, double> points_on_line = frame->GetPointsOnLine(kv.second);
    for(auto& pkv : points_on_line){
      MappointPtr mpt = frame->GetMappoint(pkv.first);
      if(!mpt || !mpt->IsValid()) continue;
      Eigen::Vector3d p = mpt->GetPosition();
      points.emplace_back(p(0), p(1), p(2));
    }
  }
  if(points.size() < 2) return false;

  cv::Vec6f line;
  for(size_t i = 0; i < 4; i++){
    // fit line
    cv::fitLine(points, line, cv::DIST_L2, 0, 5e-2, 1e-2);

    // remove outlier
    std::vector<float> dist;
    CVPointLineDistance3D(points, line, dist);
    size_t inlier_num = 0;
    for(size_t j = 0; j < points.size(); j++){
      if(dist[j] < 0.2){
        points[inlier_num] = points[j];
        inlier_num++;
      }
    }
    points.resize(inlier_num);

    // check
    if(inlier_num == dist.size() || inlier_num < 3){
      break;
    }
  }
  if(points.size() < 2) return false;

  // set line
  Vector6d line_cart;
  line_cart << line[3], line[4], line[5], line[0], line[1], line[2];
  g2o::Line3D line_3d = g2o::Line3D::fromCartesian(line_cart);
  mapline->SetLine3D(line_3d);

  // set endpoints
  size_t md = 0;
  float max_v = std::abs(line[0]);
  for(size_t i = 1; i < 3; i++){
    float v = std::abs(line[i]);
    if(v > max_v){
      md = i;
      max_v = v;
    }
  }

  std::vector<size_t> order;
  order.resize(points.size());
  std::iota(order.begin(), order.end(), 0);       
  std::sort(order.begin(), order.end(), [&points, &md](size_t i1, size_t i2) { 
    if(md == 0) return points[i1].x < points[i2].x;
    if(md == 1) return points[i1].y < points[i2].y;
    if(md == 2) return points[i1].z < points[i2].z;
  });
  Vector6d endpoints;
  size_t min_idx = order[0], max_idx = order[(order.size()-1)];
  endpoints << points[min_idx].x, points[min_idx].y, points[min_idx].z, points[max_idx].x, points[max_idx].y, points[max_idx].z;
  mapline->SetEndpoints(endpoints, false);
  mapline->SetEndpointsUpdateStatus(false);
  for(const auto& kv : obversers){
    FramePtr frame = GetFramePtr(kv.first);
    if(!frame) continue;
    mapline->SetObverserEndpointStatus(kv.first, 1);
  }
  return true;
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
  double distances[num_valid_obversers][num_valid_obversers];
  for(size_t i = 0; i < num_valid_obversers; i++){
    distances[i][i]=0;
    for(size_t j = i + 1; j < num_valid_obversers; j++){
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
    if(deeper_layer.empty()) break;

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
  MapOfLine3d lines;
  std::vector<CameraPtr> camera_list;
  VectorOfMonoPointConstraints mono_point_constraints;
  VectorOfStereoPointConstraints stereo_point_constraints;
  VectorOfMonoLineConstraints mono_line_constraints;
  VectorOfStereoLineConstraints stereo_line_constraints;

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
  std::vector<MaplinePtr> maplines;
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

    const std::vector<MaplinePtr>& neighbor_maplines = neighbor_frame->GetConstAllMaplines();
    for(const MaplinePtr& mpl : neighbor_maplines){
      if(!mpl || !mpl->IsValid() || mpl->local_map_optimization_frame_id == new_frame_id) continue;
      mpl->local_map_optimization_frame_id = new_frame_id;
      maplines.push_back(mpl);
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
    }
    fixed_frame_num += to_add_fixed_num;
  }

  // add point constraint
  for(auto& mpt : mappoints){
    if(!mpt || !mpt->IsValid()) continue;

    // vertex
    int mpt_id = mpt->GetId();
    Position3d point;
    point.p = mpt->GetPosition();
    point.fixed = false;

    // constraints
    VectorOfMonoPointConstraints tmp_mono_point_constraints;
    VectorOfStereoPointConstraints tmp_stereo_point_constraints;
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
        tmp_stereo_point_constraints.push_back(stereo_constraint);
      }else{
        MonoPointConstraintPtr mono_constraint = std::shared_ptr<MonoPointConstraint>(new MonoPointConstraint()); 
        mono_constraint->id_pose = kv.first;
        mono_constraint->id_point = mpt_id;
        mono_constraint->id_camera = 0;
        mono_constraint->inlier = true;
        mono_constraint->keypoint = keypoint.head(2);
        mono_constraint->pixel_sigma = 0.8;
        tmp_mono_point_constraints.push_back(mono_constraint);
      }
    }

    // add to optimization
    if(tmp_stereo_point_constraints.size() > 0 || tmp_mono_point_constraints.size() > 1){
      points.insert(std::pair<int, Position3d>(mpt_id, point));
      mono_point_constraints.insert(mono_point_constraints.end(),
          tmp_mono_point_constraints.begin(), tmp_mono_point_constraints.end());
      stereo_point_constraints.insert(stereo_point_constraints.end(),
          tmp_stereo_point_constraints.begin(), tmp_stereo_point_constraints.end());
    }
  }

  // add line constraint
  for(auto& mpl : maplines){
    if(!mpl || !mpl->IsValid()) continue;

    // vertex
    int mpl_id = mpl->GetId();
    Line3d line_3d;
    line_3d.line_3d = mpl->GetLine3D();
    line_3d.fixed = false;

    // constraints
    VectorOfMonoLineConstraints tmp_mono_line_constraints;
    VectorOfStereoLineConstraints tmp_stereo_line_constraints;
    const std::map<int, int> obversers = mpl->GetAllObversers();
    for(auto& kv : obversers){
      FramePtr kf = GetFramePtr(kv.first);
      if(!kf || (kf->local_map_optimization_frame_id != new_frame_id && kf->local_map_optimization_fix_frame_id != new_frame_id)) continue;

      Eigen::Vector4d line_left, line_right;
      if(!kf->GetLine(kv.second, line_left)) continue;
      if(kf->GetLineRight(kv.second, line_right)){
        StereoLineConstraintPtr stereo_line_constraint = std::shared_ptr<StereoLineConstraint>(new StereoLineConstraint()); 
        stereo_line_constraint->id_pose = kv.first;
        stereo_line_constraint->id_line = mpl_id;
        stereo_line_constraint->id_camera = 0;
        stereo_line_constraint->inlier = true;
        stereo_line_constraint->line_2d << line_left, line_right;
        stereo_line_constraint->pixel_sigma = 0.8;
        tmp_stereo_line_constraints.push_back(stereo_line_constraint);
      }else{
        MonoLineConstraintPtr mono_line_constraint = std::shared_ptr<MonoLineConstraint>(new MonoLineConstraint()); 
        mono_line_constraint->id_pose = kv.first;
        mono_line_constraint->id_line = mpl_id;
        mono_line_constraint->id_camera = 0;
        mono_line_constraint->inlier = true;
        mono_line_constraint->line_2d = line_left;
        mono_line_constraint->pixel_sigma = 0.8;
        tmp_mono_line_constraints.push_back(mono_line_constraint);
      }
    }
    if(tmp_stereo_line_constraints.size() > 0 || tmp_mono_line_constraints.size() > 1){
      lines.insert(std::pair<int, Line3d>(mpl_id, line_3d));
      mono_line_constraints.insert(mono_line_constraints.end(),
          tmp_mono_line_constraints.begin(), tmp_mono_line_constraints.end());
      stereo_line_constraints.insert(stereo_line_constraints.end(),
          tmp_stereo_line_constraints.begin(), tmp_stereo_line_constraints.end());
    }
  }

  LocalmapOptimization(poses, points, lines, camera_list, mono_point_constraints, 
      stereo_point_constraints, mono_line_constraints, stereo_line_constraints, _backend_optimization_config);

  // erase point outliers
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

  // erase line outliers
  std::vector<std::pair<FramePtr, MaplinePtr>> line_outliers;
  for(auto& mono_line_constraint : mono_line_constraints){
    if(!mono_line_constraint->inlier){
      std::map<int, FramePtr>::iterator frame_it = _keyframes.find(mono_line_constraint->id_pose);
      std::map<int, MaplinePtr>::iterator mpl_it = _maplines.find(mono_line_constraint->id_line);
      if(frame_it != _keyframes.end() && mpl_it != _maplines.end() && frame_it->second && mpl_it->second){
        line_outliers.emplace_back(frame_it->second, mpl_it->second);
      }
    }
  }

  for(auto& stereo_line_constraint : stereo_line_constraints){
    if(!stereo_line_constraint->inlier){
      std::map<int, FramePtr>::iterator frame_it = _keyframes.find(stereo_line_constraint->id_pose);
      std::map<int, MaplinePtr>::iterator mpl_it = _maplines.find(stereo_line_constraint->id_line);
      if(frame_it != _keyframes.end() && mpl_it != _maplines.end() && frame_it->second && mpl_it->second){
        line_outliers.emplace_back(frame_it->second, mpl_it->second);
      }
    }
  }

  RemoveLineOutliers(line_outliers);

  UpdateFrameConnection(new_frame);
  // PrintConnection();

  // copy back to map
  KeyframeMessagePtr keyframe_message = std::shared_ptr<KeyframeMessage>(new KeyframeMessage);
  MapMessagePtr map_message = std::shared_ptr<MapMessage>(new MapMessage);
  MapLineMessagePtr mapline_message = std::shared_ptr<MapLineMessage>(new MapLineMessage);

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

  for(auto& kv : lines){
    int mpl_id = kv.first;
    Line3d line = kv.second; 
    if(_maplines.count(mpl_id) == 0) continue;
    MaplinePtr mpl = _maplines[mpl_id];
    mpl->SetLine3D(line.line_3d);
    mpl->SetEndpointsValidStatus(UppdateMapline(mpl));
    if(!mpl->EndpointsValid()) continue;
    const Vector6d& endpoints = mpl->GetEndpoints();
    mapline_message->ids.push_back(mpl_id);
    mapline_message->lines.push_back(endpoints); 
  }

  _ros_publisher->PublisheKeyframe(keyframe_message);
  _ros_publisher->PublishMap(map_message);
  _ros_publisher->PublishMapLine(mapline_message);  
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
      // delete mappoint if it has only a mono obversor
      bool delete_mappoint = true;
      if(mpt->ObverserNum() > 0){
        std::map<int, FramePtr>::iterator obverser_it = _keyframes.find(obversers.begin()->first);
        if(obverser_it != _keyframes.end()){
          int ktp_idx = mpt->GetKeypointIdx(obverser_it->first);
          if(obverser_it->second->GetRightPosition(ktp_idx) < 0){
            obverser_it->second->RemoveMappoint(obversers.begin()->second);
          }else{
            delete_mappoint = false;
          }
        }
      }
      if(delete_mappoint) mpt->SetBad();
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

void Map::RemoveLineOutliers(const std::vector<std::pair<FramePtr, MaplinePtr>>& line_outliers){
  for(auto& kv : line_outliers){
    FramePtr frame = kv.first;
    MaplinePtr mpl = kv.second;
    if(!frame || !mpl || mpl->IsBad()) continue;

    // remove connection in mappoint
    mpl->RemoveObverser(frame->GetFrameId());
    std::map<int, int> obversers = mpl->GetAllObversers();
    if(mpl->ObverserNum() < 2 && !mpl->IsBad()){
      // delete mapline if it has only a mono obversor
      bool delete_mapline = true;
      if(mpl->ObverserNum() > 0){
        std::map<int, FramePtr>::iterator obverser_it = _keyframes.find(obversers.begin()->first);
        if(obverser_it != _keyframes.end()){
          int line_idx = mpl->GetLineIdx(obverser_it->first);
          if(!obverser_it->second->GetRightLineStatus(line_idx)){
            obverser_it->second->RemoveMapline(obversers.begin()->second);
          }else{
            delete_mapline = false;
          }
        }
      }
      if(delete_mapline){
        mpl->SetBad();
      } 
    }

    frame->RemoveMapline(mpl);
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

  for(auto& mpt : mappoints){
    // check whether mappoint is valid
    if(!mpt || !mpt->IsValid()) continue;

    // check whether mappoint is in the front of camera
    const Eigen::Vector3d& pw = mpt->GetPosition();
    Eigen::Vector3d pc = Rwc.transpose() * (pw - twc);
    if(pc(2) <= 0) continue;

    // check whether mappoint can project on the image
    Eigen::Vector3d p2D;
    camera->StereoProject(p2D, pc);
    if(p2D(0) <= 0 || p2D(0) >= image_width || p2D(1) <=0 || p2D(1) >= image_height) continue;

    // find neighbor features 
    std::vector<int> candidate_ids;
    frame->FindNeighborKeypoints(p2D, candidate_ids, r, true);
    if(candidate_ids.empty()) continue;

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
    }
  }
}

void Map::SaveKeyframeTrajectory(std::string file_path){
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