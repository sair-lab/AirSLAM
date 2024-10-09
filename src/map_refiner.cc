#include "map_builder.h"

#include <assert.h>
#include <iostream> 
#include <Eigen/Core> 
#include <Eigen/Geometry> 
#include <opencv2/core/eigen.hpp>
#include <boost/serialization/serialization.hpp>

#include "map_refiner.h"
#include "super_glue.h"
#include "read_configs.h"
#include "imu.h"
#include "dataset.h"
#include "camera.h"
#include "frame.h"
#include "point_matcher.h"
#include "map.h"
#include "g2o_optimization/g2o_optimization.h"
#include "g2o_optimization/edge_relative_pose.h"
#include "timer.h"
// #include "debug.h"

MapRefiner::MapRefiner(){
}

MapRefiner::MapRefiner(MapRefinementConfigs& configs, ros::NodeHandle nh): odometry_length(0), 
    _configs(configs), _stop(false), _stopped(false), _map_ready(false){
  _point_matcher = std::shared_ptr<PointMatcher>(new PointMatcher(configs.point_matcher_config));
  _ros_publisher = std::shared_ptr<RosPublisher>(new RosPublisher(configs.ros_publisher_config, nh));
  _visualization_thread = std::thread(boost::bind(&MapRefiner::PubMap, this));
}

void MapRefiner::LoadMap(const std::string& map_root){
  _map_mutex.lock();
  std::string map_v0_path = ConcatenateFolderAndFileName(map_root, "AirSLAM_mapv0.bin");

  std::cout << "map_v0_path = " << map_v0_path << std::endl;

  _map = std::shared_ptr<Map>(new Map());
  std::ifstream ifs(map_v0_path, std::ios::binary);
  boost::archive::binary_iarchive ia(ifs);
  ia >> _map;

  _map->SetIMUInit(false);

  _map->SetRosPublisher(_ros_publisher);
  _camera = _map->GetCameraPtr();
  _map_ready = true;

  _map->CheckMap();
  _map_mutex.unlock();
}
  
void MapRefiner::LoadVocabulary(const std::string voc_path){
  _database = std::shared_ptr<Database>(new Database(voc_path));
}

void MapRefiner::UpdateCovisibilityGraph(){
  _map_mutex.lock();
  _map->UpdateCovisibilityGraph();
  _map_mutex.unlock();
}

int MapRefiner::LoopDetection(){
  Eigen::Vector3d last_position, current_position;
  int num_frame = 0;
  _map_mutex.lock();
  for(const auto& kv : _map->_keyframes){
    FramePtr frame = kv.second;
    if(num_frame == 0){
      last_position = frame->GetPose().block<3, 1>(0, 3);
      num_frame++;
    }else{
      current_position = frame->GetPose().block<3, 1>(0, 3);
      double distance = (current_position - last_position).norm();
      odometry_length += distance;

      last_position = current_position;
      num_frame++;
    }

    DBoW2::WordIdToFeatures word_features; 
    DBoW2::BowVector bow_vector;
    std::vector<DBoW2::WordId> word_of_features;
    _database->FrameToBow(frame, word_features, bow_vector, word_of_features);
    frame->DetectSentences(word_of_features);
    LoopDetection(frame, word_features, bow_vector);
    _database->AddFrame(frame, word_features, bow_vector);
  }
  _map_mutex.unlock();
  return loop_frame_pairs.size();
}

void MapRefiner::LoopDetection(FramePtr frame, DBoW2::WordIdToFeatures& word_features, DBoW2::BowVector& bow_vector){
  int frame_id = frame->GetFrameId();
  // query
  std::map<FramePtr, int> frame_sharing_words;
  _database->Query(bow_vector, frame_sharing_words);

  if(frame_sharing_words.empty()) return;

  // filtration
  int max_sharing_words = 0;
  for(const auto& kv : frame_sharing_words){
    max_sharing_words = kv.second > max_sharing_words ? kv.second : max_sharing_words;
  }
  int sharing_wrods_num_thr = std::max(static_cast<int>(max_sharing_words * 0.5f), 8);

  std::map<FramePtr, int> covi_frames;
  _map->GetConnectedFrames(frame, covi_frames);
  std::map<FramePtr, int>::iterator fsw_it = frame_sharing_words.begin();
  for(; fsw_it != frame_sharing_words.end();){
    FramePtr fsw = fsw_it->first;
    if(fsw->GetFrameId() >= frame_id || fsw_it->second < sharing_wrods_num_thr || covi_frames.count(fsw)){
      fsw_it = frame_sharing_words.erase(fsw_it);
    }else{
      fsw_it++;
    }
  }

  if(frame_sharing_words.empty()) return;

  // scoring
  std::map<FramePtr, double> frame_scores;
  fsw_it = frame_sharing_words.begin();
  for(; fsw_it != frame_sharing_words.end(); fsw_it++){
    FramePtr fsw = fsw_it->first;
    frame_scores[fsw] = _database->Score(_database->_frame_bow_vectors[fsw], bow_vector);
  }

  // grouping
  std::map<FramePtr, LoopGroupCandidate> group_candidates;
  FramePtr best_deputy;
  double best_group_score = -1; 
  std::map<FramePtr, LoopGroupCandidate>::iterator loop_group_iter;
  std::map<FramePtr, double>::iterator fs_it = frame_scores.begin();
  for(; fs_it != frame_scores.end(); fs_it++){
    FramePtr fsw = fs_it->first;
    FramePtr deputy_of_group = fsw;
    double deputy_score = fs_it->second;

    LoopGroupCandidate group_candidate;
    group_candidate.group_frames.insert(fsw);
    group_candidate.group_score += deputy_score;

    std::map<FramePtr, int> fsw_covi_frames;
    _map->GetConnectedFrames(fsw, fsw_covi_frames);
    for(auto& kv : fsw_covi_frames){
      FramePtr fsw_covi_frame = kv.first;
      if(kv.second > 10 && frame_scores.count(fsw_covi_frame)){
        double fsw_covi_score = frame_scores[fsw_covi_frame];
        group_candidate.group_frames.insert(fsw_covi_frame);
        group_candidate.group_score += fsw_covi_score;

        if(fsw_covi_score > deputy_score){
          deputy_of_group = fsw_covi_frame;
          deputy_score = fsw_covi_score;
        }
      }
    }

    loop_group_iter = group_candidates.find(deputy_of_group);
    if(loop_group_iter == group_candidates.end() || loop_group_iter->second.group_score < group_candidate.group_score){
      group_candidates[deputy_of_group] = group_candidate;

      if(group_candidate.group_score > best_group_score){
        best_group_score = group_candidate.group_score;
        best_deputy = deputy_of_group;
      }
    }
  }

  if(best_group_score < 0) return;

  // filtration of group by geometric verification
  Eigen::Vector3d current_position = frame->GetPose().block<3, 1>(0, 3);
  Eigen::Vector3d loop_position;
  double loop_distance_thr = odometry_length * 0.03;
  loop_group_iter = group_candidates.begin();
  for(; loop_group_iter != group_candidates.end();){
    FramePtr fsw = loop_group_iter->first;
    loop_position = fsw->GetPose().block<3, 1>(0, 3);
    double distance =  (current_position - loop_position).norm();
    if(distance > loop_distance_thr){  
      loop_group_iter = group_candidates.erase(loop_group_iter);
    }else{
      loop_group_iter++;
    }

  }


  // filtration of group by group score
  if(group_candidates.size() > 3){
    double group_score_thr = best_group_score * 0.5;
    loop_group_iter = group_candidates.begin();
    for(; loop_group_iter != group_candidates.end();){
      if(loop_group_iter->second.group_score < group_score_thr){  
        loop_group_iter = group_candidates.erase(loop_group_iter);
      }else{
        loop_group_iter++;
      }
    }
  }

  // sorting of group
  std::vector<std::pair<FramePtr, LoopGroupCandidate>> group_vector(group_candidates.begin(), group_candidates.end());
  std::sort(group_vector.begin(), group_vector.end(), [](const auto &a, const auto &b) {
      return a.second.group_score > b.second.group_score;
  });

  // feature matching
  const int GoodCandidateNum = group_vector.size() <= 5 ? group_vector.size() : 5; 
  std::vector<cv::DMatch> best_matches;
  FramePtr best_candidate;
  const Eigen::Matrix<float, 259, Eigen::Dynamic>& query_features = frame->GetAllFeatures();
  for(int i = 0; i < GoodCandidateNum; i++){
    FramePtr good_candidate = group_vector[i].first;
    const Eigen::Matrix<float, 259, Eigen::Dynamic>& good_candidate_features = good_candidate->GetAllFeatures();
    std::vector<cv::DMatch> matches;
    _point_matcher->MatchingPoints(query_features, good_candidate_features, matches, true);
    // if(matches.size() > 50){
    //   RelativatePoseEstimation(frame, word_features, good_candidate, matches, group_candidates);
    // }
    if(matches.size() > best_matches.size()){
      best_matches = matches;
      best_candidate = good_candidate;
    }
  }

  if(best_matches.size() > 50){
    RelativatePoseEstimation(frame, word_features, best_candidate, best_matches, group_candidates);
  }
}

void MapRefiner::RelativatePoseEstimation(FramePtr frame, DBoW2::WordIdToFeatures& word_features, 
    FramePtr loop_frame, std::vector<cv::DMatch>& loop_matches, std::map<FramePtr, LoopGroupCandidate>& group_candidates){
  int frame_id = frame->GetFrameId();

  std::vector<MappointPtr> matched_mappoints(frame->FeatureNum(), nullptr);
  for(const auto& match : loop_matches){
    MappointPtr mpt = loop_frame->GetMappoint(match.trainIdx);
    matched_mappoints[match.queryIdx] = mpt;
  }

  // initial pose estimation
  MapOfPoses poses;
  MapOfPoints3d points;
  MapOfLine3d lines;
  MapOfVelocity velocities;
  MapOfBias biases;
  std::vector<CameraPtr> camera_list;
  VectorOfMonoPointConstraints mono_point_constraints;
  VectorOfStereoPointConstraints stereo_point_constraints;
  VectorOfMonoLineConstraints mono_line_constraints;
  VectorOfStereoLineConstraints stereo_line_constraints;
  VectorOfIMUConstraints imu_constraints;
  Eigen::Matrix3d Rwg = _map->GetRwg();

  camera_list.emplace_back(_camera);
  AddFrameVertex(frame, poses, 0, false);

  std::vector<size_t> mono_indexes;
  std::vector<size_t> stereo_indexes;
  for(int i = 0; i < matched_mappoints.size(); i++){
    int keypoint_idx = i;
    MappointPtr mpt = matched_mappoints[i];
    if(mpt == nullptr || !mpt->IsValid()) continue;
    Eigen::Vector3d keypoint; 
    if(!frame->GetKeypointPosition(keypoint_idx, keypoint)) continue;

    int mpt_id = mpt->GetId();
    Position3d point;
    point.p = mpt->GetPosition();
    point.fixed = true;
    points.insert(std::pair<int, Position3d>(mpt_id, point));

    if(keypoint(2) > 0){
      StereoPointConstraintPtr stereo_constraint = std::shared_ptr<StereoPointConstraint>(new StereoPointConstraint()); 
      stereo_constraint->id_pose = frame_id;
      stereo_constraint->id_point = mpt_id;
      stereo_constraint->id_camera = 0;
      stereo_constraint->inlier = true;
      stereo_constraint->keypoint = keypoint;
      stereo_constraint->pixel_sigma = 0.8;
      stereo_point_constraints.push_back(stereo_constraint);
      stereo_indexes.push_back(keypoint_idx);
    }else{
      MonoPointConstraintPtr mono_constraint = std::shared_ptr<MonoPointConstraint>(new MonoPointConstraint()); 
      mono_constraint->id_pose = frame_id;
      mono_constraint->id_point = mpt_id;
      mono_constraint->id_camera = 0;
      mono_constraint->inlier = true;
      mono_constraint->keypoint = keypoint.head(2);
      mono_constraint->pixel_sigma = 0.8;
      mono_point_constraints.push_back(mono_constraint);
      mono_indexes.push_back(keypoint_idx);
    }
  }
  if(points.size() < 50) return;


  int num_inliers = FrameOptimization(poses, points, lines, velocities, biases, camera_list, 
    mono_point_constraints, stereo_point_constraints, mono_line_constraints, stereo_line_constraints,
    imu_constraints, Rwg, _configs.map_optimization_config);

  if(num_inliers < 50) return;

  std::vector<bool> outlier_mappoints(frame->FeatureNum(), false);
  for(size_t i = 0; i < mono_point_constraints.size(); i++){
    if(!mono_point_constraints[i]->inlier){
      outlier_mappoints[mono_indexes[i]] = true;
    }
  }
  for(size_t i = 0; i < stereo_point_constraints.size(); i++){
    if(!stereo_point_constraints[i]->inlier){
      outlier_mappoints[stereo_indexes[i]] = true;
    }
  }

  // find more matches
  int loop_frame_id = loop_frame->GetFrameId();
  std::set<FramePtr> loop_group_frames = group_candidates[loop_frame].group_frames;
  loop_group_frames.erase(loop_frame);

  Eigen::Matrix3d Rwq = poses.begin()->second.R;
  Eigen::Vector3d twq = poses.begin()->second.p;
  Eigen::Matrix4d Twl = loop_frame->GetPose();
  Eigen::Matrix3d Rwl = Twl.block<3, 3>(0, 0);
  Eigen::Vector3d twl = Twl.block<3, 1>(0, 3);
  Eigen::Matrix3d Rlq = Rwl.transpose() * Rwq;
  Eigen::Vector3d tlq = Rwl.transpose() * (twq - twl);
  Eigen::Matrix3d tx;
  tx << 0, -tlq(2), tlq(1), tlq(2), 0, -tlq(0), -tlq(1), tlq(0), 0;

  //*/
  Eigen::Matrix3d camera_K;
  camera_K << _camera->Fx(), 0, _camera->Cx(), 0, _camera->Fy(), _camera->Cy(), 0, 0, 1;
  Eigen::Matrix3d F = camera_K.transpose().inverse() * tx * Rlq * camera_K;

  std::function<bool(const Eigen::Vector2d&, const Eigen::Vector2d&)> check_epipolar = [&](const Eigen::Vector2d& p1, const Eigen::Vector2d& p2){
    Eigen::Vector3d p1_, p2_;
    p1_ << p1, 1;
    p2_ << p2, 1;

    Eigen::Vector3d el = F * p1_;
    double s = el.head<2>().norm();
    double er = p2_.transpose() * el;
    er = er / s;
    return er < 10;
  };

  std::function<bool(const int&, const MappointPtr)> check_reprojection_error = [&](const int& idx, const MappointPtr mpt){
    if(!mpt || !mpt->IsValid()) return false;
    Eigen::Vector3d p_on_query_frame;
    if(!frame->GetKeypointPosition(idx, p_on_query_frame)) return false;

    Eigen::Vector3d pw = mpt->GetPosition();
    Eigen::Vector3d pc = Rwq.transpose() * (pw - twq);
    Eigen::Vector3d p_2d;
    _camera->StereoProject(p_2d, pc);

    Eigen::Vector3d diff = p_on_query_frame - p_2d;
    if(p_on_query_frame(2) > 0){
      double chi2 = diff.transpose() * diff;
      return (chi2 < _configs.map_optimization_config.mono_point);
    }else{
      double chi2 = diff.head<2>().transpose() * diff.head<2>();
      return (chi2 < _configs.map_optimization_config.stereo_point);
    }
  };

  int new_found_matches = 0;
  std::function<MappointPtr(const int&, const DBoW2::WordId&)> find_more_matches_in_group = [&](const int& idx, const DBoW2::WordId& word_id){
    
    Eigen::Matrix<float, 256, 1> query_descriptor, candidate_descriptor;
    if(!frame->GetDescriptor(idx, query_descriptor)){
      return std::shared_ptr<Mappoint>(nullptr);
    }
    FramePtr best_match_frame;
    int best_match_idx = -1;
    float best_distance = 5;

    for(const auto& kv : _database->_inverted_file[word_id]){
      FramePtr f = kv.first;
      if(loop_group_frames.find(f) == loop_group_frames.end()) continue;
      for(const int& macth_candidate_idx : kv.second){
        if(!f->GetDescriptor(macth_candidate_idx, candidate_descriptor)) continue;
        float distance = DescriptorDistance(query_descriptor, candidate_descriptor);
        if(distance < best_distance){
          best_match_frame = f;
          best_match_idx = macth_candidate_idx;
        }
      }
    }
    if(best_match_idx < 0){
      return std::shared_ptr<Mappoint>(nullptr);
    }

    MappointPtr bast_match_mpt = best_match_frame->GetMappoint(best_match_idx);
    if(check_reprojection_error(idx, bast_match_mpt)){
      new_found_matches++;
      return bast_match_mpt;
    }
    return std::shared_ptr<Mappoint>(nullptr);
  };

  for(const auto& kv : word_features){
    const DBoW2::WordId word_id = kv.first;
    for(const auto& idx : kv.second){
      MappointPtr mpt = matched_mappoints[idx];
      if(!mpt || mpt->IsBad() || outlier_mappoints[idx]){ // find match from other frames in the loop group
        MappointPtr match_mpt = find_more_matches_in_group(idx, word_id);
        matched_mappoints[idx] = match_mpt;
      }else if(!mpt->IsValid()){        // check through E/F, todo: 如果query上是双目，则用重投影误差
        bool is_good_match = true;
        int idx_on_loop_frame = mpt->GetKeypointIdx(loop_frame_id);
        Eigen::Vector3d p_on_query_frame, p_on_loop_frame;
        is_good_match = frame->GetKeypointPosition(idx, p_on_query_frame);
        if(is_good_match){
          is_good_match = loop_frame->GetKeypointPosition(idx_on_loop_frame, p_on_loop_frame);
        }
        if(is_good_match){
          is_good_match = check_epipolar(p_on_query_frame.head<2>(), p_on_loop_frame.head<2>());
        }

        if(is_good_match){
          mpt->AddObverser(frame_id, idx);
          _map->TriangulateMappoint(mpt);
        }else{ // find match from other frames in the loop group
          MappointPtr match_mpt = find_more_matches_in_group(idx, word_id);
          matched_mappoints[idx] = match_mpt;
        }
      }
    }
  }
  //*/


  // merge mappoints
  // save loop results
  LoopFramePair loop_frame_pair;
  loop_frame_pair.query_frame = frame;
  loop_frame_pair.loop_frame = loop_frame;
  loop_frame_pair.Rlq = Rlq;
  loop_frame_pair.tlq = tlq;
  loop_frame_pairs.emplace_back(loop_frame_pair);

  for(size_t i = 0; i < matched_mappoints.size(); i++){
    if(!matched_mappoints[i]) continue;
    MappointPtr mpt_on_query_frame = frame->GetMappoint(i);
    if(!mpt_on_query_frame){
      mpt_on_query_frame = matched_mappoints[i];
      frame->InsertMappoint(i, mpt_on_query_frame);
      _map->InsertMappoint(mpt_on_query_frame);
    }
    merged_mappoints[mpt_on_query_frame].insert(matched_mappoints[i]);
  }
}


void MapRefiner::PoseGraphRefinement(){
  if(_map->_mappoints.size() < 80000) return; // only for large map

  _map_mutex.lock();
  // 1. add frame vertexes and constraints between adjacent frames
  std::vector<CameraPtr> camera_list;
  camera_list.emplace_back(_camera);

  MapOfPoses poses;
  VectorOfRelativePoseConstraints relative_pose_constraints;
  std::map<int, FramePtr>::iterator it_frame = _map->_keyframes.begin();
  std::map<int, FramePtr>::iterator it_next_frame = it_frame;
  it_next_frame++;
  std::map<int, FramePtr>::iterator it_frame_end = _map->_keyframes.end();
  for(;it_next_frame != it_frame_end; it_frame++, it_next_frame++){
    const int frame_id = it_frame->first;
    FramePtr frame = it_frame->second;
    bool fix_this_frame = (frame_id == 0);
    AddFrameVertex(frame, poses, 0, fix_this_frame);

    Eigen::Matrix4d Twc1 = frame->GetPose();
    Eigen::Matrix4d Twc2 = it_next_frame->second->GetPose();
    Eigen::Matrix3d Rwc1 = Twc1.block<3, 3>(0, 0);
    Eigen::Vector3d twc1 = Twc1.block<3, 1>(0, 3);
    Eigen::Matrix3d Rwc2 = Twc2.block<3, 3>(0, 0);
    Eigen::Vector3d twc2 = Twc2.block<3, 1>(0, 3);
    Eigen::Matrix3d Rc1c2 = Rwc1.transpose() * Rwc2;
    Eigen::Vector3d tc1c2 = Rwc1.transpose() * (twc2 - twc1);

    RelativePoseConstraintPtr rpc = std::shared_ptr<RelativePoseConstraint>(new RelativePoseConstraint());
    rpc->id_pose1 = frame_id;
    rpc->id_pose2 = it_next_frame->first;
    rpc->id_camera1 = 0;
    rpc->id_camera2 = 0;
    rpc->Rc1c2 = Rc1c2;
    rpc->tc1c2 = tc1c2;
    relative_pose_constraints.push_back(rpc);
  }

  // 2. add loop constraints
  for(const LoopFramePair& loop_frame_pair : loop_frame_pairs){
    RelativePoseConstraintPtr rpc = std::shared_ptr<RelativePoseConstraint>(new RelativePoseConstraint());
    rpc->id_pose1 = loop_frame_pair.loop_frame->GetFrameId();
    rpc->id_pose2 = loop_frame_pair.query_frame->GetFrameId();
    rpc->id_camera1 = 0;
    rpc->id_camera2 = 0;
    rpc->Rc1c2 = loop_frame_pair.Rlq;
    rpc->tc1c2 = loop_frame_pair.tlq;
    relative_pose_constraints.push_back(rpc);
  }

  PoseGraphOptimization(poses, camera_list, relative_pose_constraints);

  std::map<int, int> frame_id_to_matrix_idx;
  std::vector<Eigen::Matrix3d> mpt_tr;
  std::vector<Eigen::Vector3d> mpt_tt;
  // update frame poses
  for(auto& kv : poses){
    int frame_id = kv.first;
    FramePtr frame = _map->_keyframes[frame_id];
    Pose3d pose = kv.second;
    if(pose.fixed) continue;
    Eigen::Matrix4d pose_eigen = Eigen::Matrix4d::Identity();
    pose_eigen.block<3, 3>(0, 0) = pose.R;
    pose_eigen.block<3, 1>(0, 3) = pose.p;

    Eigen::Matrix4d Two = frame->GetPose();
    Eigen::Matrix3d Rwo = Two.block<3, 3>(0, 0);
    Eigen::Vector3d two = Two.block<3, 1>(0, 3);

    // to check again
    Eigen::Matrix3d tr = pose.R * Rwo.transpose();
    Eigen::Vector3d tt = -tr * two + pose.p;
    frame_id_to_matrix_idx[frame_id] = mpt_tr.size();
    mpt_tr.emplace_back(tr);
    mpt_tt.emplace_back(tt);
    frame->SetPose(pose_eigen);
  }

  // update mappoint position
  std::map<int, int>::iterator it_mi;
  for(auto& kv : _map->_mappoints){
    MappointPtr mpt = kv.second;
    if(!mpt || !mpt->IsValid()) continue;

    const std::map<int, int>& obversers = mpt->GetAllObversers();
    for(const auto& obverser : obversers){
      it_mi = frame_id_to_matrix_idx.find(obverser.first);
      if(it_mi == frame_id_to_matrix_idx.end()) continue;

      int mi = it_mi->second;
      Eigen::Vector3d pw = mpt->GetPosition();
      Eigen::Vector3d new_pw = mpt_tr[mi] * pw + mpt_tt[mi];
      mpt->SetPosition(new_pw);
      break;
    }
  }

  // update mapline position
  for(auto& kv : _map->_maplines){
    MaplinePtr mpl = kv.second;
    if(!mpl || !mpl->IsValid()) continue;

    const std::map<int, int>& obversers = mpl->GetAllObversers();
    for(const auto& obverser : obversers){
      it_mi = frame_id_to_matrix_idx.find(obverser.first);
      if(it_mi == frame_id_to_matrix_idx.end()) continue;
      int mi = it_mi->second;
      Eigen::Matrix3d Rno = mpt_tr[mi];
      Eigen::Vector3d tno = mpt_tt[mi];

      g2o::Line3D line_3d = mpl->GetLine3D();
      g2o::SE3Quat Tno_se3(Rno, tno);
      auto Tno_line = g2o::internal::fromSE3Quat(Tno_se3);
      mpl->SetLine3D((Tno_line*line_3d));
      if(mpl->EndpointsValid()){
        Vector6d& line_v = mpl->GetEndpoints();

        Eigen::Vector3d new_endpoint1 = Rno * line_v.head<3>() + tno;
        Eigen::Vector3d new_endpoint2 = Rno * line_v.tail<3>() + tno;
        Vector6d new_line_v;
        new_line_v << new_endpoint1, new_endpoint2;
        mpl->SetEndpoints(new_line_v, false);
      }
      break;
    }
  }
  _map_mutex.unlock();
}

void MapRefiner::MergeMap(){
  _map_mutex.lock();
  MergeMappoints();
  merged_mappoints.clear();
  GlobalBA(_map, _configs.map_optimization_config, true, true, 10, 10);
  MergeMaplines();
  // GlobalBA(_map, _configs.map_optimization_config, false, true, 10, 10);
  _map_mutex.unlock();
}

void MapRefiner::MergeMappoints(){
  if(merged_mappoints.empty()){
    return;
  }

  for(auto& kv : merged_mappoints){
    kv.second.insert(kv.first);
  }

  // group mappoints to be merged, each group can be merged into one mappoint
  std::map<int, int> mpt_id_to_group_id;
  std::map<int, std::set<int>> mappoint_groups;

  std::map<int, int>::iterator it_group;
  int new_group_id = 0;
  for(const auto& kv : merged_mappoints){
    std::set<int> found_groups;
    for(const MappointPtr& mpt : kv.second){
      it_group = mpt_id_to_group_id.find(mpt->GetId());
      if(it_group != mpt_id_to_group_id.end()){
        found_groups.insert(it_group->second);
      }
    }

    int best_group_id = -1;
    if(found_groups.size() == 0){
      best_group_id = new_group_id;
      new_group_id++;
    }else if(found_groups.size() == 1){
      best_group_id = *(found_groups.begin());
    }else{
      best_group_id = *(found_groups.begin());
      for(auto group_id : found_groups){
        if(group_id == best_group_id) continue;

        for(int mpt_in_group : mappoint_groups[group_id]){
          mpt_id_to_group_id[mpt_in_group] = best_group_id;
          mappoint_groups[best_group_id].insert(mpt_in_group);
        }
        mappoint_groups.erase(group_id);
      }
    }

    for(const MappointPtr& mpt : kv.second){
      int mpt_id = mpt->GetId();
      mpt_id_to_group_id[mpt_id] = best_group_id;
      mappoint_groups[best_group_id].insert(mpt_id);
    }
  }

  std::cout << "Before merging, mappoint size: " << _map->_mappoints.size() << std::endl;

  // merge mappoint group
  int keep_num = mappoint_groups.size();
  int sum_num = 0;
  for(const auto& kv : mappoint_groups){
    sum_num += kv.second.size();
    MergeMappointGroup(kv.second);
  }
  std::cout << "After merging, mappoint size: " << _map->_mappoints.size() << std::endl;

  std::cout << "sum_num = " << sum_num << ", keep_num = " << keep_num << std::endl;
  std::cout << "Remove " << (sum_num - keep_num) << " mappoints." << std::endl;
}

void MapRefiner::MergeMappointGroup(const std::set<int>& mappoint_group){
  if(mappoint_group.size() < 2){
    return;
  }

  // 1. find the best mappoint
  int best_mpt_id = *(mappoint_group.begin());
  for(const int& mpt_id : mappoint_group){
    MappointPtr mpt = _map->_mappoints[mpt_id];
    if(mpt->IsValid()){
      best_mpt_id = mpt_id;
      break;
    }
  }

  // 2. add obversers of the best mappoint
  MappointPtr best_mpt = _map->_mappoints[best_mpt_id];
  std::map<int, int>& obversers_of_best_mpt = best_mpt->GetAllObversers();
  for(const int& mpt_id : mappoint_group){
    if(mpt_id == best_mpt_id) continue;
    MappointPtr mpt = _map->_mappoints[mpt_id];

    std::map<int, int>& obversers = mpt->GetAllObversers();
    for(const auto& kv : obversers){
      int frame_id = kv.first;
      int kpt_id = kv.second;
      if(obversers_of_best_mpt.find(frame_id) == obversers_of_best_mpt.end()){
        best_mpt->AddObverser(frame_id, kpt_id);
        _map->_keyframes[frame_id]->InsertMappoint(kpt_id, best_mpt);
        _map->_keyframes[frame_id]->SetTrackId(kpt_id, best_mpt_id);
      }
    }
  }

  if(!best_mpt->IsValid()){
    _map->TriangulateMappoint(best_mpt);
  }

  // 3. remove other mappoints in the group
  for(const int& mpt_id : mappoint_group){
    if(mpt_id == best_mpt_id) continue;
    MappointPtr mpt = _map->_mappoints[mpt_id];
    mpt->SetBad();
    _map->_mappoints.erase(mpt_id);
  }
}

void MapRefiner::MergeMaplines(){
  // 1. Associate mappoints and maplines
  // MappointPtr <-> std::set<Mapline Id>
  std::map<MappointPtr, std::set<int>> maplines_share_mappoint;
  // MaplinePtr <-> std::set<Mappoint Id>
  std::map<MaplinePtr, std::set<int>> mappoints_on_mapline;
  for(auto& kv : _map->_keyframes){
    FramePtr frame = kv.second;
    const std::vector<MaplinePtr>& maplines = frame->GetAllMaplines();

    const std::vector<std::map<int, double>>& points_on_lines = frame->GetPointsOnLines();
    for(int i = 0; i < points_on_lines.size(); i++){
      if(points_on_lines[i].empty()) continue;

      for(const auto& points_and_distances : points_on_lines[i]){
        MaplinePtr mpl = maplines[i];
        MappointPtr mpt = frame->GetMappoint(points_and_distances.first);

        if(mpt && mpl && _map->_mappoints.count(mpt->GetId())> 0 && _map->_maplines.count(mpl->GetId())>0 ){
          maplines_share_mappoint[mpt].insert(mpl->GetId());
          mappoints_on_mapline[mpl].insert(mpt->GetId());
        }
      }
    }
  }

  // 2. find mapline pairs sharing mappoints
  // Mappoint Id <-> (Mappoint Id <-> number of sharing mappoints)
  std::map<int, std::map<int, int>> point_num_shared_by_lines;
  for(auto& kv : maplines_share_mappoint){
    const std::set<int>& mpl_ids = kv.second;
    if(mpl_ids.size() < 2) continue;
    int best_mpl_id = *(mpl_ids.begin());
    for(const int& mpl_id : mpl_ids){
      if(point_num_shared_by_lines[best_mpl_id].count(mpl_id) > 0){
        point_num_shared_by_lines[best_mpl_id][mpl_id] += 1;
      }else{
        point_num_shared_by_lines[best_mpl_id].insert(std::pair<int, int>(mpl_id, 1));
      }
    }
  }

  // 3. find initial mapline groups
  std::function<bool(const MaplinePtr&, const MaplinePtr&, double)> check_is_same_line = 
      [&](const MaplinePtr& mpl1, const MaplinePtr& mpl2, double thr){
    return true;

    assert(mpl1->IsValid());
    const std::map<int, int>& obversers = mpl2->GetAllObversers();
    for(const auto& kv : obversers){
      int frame_id = kv.first;
      int line2d_id = kv.second;

      FramePtr frame = _map->GetFramePtr(frame_id);
      if(!frame) continue;;

      Eigen::Vector4d obs;
      if(frame->GetLine(line2d_id, obs)) continue;

      const Eigen::Matrix4d& Twc = frame->GetPose();
      Eigen::Matrix3d Rwc = Twc.block<3, 3>(0, 0);
      Eigen::Vector3d twc = Twc.block<3, 1>(0, 3);
      Eigen::Matrix3d Rcw = Rwc.transpose();
      Eigen::Matrix4d Tcw = Eigen::Matrix4d::Identity();
      Tcw.block<3, 3>(0, 0) = Rcw;
      Tcw.block<3, 1>(0, 3) = -Rcw * twc;

      g2o::Isometry3 Tcw_iso(Tcw);
      g2o::Line3D lw = mpl1->GetLine3D();
      g2o::Line3D lc = Tcw_iso * lw;

      CameraPtr camera = frame->GetCamera();
      double fx = camera->Fx();
      double fy = camera->Fy();
      double cx = camera->Cx();
      double cy = camera->Cy();
      double W = camera->ImageWidth();
      double H = camera->ImageHeight();
      Eigen::Vector3d Kv;
      Kv << -fy * cx, -fx * cy, fx * fy;

      Eigen::Vector3d w = lc.w();
      Eigen::Vector3d line_2d;
      line_2d(0) = fy * w(0);
      line_2d(1) = fx * w(1);
      line_2d(2) = Kv.transpose() * w;

      double line_2d_norm = line_2d.head(2).norm();
      Eigen::Vector2d error;
      error(0) = obs(0) * line_2d(0) + obs(1) * line_2d(1) + line_2d(2);
      error(1) = obs(2) * line_2d(0) + obs(3) * line_2d(1) + line_2d(2);
      error = error / line_2d_norm;

      double error_thr = H * W * thr * thr;
      if(error(0) * error(0) > error_thr || error(1) * error(1) > error_thr){
        return false;
      }
    }
    return true;
  };

  const int SharingMappointNum1 = 3;
  const int SharingMappointNum2 = 5;
  std::map<int, std::set<int>> mapline_groups;
  for(auto& kv : point_num_shared_by_lines){
    int best_mpl_id = kv.first;
    for(auto& map_id_map_sharing_num : kv.second){
      int mpl_id = map_id_map_sharing_num.first;
      int sharing_mappoint_num = map_id_map_sharing_num.second;
      if(sharing_mappoint_num < SharingMappointNum1){
        continue;
      }else if(sharing_mappoint_num < SharingMappointNum2){
        // check whther the two maplines are the same mapline
        MaplinePtr best_mpl = _map->_maplines[best_mpl_id];
        MaplinePtr mpl = _map->_maplines[mpl_id];

        if(best_mpl->IsValid()){
          if(!check_is_same_line(best_mpl, mpl, 0.25)){
            continue;
          }
        }else if(mpl->IsValid()){
          if(!check_is_same_line(mpl, best_mpl, 0.25)){
            continue;
          }
        }else{
          continue;
        }
      }
      mapline_groups[best_mpl_id].insert(mpl_id);
    }
  }

  for(auto& kv : mapline_groups){
    kv.second.insert(kv.first);
  }


  // 3. merge mapline groups
  // Group Id <-> (Mapline Ids), each group can be merged into a mapline
  std::map<int, std::set<int>> merged_mapline_groups;
  std::map<int, int> mpl_id_to_group_id;
  int new_group_id = 0;
  std::map<int, int>::iterator it_group;
  for(auto& kv : mapline_groups){
    std::set<int> found_groups;
    for(const int& mpl_id : kv.second){
      it_group = mpl_id_to_group_id.find(mpl_id);
      if(it_group != mpl_id_to_group_id.end()){
        found_groups.insert(it_group->second);
      }
    }

   int best_group_id = -1;
    if(found_groups.size() == 0){
      best_group_id = new_group_id;
      new_group_id++;
    }else if(found_groups.size() == 1){
      best_group_id = *(found_groups.begin());
    }else{
      best_group_id = *(found_groups.begin());
      for(auto group_id : found_groups){
        if(group_id == best_group_id) continue;

        for(int mpl_in_group : merged_mapline_groups[group_id]){
          mpl_id_to_group_id[mpl_in_group] = best_group_id;
          merged_mapline_groups[best_group_id].insert(mpl_in_group);
        }
        merged_mapline_groups.erase(group_id);
      }
    }

    for(const int& mpl_id : kv.second){
      mpl_id_to_group_id[mpl_id] = best_group_id;
      merged_mapline_groups[best_group_id].insert(mpl_id);
    }
  }

  std::cout << "Before merging, _maplines size: " << _map->_maplines.size() << std::endl;

  // merge maplines
  int keep_num = merged_mapline_groups.size();
  int sum_num = 0;
  for(auto& kv : merged_mapline_groups){
    sum_num += kv.second.size();
    MergeMaplineGroup(kv.second);
  }

  std::cout << "After merging, _maplines size: " << _map->_maplines.size() << std::endl;

  std::cout << "sum_num = " << sum_num << ", keep_num = " << keep_num << std::endl;
  std::cout << "Remove " << (sum_num - keep_num) << " maplines." << std::endl;

}

void MapRefiner::MergeMaplineGroup(const std::set<int>& mapline_group){
  if(mapline_group.size() < 2){
    return;
  }

  // 1. find the best mapline
  int best_mpl_id = *(mapline_group.begin());
  for(const int& mpl_id : mapline_group){
    MaplinePtr mpl = _map->_maplines[mpl_id];
    if(mpl->IsValid()){
      best_mpl_id = mpl_id;
      break;
    }
  }

  // 2. add obversers of the best mapline
  MaplinePtr best_mpl = _map->_maplines[best_mpl_id];
  const std::map<int, int>& obversers_of_best_mpl = best_mpl->GetAllObversers();
  for(const int& mpl_id : mapline_group){
    if(mpl_id == best_mpl_id) continue;
    MaplinePtr mpl = _map->_maplines[mpl_id];

    const std::map<int, int>& obversers = mpl->GetAllObversers();
    for(const auto& kv : obversers){
      int frame_id = kv.first;
      int line2d_id = kv.second;
      if(obversers_of_best_mpl.find(frame_id) == obversers_of_best_mpl.end()){
        best_mpl->AddObverser(frame_id, line2d_id);
        _map->_keyframes[frame_id]->InsertMapline(line2d_id, best_mpl);
        _map->_keyframes[frame_id]->SetTrackId(line2d_id, best_mpl_id);
      }
    }
  }

  if(!best_mpl->IsValid()){
    _map->TriangulateMaplineByMappoints(best_mpl);
  }

  // 3. remove other mappoints in the group
  for(const int& mpl_id : mapline_group){
    if(mpl_id == best_mpl_id) continue;
    MaplinePtr mpl = _map->_maplines[mpl_id];
    mpl->SetBad();
    _map->_maplines.erase(mpl_id);
  }
}

void MapRefiner::BuildJunctionDatabase(){
  // train vocabulary
  const int k = 10;
  const int L = 3;
  const DBoW2::WeightingType weight = DBoW2::WeightingType::TF_IDF;
  const DBoW2::ScoringType scoring = DBoW2::ScoringType::L1_NORM;

  SuperpointVocabularyPtr jun_voc = std::shared_ptr<SuperpointVocabulary>(new SuperpointVocabulary(k, L, weight, scoring));
  std::vector<std::vector<Eigen::Matrix<float, 256, 1>>> features;
  features.reserve(_map->_keyframes.size());
  _map_mutex.lock();
  for(const auto& kv : _map->_keyframes){
    FramePtr frame = kv.second;
    const Eigen::Matrix<float, 259, Eigen::Dynamic>& junctions = frame->GetJunctions();
    std::vector<Eigen::Matrix<float, 256, 1>> frame_feature;
    frame_feature.reserve(junctions.cols());
    for(int j = 0; j < junctions.cols(); j++){
      frame_feature.emplace_back(junctions.block(3, j, 256, 1));
    }
    features.emplace_back(frame_feature);

    frame->FindJunctionConnections();
  }
  _map_mutex.unlock();

  jun_voc->create(features);

  // build junction database
  DatabasePtr _junction_database = std::shared_ptr<Database>(new Database(jun_voc));
  _map_mutex.lock();
  for(const auto& kv : _map->_keyframes){
    FramePtr frame = kv.second;
    Eigen::Matrix<float, 259, Eigen::Dynamic> junctions = frame->GetJunctions();
    DBoW2::WordIdToFeatures word_features; 
    DBoW2::BowVector bow_vector;
    std::vector<DBoW2::WordId> word_of_features;
    _junction_database->FrameToBow(junctions, word_features, bow_vector, word_of_features);
    _junction_database->AddFrame(frame, word_features, bow_vector);
  }

  _map->_junction_database = _junction_database;
  _map->_junction_voc = jun_voc;
  _map_mutex.unlock();
}

void MapRefiner::SaveTrajectory(std::string save_path){
  _map_mutex.lock();
  _map->SaveKeyframeTrajectory(save_path);
  _map_mutex.unlock();
}

void MapRefiner::GlobalMapOptimization(){
  _map_mutex.lock();
  GlobalBA(_map, _configs.map_optimization_config, true, true, 50, 40);
  _map_mutex.unlock();
}

void MapRefiner::SaveFinalMap(std::string map_root){
  _map_mutex.lock();

  // 1. remove invalid mappoints and maplines
  int invalid_mappoint_num = _map->RemoveInValidMappoints();
  int invalid_mapline_num = _map->RemoveInValidMaplines();

  _map->_database = _database;

  std::string map_path = ConcatenateFolderAndFileName(map_root, "AirSLAM_mapv1.bin");
  std::ofstream ofs(map_path, std::ios::binary);
  std::cout << "map_path = " << map_path << std::endl;
  boost::archive::binary_oarchive oa(ofs);
  oa << _map;
  _map_mutex.unlock();
}

void MapRefiner::PubMap(){
  ros::Rate loop_rate(5); 
  while(ros::ok() && !_stop){
    _map_mutex.lock();
    if(_map_ready){
      _map->Publish(ros::Time::now().toSec(), true);
    }
    _map_mutex.unlock();
    ros::spinOnce(); 
    loop_rate.sleep(); 
  }
  std::cout << "PubMap is over" << std::endl;
  _stopped = true;
}

void MapRefiner::StopVisualization(){
  _stop = true;
  while(!_stopped){
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  _visualization_thread.join();

  std::cout << "_visualization_thread is stopped" << std::endl;
  _ros_publisher->ShutDown();
}

void MapRefiner::Wait(int breakpoint){
  if(!breakpoint) return;

  std::cout << "Please to press c for the next step..." << std::endl;
  char input;
  while(1){
    std::cin >> input;
    if(input == 'c'){
      break;
    }
  }
}
