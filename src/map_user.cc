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
#include "debug.h"
#include "map_user.h"


MapUser::MapUser(){
}

MapUser::MapUser(RelocalizationConfigs& configs, ros::NodeHandle nh): _configs(configs), _stop(false){
  _camera = std::shared_ptr<Camera>(new Camera(configs.camera_config_path));
  _feature_detector = std::shared_ptr<FeatureDetector>(new FeatureDetector(configs.plnet_config));
  _point_matcher = std::shared_ptr<PointMatcher>(new PointMatcher(configs.point_matcher_config));
  _map = std::shared_ptr<Map>(new Map());
  _ros_publisher = std::shared_ptr<RosPublisher>(new RosPublisher(_configs.ros_publisher_config, nh));

  _reloc_message = std::shared_ptr<RelocMessage>(new RelocMessage);
}

void MapUser::PubMap(){
  KeyframeMessagePtr keyframe_message = std::shared_ptr<KeyframeMessage>(new KeyframeMessage);
  MapMessagePtr map_message = std::shared_ptr<MapMessage>(new MapMessage);
  MapLineMessagePtr mapline_message = std::shared_ptr<MapLineMessage>(new MapLineMessage);

  for(auto&kv : _map->_keyframes){
    const Eigen::Matrix4d& pose_eigen = kv.second->GetPose();
    keyframe_message->times.push_back(kv.second->GetTimestamp());
    keyframe_message->ids.push_back(kv.first);
    keyframe_message->poses.emplace_back(pose_eigen);
  }

  for(auto& kv : _map->_mappoints){
    if(kv.second->IsValid()){
      map_message->ids.push_back(kv.first);
      map_message->points.push_back(kv.second->GetPosition());
    }
  }

  for(auto& kv : _map->_maplines){
    if(kv.second->IsValid() && kv.second->EndpointsValid()){
      const Vector6d& endpoints = kv.second->GetEndpoints();
      mapline_message->ids.push_back(kv.first);
      mapline_message->lines.emplace_back(endpoints); 
    }
  }

  while(!_stop){
    double current_time = ros::Time::now().toSec();
    keyframe_message->time = current_time;
    map_message->time = current_time;
    mapline_message->time = current_time;

    _ros_publisher->PublisheKeyframe(keyframe_message);
    _ros_publisher->PublishMap(map_message);
    _ros_publisher->PublishMapLine(mapline_message);  
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

void MapUser::StopVisualization(){
  _stop = true;
  _ros_publisher->ShutDown();
  _visualization_thread.join();
}

void MapUser::LoadMap(const std::string& map_root){
  std::string map_v1_path = ConcatenateFolderAndFileName(map_root, "AirSLAM_mapv1.bin");
  std::ifstream ifs(map_v1_path, std::ios::binary);
  boost::archive::binary_iarchive ia(ifs);
  ia >> _map;

  new_frame_id = _map->_keyframes.rbegin()->first;
  _database = _map->_database;

  _junction_database = _map->_junction_database;
  _junction_database->LoadVocabulary(_map->_junction_voc);

  _reloc_message->map_scale = _map->MapScale() / 80;
  
  _visualization_thread = std::thread(boost::bind(&MapUser::PubMap, this));
}
  
void MapUser::LoadVocabulary(const std::string voc_path){
  _database->LoadVocabulary(voc_path);
}

bool MapUser::Relocalization(cv::Mat& image, Eigen::Matrix4d& pose){
  cv::Mat image_rect;
  _camera->UndistortImage(image, image_rect);
  Eigen::Matrix<float, 259, Eigen::Dynamic> features, junctions;
  std::vector<Eigen::Vector4d> feature_lines;
  _feature_detector->Detect(image_rect, features, feature_lines, junctions);

  FramePtr frame = std::shared_ptr<Frame>(new Frame(new_frame_id++, false, _camera, 0));
  int frame_id = frame->GetFrameId();
  frame->AddLeftFeatures(features, feature_lines);
  frame->AddJunctions(junctions);

  ros::Time now = ros::Time::now();
  if(_configs.ros_publisher_config.feature){
    FeatureMessgaePtr feature_message = std::shared_ptr<FeatureMessgae>(new FeatureMessgae);
    feature_message->time = now.toSec();
    feature_message->image = image_rect;
    feature_message->keypoints = frame->GetAllKeypoints();
    feature_message->fm_type = FeatureMessgaeType::RelocFeature;
    feature_message->lines = frame->GatAllLines();
    _ros_publisher->PublishFeature(feature_message);
  }

  DBoW2::WordIdToFeatures word_features, junction_word_features; 
  DBoW2::BowVector bow_vector, junction_bow_vector;
  std::vector<DBoW2::WordId> word_of_features, junction_word_of_features;
  _database->FrameToBow(features, word_features, bow_vector, word_of_features);
  _junction_database->FrameToBow(junctions, junction_word_features, junction_bow_vector, junction_word_of_features);

  // query
  std::map<FramePtr, int> frame_sharing_words;
  _database->Query(bow_vector, frame_sharing_words);

  if(frame_sharing_words.empty()) return false;

  // filtration
  int max_sharing_words = 0;
  for(const auto& kv : frame_sharing_words){
    max_sharing_words = kv.second > max_sharing_words ? kv.second : max_sharing_words;
  }
  int sharing_wrods_num_thr = std::max(static_cast<int>(max_sharing_words * 0.3f), 8);

  std::map<FramePtr, int>::iterator fsw_it = frame_sharing_words.begin();
  for(; fsw_it != frame_sharing_words.end();){
    FramePtr fsw = fsw_it->first;
    if(fsw_it->second < sharing_wrods_num_thr){
      fsw_it = frame_sharing_words.erase(fsw_it);
    }else{
      fsw_it++;
    }
  }

  if(frame_sharing_words.empty()) return false;

  // scoring
  std::map<FramePtr, double> frame_scores;
  fsw_it = frame_sharing_words.begin();
  for(; fsw_it != frame_sharing_words.end(); fsw_it++){
    FramePtr fsw = fsw_it->first;
    frame_scores[fsw] = _database->Score(_database->_frame_bow_vectors[fsw], bow_vector);
  }

  int print_debug_info = 0;
  if(print_debug_info){
    std::cout << "======================= frame_sharing_words scoring =================  " << std::endl;
    for(auto& kv : frame_scores){
      double frame_time = kv.first->GetTimestamp();
      std::cout << "frmae time = " << std::fixed << std::setprecision(9) << frame_time << ", score = " << kv.second << std::endl;
    }
  }

  // grouping
  std::map<FramePtr, RelocalizationGroupCandidate> group_candidates;
  FramePtr best_deputy;
  double best_group_score = -1; 
  std::map<FramePtr, RelocalizationGroupCandidate>::iterator relocalization_group_iter;
  std::map<FramePtr, double>::iterator fs_it = frame_scores.begin();
  for(; fs_it != frame_scores.end(); fs_it++){
    FramePtr fsw = fs_it->first;
    FramePtr deputy_of_group = fsw;
    double deputy_score = fs_it->second;

    RelocalizationGroupCandidate group_candidate;
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

    relocalization_group_iter = group_candidates.find(deputy_of_group);
    if(relocalization_group_iter == group_candidates.end() || relocalization_group_iter->second.group_score < group_candidate.group_score){
      group_candidates[deputy_of_group] = group_candidate;

      if(group_candidate.group_score > best_group_score){
        best_group_score = group_candidate.group_score;
        best_deputy = deputy_of_group;
      }
    }   
  }

  if(best_group_score < 0) return false;

  // only accumulate the scores of top 5 frames for each group
  const size_t CoviFrameScoreNum = 5;
  best_group_score = 0.0;
  for(auto& kv : group_candidates){
    std::vector<double> group_scores;
    for(FramePtr f : kv.second.group_frames){
      group_scores.push_back(frame_scores[f]);
    }

    if(group_scores.size() > CoviFrameScoreNum){
      std::sort(group_scores.rbegin(), group_scores.rend()); 
    }

    double sum_group_score = 0;
    int N = std::min(CoviFrameScoreNum, group_scores.size());
    for(size_t i = 0; i < N; i++){
      sum_group_score += group_scores[i];
    }

    kv.second.group_score = sum_group_score;
    best_group_score = std::max(best_group_score, sum_group_score);
  }

  if(print_debug_info){
    std::cout << "======================= frame_sharing_words grouping =================  " << std::endl;
    std::cout << "best_group_score = " << best_group_score << std::endl;
    for(auto& kv : group_candidates){
      double frame_time = kv.first->GetTimestamp();
      std::cout << "frmae time = " << std::fixed << std::setprecision(9) << frame_time << ", group_score = " << kv.second.group_score << std::endl;
    }
  }

  // filtration of group
  if(group_candidates.size() > 3){
    double group_score_thr = best_group_score * 0.5;
    auto it = group_candidates.begin();
    for(; it != group_candidates.end();){
      if(it->second.group_score < group_score_thr){  
        it = group_candidates.erase(it);
      }else{
        it++;
      }
    }
  }

  // sorting of group
  std::vector<std::pair<FramePtr, RelocalizationGroupCandidate>> group_vector(group_candidates.begin(), group_candidates.end());
  std::sort(group_vector.begin(), group_vector.end(), [](const auto &a, const auto &b) {
      return a.second.group_score > b.second.group_score;
  });

  if(print_debug_info){
    std::cout << "======================= sorting of group =================  " << std::endl;
    for(auto& kv : group_vector){
      double frame_time = kv.first->GetTimestamp();
      double junction_frame_scores = _junction_database->Score(_junction_database->_frame_bow_vectors[kv.first], junction_bow_vector);
      std::cout << "frmae time = " << std::fixed << std::setprecision(9) << frame_time 
                << ", group_score = " << kv.second.group_score 
                << ", junction_frame_scores = " << junction_frame_scores
                << std::endl;

    }
  }

  // std::cout << "======================= Find same sentences =================  " << std::endl;
  FrameFeatures::iterator ffit;
  frame->FindJunctionConnections();
  const std::vector<std::set<int>>& junction_connections = frame->GetJunctionConnections();
  const int CurrentJunctionNum = frame->JunctionNum();
  for(auto& kv : group_vector){
    FramePtr kf = kv.first;
    const int KFJunctionNum = kf->JunctionNum();
    std::vector<std::vector<bool>> match_matrix(CurrentJunctionNum, std::vector<bool>(KFJunctionNum, false));

    std::vector<std::vector<int>> match_junctions;
    match_junctions.resize(CurrentJunctionNum);
    // for each group
    for(int i = 0; i < junction_word_of_features.size(); i++){
      DBoW2::WordId word_id = junction_word_of_features[i];
      if(word_id >= UINT_MAX) continue;
      ffit = _junction_database->_inverted_file[word_id].find(kf);
      if(ffit == _junction_database->_inverted_file[word_id].end()) continue;

      match_junctions[i] = ffit->second;
      for(const int& j : ffit->second){
        match_matrix[i][j] = true;
      }
    }

    const std::vector<std::set<int>>& kf_junction_connections = kf->GetJunctionConnections();
    int match_num = 0;
    int line_match_num = 0;
    for(int i = 0; i < match_junctions.size(); i++){
      if(match_junctions[i].size() < 1 || junction_connections[i].size() < 1) continue;
      match_num += match_junctions[i].size();
      for(const int junction_id : match_junctions[i]){
        if(kf_junction_connections[junction_id].size() < 1) continue;
        for(const int& idx1 : junction_connections[i]){
          for(const int& idx2: kf_junction_connections[junction_id]){
            if(match_matrix[idx1][idx2]){
              line_match_num++;
            }
          }
        }
      }

    }

    double rate = (match_num > 0) ? (double)line_match_num / match_num : 0;
    double junction_frame_scores = _junction_database->Score(_junction_database->_frame_bow_vectors[kv.first], junction_bow_vector);
    kv.second.group_score += (junction_frame_scores * (1 + rate));

    if(print_debug_info){
      std::cout << "frmae time = " << std::fixed << std::setprecision(9) << kf->GetTimestamp() 
                << ", rate = " << rate 
                << ", score = " << _junction_database->Score(_junction_database->_frame_bow_vectors[kv.first], junction_bow_vector) 
                << ", match_num = " << match_num 
                << ", line_match_num = " << line_match_num 
                << std::endl;

      std::string image_root = "/media/bssd/datasets/tartanair/euroc_style/with_time/abandonedfactory/P000/cam0/data";
      std::string save_root = "/media/bssd/datasets/tartanair/mapping_relocalization/results/tmp/debug/relo_sentence_debug";
      DrawDbowJunctionMatchingResults(frame, image_rect, kf, match_matrix, image_root, save_root);
    }
  }

  std::sort(group_vector.begin(), group_vector.end(), [](const auto &a, const auto &b) {
      return a.second.group_score > b.second.group_score;
  });

  if(print_debug_info){
    std::cout << "======================= re-sorting of group after detecting sentences =================  " << std::endl;
    for(auto& kv : group_vector){
      double frame_time = kv.first->GetTimestamp();
      std::cout << "frmae time = " << std::fixed << std::setprecision(9) << frame_time << ", group_score = " << kv.second.group_score << std::endl;
    }
  }


  // feature matching
  std::vector<cv::DMatch> relocalization_matches;
  FramePtr relocalization_frame;
  const size_t GoodCandidateNum = std::min((size_t)3, group_vector.size());    
  const Eigen::Matrix<float, 259, Eigen::Dynamic>& query_features = frame->GetAllFeatures();
  for(size_t i = 0; i < GoodCandidateNum; i++){
    FramePtr good_candidate = group_vector[i].first;
    const Eigen::Matrix<float, 259, Eigen::Dynamic>& good_candidate_features = good_candidate->GetAllFeatures();
    std::vector<cv::DMatch> matches;
    _point_matcher->MatchingPoints(query_features, good_candidate_features, matches, true);
    if(matches.size() > relocalization_matches.size()){
      relocalization_matches = matches;
      relocalization_frame = good_candidate;
    }

    // if(relocalization_matches.size() > 50) break;
  }
  if(relocalization_matches.size() < _configs.min_inlier) return false;

  // std::map<int, MappointPtr> matched_mappoints;
  std::vector<MappointPtr> matched_mappoints(frame->FeatureNum(), nullptr);
  for(const auto& match : relocalization_matches){
    MappointPtr mpt = relocalization_frame->GetMappoint(match.trainIdx);
    matched_mappoints[match.queryIdx] = mpt;
  }

  std::vector<int> cv_inliers;
  Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
  int num_inliers = SolvePnPWithCV(frame, matched_mappoints, Twc, cv_inliers);
  frame->SetPose(Twc);
  pose = Twc;

  if(_configs.pose_refinement){
    // pose estimation
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

    if(points.size() < _configs.min_inlier) return false;

    num_inliers = FrameOptimization(poses, points, lines, velocities, biases, camera_list, 
      mono_point_constraints, stereo_point_constraints, mono_line_constraints, stereo_line_constraints,
      imu_constraints, Rwg, _configs.pose_estimation_config);

    pose = Eigen::Matrix4d::Identity();
    pose.block<3, 3>(0, 0) = poses.begin()->second.R;
    pose.block<3, 1>(0, 3) = poses.begin()->second.p;
  }


  if(num_inliers < _configs.min_inlier) return false;

  // visualization
  if(_configs.ros_publisher_config.reloc){
    FramePoseMessagePtr frame_pose_message = std::shared_ptr<FramePoseMessage>(new FramePoseMessage);
    frame_pose_message->time = now.toSec();
    frame_pose_message->pose = pose;

    _reloc_message->times.push_back(now.toSec());
    _reloc_message->poses.push_back(pose);
    _reloc_message->mappoints.clear();
    _reloc_message->mappoints.reserve(num_inliers);
    for(int i = 0; i < matched_mappoints.size(); i++){
      if(cv_inliers[i] >= 0){
        _reloc_message->mappoints.push_back(matched_mappoints[i]->GetPosition()); 
      }
    }

    _ros_publisher->PublishFramePose(frame_pose_message);
    _ros_publisher->PubRelocResults(_reloc_message);
  }

  return true;
}

Eigen::Matrix4d MapUser::GetBaseFramePose(){
  return _map->_keyframes.begin()->second->GetPose();
}

double MapUser::GetBaseFrameTimestamp(){
  return _map->_keyframes.begin()->second->GetTimestamp();
}