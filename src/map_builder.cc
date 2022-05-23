#include "map_builder.h"

#include <assert.h>
#include <iostream> 
#include <Eigen/Core> 
#include <Eigen/Geometry> 

#include "super_point.h"
#include "super_glue.h"
#include "read_configs.h"
#include "dataset.h"
#include "camera.h"
#include "frame.h"
#include "point_matching.h"
#include "map.h"
#include "debug.h"

MapBuilder::MapBuilder(Configs& configs): _init(false), _track_id(0), _configs(configs){
  _camera = std::shared_ptr<Camera>(new Camera(configs.camera_config_path));
  _superpoint = std::shared_ptr<SuperPoint>(new SuperPoint(configs.superpoint_config));
  if (!_superpoint->build()){
    std::cout << "Error in SuperPoint building" << std::endl;
    exit(0);
  }
  _point_matching = std::shared_ptr<PointMatching>(new PointMatching(configs.superglue_config));
  _map = std::shared_ptr<Map>(new Map(_camera));
}

void MapBuilder::AddInput(int frame_id, cv::Mat& image_left, cv::Mat& image_right){
  // undistort image
  cv::Mat image_left_rect, image_right_rect;
  _camera->UndistortImage(image_left, image_right, image_left_rect, image_right_rect);

  // extract features
  Eigen::Matrix<double, 259, Eigen::Dynamic> features_left, features_right;
  if(!_superpoint->infer(image_left_rect, features_left)){
    std::cout << "Failed when extracting features of left image !" << std::endl;
    return;
  }
  if(!_superpoint->infer(image_right_rect, features_right)){
    std::cout << "Failed when extracting features of right image !" << std::endl;
    return;
  }

  // stereo_match
  std::vector<cv::DMatch> stereo_matches;
  StereoMatch(features_left, features_right, stereo_matches);

  // construct frame
  FramePtr frame = std::shared_ptr<Frame>(new Frame(frame_id, false, _camera));
  frame->AddFeatures(features_left, features_right, stereo_matches);

  // init
  std::vector<int> track_ids(features_left.cols(), -1);
  if(!_init){
    _init = Init(frame);
    if(_init){
      _last_frame = frame;
      _last_image = image_left_rect;

      _last_keyframe = frame;
      _last_keyimage = image_left_rect;

      Eigen::Matrix4d frame_pose = frame->GetPose();
      _last_pose.p = frame_pose.block<3, 1>(0, 3);
      _last_pose.q = frame_pose.block<3, 3>(0, 0);
    }
    return;
  }

  // first track with last keyframe
  bool track_keyframe = true;
  std::vector<cv::DMatch> matches;
  int num_match = TrackFrame(_last_keyframe, frame, matches);
  if(num_match < _configs.keyframe_config.min_num_match){
    if(_num_since_last_keyframe > 1){
      // if failed, track with last frame
      track_keyframe = false;
      matches.clear();
      num_match = TrackFrame(_last_frame, frame, matches);
      if(num_match < _configs.keyframe_config.min_num_match){
        _map->InsertKeyframe(_last_frame);
        _num_since_last_keyframe = 1;
        _last_keyframe = _last_frame;
        _last_keyimage = _last_image;
        std::cout << "Insert a keyframe" << std::endl;
        return;
      }
    }else{
      return;
    }
  }

  // update last frame
  _last_frame = frame;
  _last_image = image_left_rect;
  Eigen::Matrix4d frame_pose = frame->GetPose();
  _last_pose.p = frame_pose.block<3, 1>(0, 3);
  _last_pose.q = frame_pose.block<3, 3>(0, 0);

  if(track_keyframe){
    // select keyframe
    Eigen::Matrix4d& last_keyframe_pose = _last_keyframe->GetPose();
    Eigen::Matrix3d last_R = last_keyframe_pose.block<3, 3>(0, 0);
    Eigen::Vector3d last_t = last_keyframe_pose.block<3, 1>(0, 3);
    Eigen::Matrix3d current_R = frame_pose.block<3, 3>(0, 0);
    Eigen::Vector3d current_t = frame_pose.block<3, 1>(0, 3);

    Eigen::Matrix3d delta_R = last_R.transpose() * current_R;
    Eigen::AngleAxisd angle_axis(delta_R); 
    double delta_angle = angle_axis.angle();
    double delta_distance = (current_t - last_t).norm();

    bool not_enough_match = (num_match < _configs.keyframe_config.max_num_match);
    bool large_delta_angle = (delta_angle > _configs.keyframe_config.max_angle);
    bool large_distance = (delta_distance > _configs.keyframe_config.max_distance);

    if(!(not_enough_match || large_delta_angle || large_distance)) return;
  }


  // ////// for debug //////
  // std::string save_root = _configs.saving_dir;
  // std::string debug_save_dir = ConcatenateFolderAndFileName(save_root, "debug");
  // MakeDir(debug_save_dir);  
  // std::string save_image_name = "matching_" + std::to_string((frame_id-1)) + "_" + std::to_string(frame_id) + ".jpg";
  // std::string save_image_path = ConcatenateFolderAndFileName(debug_save_dir, save_image_name);
  // std::vector<cv::KeyPoint>& last_kpts = _last_frame->GetAllKeypoints();
  // std::vector<cv::KeyPoint>& kpts = frame->GetAllKeypoints();
  // SaveMatchingResult(_last_image, last_kpts, new_image, kpts, matches, save_image_path);
  // ///////////////////////

  _map->InsertKeyframe(frame);
  _num_since_last_keyframe = 1;
  _last_keyframe = frame;
  _last_keyimage = image_left_rect;
  std::cout << "Insert a keyframe" << std::endl;
  return;
}

 void MapBuilder::StereoMatch(Eigen::Matrix<double, 259, Eigen::Dynamic>& features_left, 
      Eigen::Matrix<double, 259, Eigen::Dynamic>& features_right, std::vector<cv::DMatch>& matches){
  const double MaxYDiff = 2;
  std::vector<cv::DMatch> superglue_matches;
  _point_matching->MatchingPoints(features_left, features_right, superglue_matches);

  double min_x_dif = _camera->BF() / _camera->DepthUpperThr();
  double max_x_dif = _camera->BF() / _camera->DepthLowerThr();

  for(cv::DMatch& match : superglue_matches){
    int idx_left = match.queryIdx;
    int idx_right = match.trainIdx;

    double dx = features_left(1, idx_left) - features_right(1, idx_right);
    double dy = features_left(2, idx_left) - features_right(2, idx_right);

    if(dx > min_x_dif && dx < max_x_dif && dy <= MaxYDiff){
      matches.emplace_back(match);
    }
  }
}

bool MapBuilder::Init(FramePtr frame){
  int feature_num = frame->FeatureNum();
  if(feature_num < 150) return false;

  // construct mappoints
  std::vector<int> track_ids(feature_num, -1);
  int stereo_point_num = 0;
  int frame_id = frame->GetFrameId();
  Eigen::Vector3d tmp_position;
  std::vector<MappointPtr> new_mappoints;
  for(size_t i = 0; i < feature_num; i++){
    if(frame->BackProjectPoint(i, tmp_position)){
      stereo_point_num++;
      track_ids[i] = _track_id++;
      MappointPtr mappoint = std::shared_ptr<Mappoint>(new Mappoint(track_ids[i], tmp_position));
      mappoint->AddObverser(frame_id, i);
      frame->InsertMappoint(i, mappoint);
      new_mappoints.push_back(mappoint);
    }
  }

  // add frame and mappoints to map
  if(stereo_point_num < 100) return false;
  Eigen::Matrix4d init_pose = Eigen::Matrix4d::Identity();
  frame->SetPose(init_pose);
  frame->SetPoseFixed(true);
  _map->InsertKeyframe(frame);
  _num_since_last_keyframe = 1;
  for(MappointPtr mappoint : new_mappoints){
    _map->InsertMappoint(mappoint);
  }
  return true;
}

int MapBuilder::TrackFrame(FramePtr frame0, FramePtr frame1, std::vector<cv::DMatch>& matches){
  Eigen::Matrix<double, 259, Eigen::Dynamic>& features0 = frame0->GetAllFeatures();
  Eigen::Matrix<double, 259, Eigen::Dynamic>& features1 = frame1->GetAllFeatures();
  int num_match = _point_matching->MatchingPoints(features1, features0, matches);
  if(num_match < _configs.keyframe_config.min_num_match){
    return num_match;
  }
  std::vector<MappointPtr> matched_mappoints(features1.cols(), nullptr);
  std::vector<MappointPtr>& frame0_mappoints = frame0->GetAllMappoints();
  for(auto& match : matches){
    int idx1 = match.queryIdx;
    int idx0 = match.trainIdx;
    matched_mappoints[idx1] = frame0_mappoints[idx0];
  }
  
  std::vector<int> inliers(-1, frame1->FeatureNum());
  int num_inliers = FramePoseOptimization(frame1, matched_mappoints, inliers);

  // update track id
  if(num_inliers > _configs.keyframe_config.min_num_match){
    for(auto& match : matches){
      int idx1 = match.queryIdx;
      int idx0 = match.trainIdx;
      if(inliers[idx1] > 0 || frame0_mappoints[idx0]->GetType() == Mappoint::Type::UnTriangulated){
        frame1->SetTrackId(idx1, frame0->GetTrackId(idx0));
        frame1->InsertMappoint(idx1, frame0_mappoints[idx0]);
      }
    }
  }
  return num_inliers;
}

int MapBuilder::FramePoseOptimization(
    FramePtr frame, std::vector<MappointPtr>& mappoints, std::vector<int>& inliers){
  MapOfPoses poses;
  MapOfPoints3d points;
  std::vector<CameraPtr> camera_list;
  VectorOfPointConstraints point_constraints;
  std::vector<int> fixed_poses;
  std::vector<int> fixed_points;
 
  camera_list.emplace_back(_camera);

  // MapOfPoses
  Pose3d pose;
  pose.p = _last_pose.p;
  pose.q = _last_pose.q;
  int frame_id = frame->GetFrameId();    
  poses.insert(std::pair<int, Pose3d>(frame_id, pose));  

  // visual constraint construction
  for(size_t i = 0; i < mappoints.size(); i++){
    // points
    MappointPtr mpt = mappoints[i];
    if(!mpt->IsValid()) continue;
    int mpt_id = mpt->GetId();
    Position3d point;
    point.p = mpt->GetPosition();
    points.insert(std::pair<int, Position3d>(mpt_id, point));
    fixed_points.push_back(i);

    // visual constraint
    PointConstraint point_constraint;
    point_constraint.id_pose = frame_id;
    point_constraint.id_point = mpt_id;
    point_constraint.id_camera = 0;
    point_constraint.pixel_sigma = 0.8;
    point_constraints.push_back(point_constraint);
    inliers[i] = mpt_id;
  }

  int num_inliers = Optimize(poses, points, camera_list, point_constraints, fixed_poses, fixed_points, inliers);

  if(num_inliers > _configs.keyframe_config.min_num_match){
    // set frame pose
    Eigen::Matrix4d frame_pose = Eigen::Matrix4d::Identity();
    frame_pose.block<3, 3>(0, 0) = pose.q.matrix();
    frame_pose.block<3, 1>(0, 3) = pose.p;
    frame->SetPose(frame_pose);

    // update tracked mappoints
    for(size_t i = 0; i < inliers.size(); i++){
      int mpt_id = inliers[i];
      if(mpt_id > 0){
        frame->InsertMappoint(i, _map->GetMappointPtr(mpt_id));
      }
    }
  }

  return num_inliers;
}

// void MapBuilder::InsertKeyframeToMap(FramePtr frame){
//   // construct new mappoint
//   std::vector<MappointPtr> new_mappoints
//   std::vector<int>& track_ids = frame->GetAllTrackIds();
//   std::unordered_map<int, int> track_ids_map;
//   for(int i = 0; i < track_ids.size(); i++){
//     track_ids_map.insert(std::pair<int, int>(track_ids[i], i));
//   }
//   std::vector<int>& last_track_ids = _last_keyframe->GetAllTrackIds();
//   for(int j = 0; j < last_track_ids.size(); j++){
//     if(last_track_ids[j] < 0) continue;

//     MappointPtr mappoint = _map->GetMappointPtr(last_track_ids[j]);
//     if(track_ids_map.count(last_track_ids[j]) > 0){
//       if(mappoint){
//         mappoint->AddObverser(frame->GetFrameId(), track_ids_map[j]);
//       }else{
//         MappointPtr new_mappoint = std::shared_ptr<Mappoint>(new Mappoint(last_track_ids[j]));
//         new_mappoint->AddObverser(frame->GetFrameId(), track_ids_map[j]);
//         new_mappoint->AddObverser(_last_keyframe->GetFrameId(), j);
//         frame->InsertMappoint(new_mappoint);
//         _last_keyframe->InsertMappoint(new_mappoint);
//         _map->InsertMappoint(new_mappoint);
//         _map->
//       }
//     }

//     if(mappoint){
      
//     }else{

//     }
//   }

// }


void MapBuilder::GlobalBundleAdjust(){
  _map->GlobalBundleAdjust();
}

void MapBuilder::SaveMap(const std::string& map_root){
  _map->SaveMap(map_root);
}