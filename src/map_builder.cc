#include "map_builder.h"

#include <assert.h>
#include <iostream> 
#include <Eigen/Core> 
#include <Eigen/Geometry> 
#include <opencv2/core/eigen.hpp>

#include "super_point.h"
#include "super_glue.h"
#include "read_configs.h"
#include "dataset.h"
#include "camera.h"
#include "frame.h"
#include "point_matching.h"
#include "map.h"
#include "timer.h"
#include "debug.h"


MapBuilder::MapBuilder(Configs& configs): _init(false), _track_id(0), _configs(configs){
  _ros_publisher = std::shared_ptr<RosPublisher>(new RosPublisher(configs.ros_publisher_config));
  _camera = std::shared_ptr<Camera>(new Camera(configs.camera_config_path));
  _superpoint = std::shared_ptr<SuperPoint>(new SuperPoint(configs.superpoint_config));
  if (!_superpoint->build()){
    std::cout << "Error in SuperPoint building" << std::endl;
    exit(0);
  }
  _point_matching = std::shared_ptr<PointMatching>(new PointMatching(configs.superglue_config));
  _map = std::shared_ptr<Map>(new Map(_camera, _ros_publisher));
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

  // // for debug
  SaveStereoMatchResult(image_left, image_right, 
      features_left, features_right, stereo_matches, _configs.saving_dir, frame_id);
  // //////////////////////////


  // construct frame
  FramePtr frame = std::shared_ptr<Frame>(new Frame(frame_id, false, _camera));
  frame->AddFeatures(features_left, features_right, stereo_matches);

  // message
  std::vector<cv::KeyPoint>& keypoints = frame->GetAllKeypoints();
  FeatureMessgaePtr feature_message = std::shared_ptr<FeatureMessgae>(new FeatureMessgae);
  feature_message->image = image_left;
  feature_message->keypoints = keypoints;
  FramePoseMessagePtr frame_pose_message = std::shared_ptr<FramePoseMessage>(new FramePoseMessage);


  // init
  if(!_init){
    if(stereo_matches.size() < 100) return;
    _init = Init(frame);
    if(_init){
      _last_frame = frame;
      _last_image = image_left_rect;
      _last_keyimage = image_left_rect;
      Eigen::Matrix4d frame_pose = frame->GetPose();
      _last_pose.p = frame_pose.block<3, 1>(0, 3);
      _last_pose.q = frame_pose.block<3, 3>(0, 0);
      feature_message->inliers = std::vector<bool>(keypoints.size(), true);
      frame_pose_message->pose = frame->GetPose();
    }else{
      feature_message->inliers = std::vector<bool>(keypoints.size(), false);
      frame_pose_message->pose = Eigen::Matrix4d::Identity();
    }
    _ros_publisher->PublishFeature(feature_message);
    _ros_publisher->PublishFramePose(frame_pose_message);
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
        InsertKeyframe(_last_frame);
        _last_keyimage = _last_image;
        Eigen::Matrix4d frame_pose = _last_frame->GetPose();
        _last_pose.p = frame_pose.block<3, 1>(0, 3);
        _last_pose.q = frame_pose.block<3, 3>(0, 0);
        return;
      }
    }else{
      return;
    }
  }

  // publish message
  {
    std::vector<bool> inliers_feature_message(keypoints.size(), false);
    for(cv::DMatch& match : matches){
      inliers_feature_message[match.trainIdx] = true;
    }
    feature_message->inliers = inliers_feature_message;
    _ros_publisher->PublishFeature(feature_message);

    frame_pose_message->pose = frame->GetPose();
    _ros_publisher->PublishFramePose(frame_pose_message);
  }
  

  ////// for debug //////
  if(track_keyframe){
    SaveTrackingResult(_last_keyimage, image_left, _last_keyframe, frame, matches, _configs.saving_dir);
  }else{
    SaveTrackingResult(_last_image, image_left, _last_frame, frame, matches, _configs.saving_dir);
  }
  ///////////////////////


  // update last frame
  _last_frame = frame;
  _last_image = image_left_rect;
  Eigen::Matrix4d frame_pose = frame->GetPose();
  _last_pose.p = frame_pose.block<3, 1>(0, 3);
  _last_pose.q = frame_pose.block<3, 3>(0, 0);
  std::cout << "frame_pose = " << frame_pose.block<3, 1>(0, 3).transpose() << std::endl;

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

    std::cout << "last_t = " << last_t.transpose() << std::endl; 
    std::cout << "current_t = " << current_t.transpose() << std::endl; 


    std::cout << "num_match = " << num_match << std::endl; 
    std::cout << "large_delta_angle = " << large_delta_angle << std::endl; 
    std::cout << "large_distance = " << large_distance << std::endl; 


    std::cout << "not_enough_match = " << not_enough_match 
              << "  large_delta_angle = " << large_delta_angle
              << "  large_distance = " << large_distance << std::endl;
    if(!(not_enough_match || large_delta_angle || large_distance)) return;
  }

  InsertKeyframe(frame);
  _last_keyimage = image_left_rect;
  std::cout << "last_keyframe id = " << frame->GetFrameId() << std::endl;
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

    double dx = std::abs(features_left(1, idx_left) - features_right(1, idx_right));
    double dy = std::abs(features_left(2, idx_left) - features_right(2, idx_right));

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
  frame->SetTrackIds(track_ids);
  InsertKeyframe(frame);
  for(MappointPtr mappoint : new_mappoints){
    _map->InsertMappoint(mappoint);
  }
  return true;
}

int MapBuilder::TrackFrame(FramePtr frame0, FramePtr frame1, std::vector<cv::DMatch>& matches){
  Eigen::Matrix<double, 259, Eigen::Dynamic>& features0 = frame0->GetAllFeatures();
  Eigen::Matrix<double, 259, Eigen::Dynamic>& features1 = frame1->GetAllFeatures();
  int num_match = _point_matching->MatchingPoints(features0, features1, matches);
  if(num_match < _configs.keyframe_config.min_num_match){
    return num_match;
  }
  std::vector<MappointPtr> matched_mappoints(features1.cols(), nullptr);
  std::vector<MappointPtr>& frame0_mappoints = frame0->GetAllMappoints();
  for(auto& match : matches){
    int idx0 = match.queryIdx;
    int idx1 = match.trainIdx;
    matched_mappoints[idx1] = frame0_mappoints[idx0];
  }
  std::vector<int> inliers(frame1->FeatureNum(), -1);
  int num_inliers = FramePoseOptimization(frame1, matched_mappoints, inliers);

  // update track id
  int RM = 0;
  if(num_inliers > _configs.keyframe_config.min_num_match){
    for(std::vector<cv::DMatch>::iterator it = matches.begin(); it != matches.end();){
      int idx0 = (*it).queryIdx;
      int idx1 = (*it).trainIdx;
      if(inliers[idx1] > 0 || !frame0_mappoints[idx0] || frame0_mappoints[idx0]->GetType() == Mappoint::Type::UnTriangulated){
        frame1->SetTrackId(idx1, frame0->GetTrackId(idx0));
        frame1->InsertMappoint(idx1, frame0_mappoints[idx0]);
      }

      if(inliers[idx1] > 0){
        it++;
      }else{
        it = matches.erase(it);
        RM++;
      }
    }
    std::cout << "remove " << RM << " matches" << std::endl;
  }

  return num_inliers;
}

int MapBuilder::FramePoseOptimization(
    FramePtr frame, std::vector<MappointPtr>& mappoints, std::vector<int>& inliers){
  // Solve PnP using opencv to get initial pose
  Eigen::Matrix4d cv_pose;
  std::vector<int> cv_inliers;
  int num_cv_inliers = SolvePnPWithCV(frame, mappoints, cv_pose, cv_inliers);

  std::cout << "cv_pose = " << cv_pose.block<3, 1>(0, 3).transpose() << std::endl;


  // Second, optimize using ceres to stereo constraints
  MapOfPoses poses;
  MapOfPoints3d points;
  std::vector<CameraPtr> camera_list;
  VectorOfPointConstraints point_constraints;
  std::vector<int> fixed_poses;
  std::vector<int> fixed_points;
 
  camera_list.emplace_back(_camera);

  // map of poses
  Pose3d pose;
  if(num_cv_inliers > _configs.keyframe_config.min_num_match){
    pose.p = cv_pose.block<3, 1>(0, 3);
    pose.q = cv_pose.block<3, 3>(0, 0);
  }else{
    pose.p = _last_pose.p;
    pose.q = _last_pose.q;
  }
  std::cout << "initial pose = " << pose.p.transpose() << std::endl;

  int frame_id = frame->GetFrameId();    
  poses.insert(std::pair<int, Pose3d>(frame_id, pose));  

  // visual constraint construction
  Eigen::Matrix<double, 259, Eigen::Dynamic>& features = frame->GetAllFeatures();
  std::vector<double>& u_right = frame->GetAllRightPosition();
  for(size_t i = 0; i < mappoints.size(); i++){
    // points
    MappointPtr mpt = mappoints[i];
    if(mpt == nullptr || !mpt->IsValid()) continue;
    Eigen::Vector3d keypoint; 
    if(!frame->GetKeypointPosition(i, keypoint)) continue;

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
    point_constraint.keypoint = keypoint;
    point_constraint.pixel_sigma = 0.8;
    point_constraints.push_back(point_constraint);
    inliers[i] = mpt_id;
  }
  // START_TIMER;
  int num_inliers = Optimize(poses, points, camera_list, point_constraints, fixed_poses, fixed_points, inliers);
  // STOP_TIMER("Optimize");

  std::cout << "poses.begin()->second.p = " << poses.begin()->second.p.transpose() << std::endl;

  if(num_inliers > _configs.keyframe_config.min_num_match){
    // set frame pose
    Eigen::Matrix4d frame_pose = Eigen::Matrix4d::Identity();
    frame_pose.block<3, 3>(0, 0) = poses.begin()->second.q.matrix();
    frame_pose.block<3, 1>(0, 3) = poses.begin()->second.p;
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

void MapBuilder::InsertKeyframe(FramePtr frame){
  // create new track id
  std::vector<int>& track_ids = frame->GetAllTrackIds();
  for(int i = 0; i < track_ids.size(); i++){
    if(track_ids[i] < 0){
      frame->SetTrackId(i, _track_id++);
    }
  }

  // insert keyframe to map
  _map->InsertKeyframe(frame);

  // update last keyframe
  _last_keyframe = frame;
  _num_since_last_keyframe = 1;

  std::cout << "Insert a keyframe" << std::endl;
}

void MapBuilder::GlobalBundleAdjust(){
  _map->GlobalBundleAdjust();
}

void MapBuilder::SaveMap(const std::string& map_root){
  _map->SaveMap(map_root);
}