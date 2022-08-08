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
#include "g2o_optimization/g2o_optimization.h"
#include "timer.h"
#include "debug.h"

// INITIALIZE_TIMER;

MapBuilder::MapBuilder(Configs& configs): _init(false), _track_id(0), _line_track_id(0), 
    _to_update_local_map(false), _configs(configs){
  _camera = std::shared_ptr<Camera>(new Camera(configs.camera_config_path));
  _superpoint = std::shared_ptr<SuperPoint>(new SuperPoint(configs.superpoint_config));
  if (!_superpoint->build()){
    std::cout << "Error in SuperPoint building" << std::endl;
    exit(0);
  }
  _point_matching = std::shared_ptr<PointMatching>(new PointMatching(configs.superglue_config));
  _line_detector = std::shared_ptr<LineDetector>(new LineDetector(configs.line_detector_config));
  _ros_publisher = std::shared_ptr<RosPublisher>(new RosPublisher(configs.ros_publisher_config));
  _map = std::shared_ptr<Map>(new Map(_camera, _ros_publisher));
}

void MapBuilder::AddInput(int frame_id, cv::Mat& image_left, cv::Mat& image_right, double timestamp){
  // undistort image 
  cv::Mat image_left_rect, image_right_rect;

  // START_TIMER;; 
  _camera->UndistortImage(image_left, image_right, image_left_rect, image_right_rect);
  // STOP_TIMER("UndistortImage");

  // extract features
  // START_TIMER;;
  Eigen::Matrix<double, 259, Eigen::Dynamic> features_left, features_right;
  std::vector<Eigen::Vector4d> lines_left, lines_right;
  if(!_superpoint->infer(image_left_rect, features_left)){
    std::cout << "Failed when extracting features of left image !" << std::endl;
    return;
  }
  if(!_superpoint->infer(image_right_rect, features_right)){
    std::cout << "Failed when extracting features of right image !" << std::endl;
    return;
  }
  _line_detector->LineExtractor(image_left_rect, lines_left);
  _line_detector->LineExtractor(image_right_rect, lines_right);
  // STOP_TIMER("Superpoint");
  // START_TIMER;;

  // stereo_match
  std::vector<cv::DMatch> stereo_matches;
  StereoMatch(features_left, features_right, stereo_matches);
  // STOP_TIMER("StereoMatch");
  // START_TIMER;;

  // // // for debug
  // SaveStereoMatchResult(image_left, image_right, 
  //     features_left, features_right, stereo_matches, _configs.saving_dir, frame_id);
  // // //////////////////////////
  // // STOP_TIMER("SaveStereoMatchResult");
  // // START_TIMER;;


  // construct frame
  FramePtr frame = std::shared_ptr<Frame>(new Frame(frame_id, false, _camera, timestamp));
  frame->AddFeatures(features_left, features_right, lines_left, lines_right, stereo_matches);
  std::cout << "Detected feature point number = " << features_left.cols() << std::endl;
  // STOP_TIMER("Construct frame");
  // START_TIMER;;


  // // // for debug
  std::cout << "SaveStereoLineMatch" << std::endl;
  SaveStereoLineMatch(image_left_rect, image_right_rect, features_left, features_right, 
      lines_left, lines_right, frame->relation_left, frame->relation_right, 
      frame->line_left_to_right_match, _configs.saving_dir, std::to_string(frame_id));
  // // //////////////////////////
  // // STOP_TIMER("SaveStereoMatchResult");
  // // START_TIMER;;


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
      _last_frame_track_well = true;
    }else{
      feature_message->inliers = std::vector<bool>(keypoints.size(), false);
      frame_pose_message->pose = Eigen::Matrix4d::Identity();
      _last_frame_track_well = false;
    }
    _ros_publisher->PublishFeature(feature_message);
    _ros_publisher->PublishFramePose(frame_pose_message);
    return;
  }

  // first track with last keyframe
  bool track_keyframe = true;
  std::vector<cv::DMatch> matches;
  // int num_match = TrackFrame(_last_keyframe, frame, matches);
  int num_match = TrackFrame(_ref_keyframe, frame, matches);
  if(num_match < _configs.keyframe_config.min_num_match){
    if(_num_since_last_keyframe > 1 && _last_frame_track_well){
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
        _last_frame_track_well = false;
        return;
      }
    }else{
      _last_frame_track_well = false;
      return;
    }
  }
  _last_frame_track_well = true;
  // STOP_TIMER("Tracking");


  // START_TIMER;
  // int track_local_map_num = TrackLocalMap(frame, num_match);
  // UpdateReferenceFrame(frame);
  int track_local_map_num = 0;
  // STOP_TIMER("TrackLocalMap");

  // std::cout << "track_local_map_num = " << track_local_map_num << "   num_match = " << num_match << std::endl;
  num_match = (track_local_map_num > 0) ? track_local_map_num : num_match;

  // START_TIMER;

  // publish message
  {
    // std::vector<bool> inliers_feature_message(keypoints.size(), false);
    // for(cv::DMatch& match : matches){
    //   inliers_feature_message[match.trainIdx] = true;
    // }
    std::vector<bool> inliers_feature_message;
    frame->GetInlierFlag(inliers_feature_message);
    feature_message->inliers = inliers_feature_message;
    _ros_publisher->PublishFeature(feature_message);

    frame_pose_message->pose = frame->GetPose();
    _ros_publisher->PublishFramePose(frame_pose_message);
  }
  // STOP_TIMER("Publish");
  // START_TIMER;

  // // ////// for debug //////
  // if(track_keyframe){
  //   SaveTrackingResult(_last_keyimage, image_left, _last_keyframe, frame, matches, _configs.saving_dir);
  // }else{
  //   SaveTrackingResult(_last_image, image_left, _last_frame, frame, matches, _configs.saving_dir);
  // }
  // // ///////////////////////

  // STOP_TIMER("SaveTrackingResult");
  // START_TIMER;
  // update last frame
  _last_frame = frame;
  _last_image = image_left_rect;
  Eigen::Matrix4d frame_pose = frame->GetPose();
  _last_pose.p = frame_pose.block<3, 1>(0, 3);
  _last_pose.q = frame_pose.block<3, 3>(0, 0);
  std::cout << "frame_pose = " << frame_pose.block<3, 1>(0, 3).transpose() << std::endl;

  if(track_keyframe){
    // select keyframe
    // Eigen::Matrix4d& last_keyframe_pose = _last_keyframe->GetPose();
    Eigen::Matrix4d& last_keyframe_pose = _ref_keyframe->GetPose();
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
    std::cout << "large_delta_angle = " << delta_angle << std::endl; 
    std::cout << "large_distance = " << delta_distance << std::endl; 


    std::cout << "not_enough_match = " << not_enough_match 
              << "  large_delta_angle = " << large_delta_angle
              << "  large_distance = " << large_distance << std::endl;
    if(!(not_enough_match || large_delta_angle || large_distance)) return;
  }

  InsertKeyframe(frame);
  _last_keyimage = image_left_rect;
  // STOP_TIMER("InsertKeyframe");

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
      Eigen::Matrix<double, 256, 1> descriptor;
      if(!frame->GetDescriptor(i, descriptor)) continue;
      MappointPtr mappoint = std::shared_ptr<Mappoint>(new Mappoint(track_ids[i], tmp_position, descriptor));
      mappoint->AddObverser(frame_id, i);
      frame->InsertMappoint(i, mappoint);
      new_mappoints.push_back(mappoint);
    }
  }

  // construct maplines
  size_t line_num = frame->LineNum();
  std::vector<MaplinePtr> new_maplines;
  for(size_t i = 0; i < line_num; i++){
    frame->SetLineTrackId(i, _line_track_id);
    MaplinePtr mapline = std::shared_ptr<Mapline>(new Mapline(_line_track_id));
    Vector6d endpoints;
    if(frame->TriangleStereoLine(i, endpoints)){
      mapline->SetEndpoints(endpoints);
      mapline->SetObverserEndpointStatus(frame_id, 1);
    }else{
      mapline->SetObverserEndpointStatus(frame_id, 0);
    }
    mapline->AddObverser(frame_id, i);
    frame->InsertMapline(i, mapline);
    new_maplines.push_back(mapline);
    _line_track_id++;
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
  for(MaplinePtr mapline : new_maplines){
    _map->InsertMapline(mapline);
  }

  _ref_keyframe = frame;

  return true;
}

int MapBuilder::TrackFrame(FramePtr frame0, FramePtr frame1, std::vector<cv::DMatch>& matches){
  // START_TIMER;
  // point tracking
  Eigen::Matrix<double, 259, Eigen::Dynamic>& features0 = frame0->GetAllFeatures();
  Eigen::Matrix<double, 259, Eigen::Dynamic>& features1 = frame1->GetAllFeatures();
  int num_match = _point_matching->MatchingPoints(features0, features1, matches);
  if(num_match < _configs.keyframe_config.min_num_match){
    return num_match;
  }
  // STOP_TIMER("MatchingPoints");

  // line tracking
  std::vector<std::map<int, double>> points_on_lines0 = frame0->GetPointsOnLines();
  std::vector<std::map<int, double>> points_on_lines1 = frame1->GetPointsOnLines();
  std::vector<int> line_matches;
  MatchLines(points_on_line0, points_on_line1, matches, features0.cols(), features1.cols(), line_matches);

  // START_TIMER;
  std::vector<int> inliers(frame1->FeatureNum(), -1);
  std::vector<MappointPtr> matched_mappoints(features1.cols(), nullptr);
  std::vector<MappointPtr>& frame0_mappoints = frame0->GetAllMappoints();
  for(auto& match : matches){
    int idx0 = match.queryIdx;
    int idx1 = match.trainIdx;
    matched_mappoints[idx1] = frame0_mappoints[idx0];
    inliers[idx1] = frame0->GetTrackId(idx0);
  }
  
  int num_inliers = FramePoseOptimization(frame1, matched_mappoints, inliers);
  // STOP_TIMER("FramePoseOptimization");

  // START_TIMER;
  // update track id
  int RM = 0;
  if(num_inliers > _configs.keyframe_config.min_num_match){
    for(std::vector<cv::DMatch>::iterator it = matches.begin(); it != matches.end();){
      int idx0 = (*it).queryIdx;
      int idx1 = (*it).trainIdx;
      if(inliers[idx1] > 0){
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
  // STOP_TIMER("Remove outliers");
  std::cout << "origin match number = " << num_match << std::endl;

  // update line track id
  for(size_t i = 0; i < features0.cols(); i++){
    int j = line_matches[i];
    if(j < 0) continue;
    int line_track_id = frame0->GetLineTrackId(i);
    if(line_track_id >= 0){
      frame1->SetLineTrackId(j, line_track_id);
    }
  }

  return num_inliers;
}

int MapBuilder::FramePoseOptimization(
    FramePtr frame, std::vector<MappointPtr>& mappoints, std::vector<int>& inliers, int pose_init){
  // Solve PnP using opencv to get initial pose
  Eigen::Matrix4d cv_pose;
  int num_cv_inliers = 0;
  if(pose_init == 0){
    std::vector<int> cv_inliers;
    num_cv_inliers = SolvePnPWithCV(frame, mappoints, cv_pose, cv_inliers);
    std::cout << "cv_pose = " << cv_pose.block<3, 1>(0, 3).transpose() << std::endl;
  }

  // Second, optimization
  MapOfPoses poses;
  MapOfPoints3d points;
  std::vector<CameraPtr> camera_list;
  VectorOfMonoPointConstraints mono_point_constraints;
  VectorOfStereoPointConstraints stereo_point_constraints;

  camera_list.emplace_back(_camera);

  // map of poses
  Pose3d pose;
  if(pose_init == 0 && num_cv_inliers > _configs.keyframe_config.min_num_match){
    pose.p = cv_pose.block<3, 1>(0, 3);
    pose.q = cv_pose.block<3, 3>(0, 0);
  }else if(pose_init == 2){
    Eigen::Matrix4d& frame_pose = frame->GetPose();
    pose.p = frame_pose.block<3, 1>(0, 3);
    pose.q = frame_pose.block<3, 3>(0, 0);
  }else{
    pose.p = _last_pose.p;
    pose.q = _last_pose.q;
  }
  int frame_id = frame->GetFrameId();    
  poses.insert(std::pair<int, Pose3d>(frame_id, pose));  
  std::cout << "initial pose = " << pose.p.transpose() << std::endl;

  // visual constraint construction
  std::vector<size_t> mono_indexes;
  std::vector<size_t> stereo_indexes;
  for(size_t i = 0; i < mappoints.size(); i++){
    // points
    MappointPtr mpt = mappoints[i];
    if(mpt == nullptr || !mpt->IsValid()) continue;
    Eigen::Vector3d keypoint; 
    if(!frame->GetKeypointPosition(i, keypoint)) continue;

    int mpt_id = mpt->GetId();
    Position3d point;
    point.p = mpt->GetPosition();
    point.fixed = true;
    points.insert(std::pair<int, Position3d>(mpt_id, point));

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
      stereo_indexes.push_back(i);
    }else{
      MonoPointConstraintPtr mono_constraint = std::shared_ptr<MonoPointConstraint>(new MonoPointConstraint()); 
      mono_constraint->id_pose = frame_id;
      mono_constraint->id_point = mpt_id;
      mono_constraint->id_camera = 0;
      mono_constraint->inlier = true;
      mono_constraint->keypoint = keypoint.head(2);
      mono_constraint->pixel_sigma = 0.8;
      mono_point_constraints.push_back(mono_constraint);
      mono_indexes.push_back(i);
    }

  }
  // START_TIMER;;
  int num_inliers = FrameOptimization(poses, points, camera_list, mono_point_constraints, stereo_point_constraints);
  // STOP_TIMER("Optimize");

  std::cout << "poses.begin()->second.p = " << poses.begin()->second.p.transpose() << std::endl;

  if(num_inliers > _configs.keyframe_config.min_num_match){
    // set frame pose
    Eigen::Matrix4d frame_pose = Eigen::Matrix4d::Identity();
    frame_pose.block<3, 3>(0, 0) = poses.begin()->second.q.matrix();
    frame_pose.block<3, 1>(0, 3) = poses.begin()->second.p;
    frame->SetPose(frame_pose);

    // update tracked mappoints
    for(size_t i = 0; i < mono_point_constraints.size(); i++){
      size_t idx = mono_indexes[i];
      if(!mono_point_constraints[i]->inlier){
        inliers[idx] = -1;
      }
    }

    for(size_t i = 0; i < stereo_point_constraints.size(); i++){
      size_t idx = stereo_indexes[i];
      if(!stereo_point_constraints[i]->inlier){
        inliers[idx] = -1;
      }
    }

  }

  return num_inliers;
}

void MapBuilder::InsertKeyframe(FramePtr frame){
  // create new track id
  std::vector<int>& track_ids = frame->GetAllTrackIds();
  for(size_t i = 0; i < track_ids.size(); i++){
    if(track_ids[i] < 0){
      frame->SetTrackId(i, _track_id++);
    }
  }

  // create new line track id
  const std::vector<int>& line_track_ids = frame->GetAllLineTrackId();
  for(size_t i = 0; i < line_track_ids.size(); i++){
    if(line_track_ids[i] < 0){
      frame->SetLineTrackId(i, _line_track_id++);
    }
  }

  // insert keyframe to map
  _map->InsertKeyframe(frame);

  // update last keyframe
  _last_keyframe = frame;
  _num_since_last_keyframe = 1;
  _ref_keyframe = frame;
  _to_update_local_map = true;

  std::cout << "Insert a keyframe" << std::endl;
}

void MapBuilder::UpdateReferenceFrame(FramePtr frame){
  int current_frame_id = frame->GetFrameId();
  std::vector<MappointPtr>& mappoints = frame->GetAllMappoints();
  std::map<FramePtr, int> keyframes;
  for(MappointPtr mpt : mappoints){
    if(!mpt || mpt->IsBad()) continue;
    const std::map<int, int> obversers = mpt->GetAllObversers();
    for(auto& kv : obversers){
      int observer_id = kv.first;
      if(observer_id == current_frame_id) continue;
      FramePtr keyframe = _map->GetFramePtr(observer_id);
      if(!keyframe) continue;
      keyframes[keyframe]++;
    }
  }
  if(keyframes.empty()) return;

  std::pair<FramePtr, int> max_covi = std::pair<FramePtr, int>(nullptr, -1);
  for(auto& kv : keyframes){
    if(kv.second > max_covi.second){
      max_covi = kv;
    }
  }
 
  if(max_covi.first->GetFrameId() != _ref_keyframe->GetFrameId()){
    _ref_keyframe = max_covi.first;
    _to_update_local_map = true;
  }
}

void MapBuilder::UpdateLocalKeyframes(FramePtr frame){
  _local_keyframes.clear();
  std::vector<std::pair<int, FramePtr>> neighbor_frames = _ref_keyframe->GetOrderedConnections(-1);
  for(auto& kv : neighbor_frames){
    _local_keyframes.push_back(kv.second);
  }

  // int current_frame_id = frame->GetFrameId();
  // std::vector<MappointPtr>& mappoints = frame->GetAllMappoints();
  // std::map<FramePtr, int> keyframes;
  // for(MappointPtr mpt : mappoints){
  //   if(!mpt || mpt->IsBad()) continue;
  //   const std::map<int, int> obversers = mpt->GetAllObversers();
  //   for(auto& kv : obversers){
  //     int observer_id = kv.first;
  //     if(observer_id == current_frame_id) continue;
  //     FramePtr keyframe = _map->GetFramePtr(observer_id);
  //     if(!keyframe) continue;
  //     keyframes[keyframe]++;
  //   }
  // }
  // if(keyframes.empty()) return;

  // std::pair<FramePtr, int> max_covi = std::pair<FramePtr, int>(nullptr, -1);
  // _local_keyframes.clear();
  // _local_keyframes.reserve(3 * keyframes.size());
  // for(auto& kv : keyframes){
  //   if(kv.second > max_covi.second){
  //     max_covi = kv;
  //   }
  //   _local_keyframes.push_back(kv.first);
  //   kv.first->tracking_frame_id = current_frame_id;
  // }

  // for(std::vector<FramePtr>::const_iterator it = _local_keyframes.begin(), it_end = _local_keyframes.end(); it!=it_end; it++){
  //   if(_local_keyframes.size() > 80) break;
  //   FramePtr kf = *it;
  //   std::vector<std::pair<int, std::shared_ptr<Frame>>> neighbors = kf->GetOrderedConnections(10);
  //   for(auto& neighbor : neighbors){
  //     if(neighbor.second->tracking_frame_id != current_frame_id){
  //       neighbor.second->tracking_frame_id = current_frame_id;
  //       _local_keyframes.push_back(neighbor.second);
  //     }
  //   }

  //   FramePtr parent = kf->GetParent();
  //   if(parent && parent->tracking_frame_id != current_frame_id){
  //     _local_keyframes.push_back(parent);
  //   }

  //   FramePtr child = kf->GetParent();
  //   if(child && child->tracking_frame_id != current_frame_id){
  //     _local_keyframes.push_back(child);
  //   }
  // }

  // if(!max_covi.first && (max_covi.second > 10)){
  //   _ref_keyframe = max_covi.first;
  // }
}

void MapBuilder::UpdateLocalMappoints(FramePtr frame){
  _local_mappoints.clear();
  int current_frame_id = frame->GetFrameId();
  for(auto& kf : _local_keyframes){
    const std::vector<MappointPtr>& mpts = kf->GetAllMappoints();
    for(auto& mpt : mpts){
      if(mpt && mpt->IsValid() && mpt->tracking_frame_id != current_frame_id){
        mpt->tracking_frame_id = current_frame_id;
        _local_mappoints.push_back(mpt);
      }
    }
  }
}

void MapBuilder::SearchLocalPoints(FramePtr frame, std::vector<std::pair<int, MappointPtr>>& good_projections){
  int current_frame_id = frame->GetFrameId();
  std::vector<MappointPtr>& mpts = frame->GetAllMappoints();
  for(auto& mpt : mpts){
    if(mpt && !mpt->IsBad()) mpt->last_frame_seen = current_frame_id;
  }

  std::vector<MappointPtr> selected_mappoints;
  for(auto& mpt : _local_mappoints){
    if(mpt && mpt->IsValid() && mpt->last_frame_seen != current_frame_id){
      selected_mappoints.push_back(mpt);
    }
  }

  // std::cout << "selected_mappoints = " << selected_mappoints.size() << std::endl;
  _map->SearchByProjection(frame, selected_mappoints, 1, good_projections);

}

int MapBuilder::TrackLocalMap(FramePtr frame, int num_inlier_thr){
  if(_to_update_local_map){
    UpdateLocalKeyframes(frame);
    UpdateLocalMappoints(frame);
  }

  std::vector<std::pair<int, MappointPtr>> good_projections;
  SearchLocalPoints(frame, good_projections);
  if(good_projections.size() < 3) return -1;

  std::vector<MappointPtr> mappoints = frame->GetAllMappoints();
  for(auto& good_projection : good_projections){
    int idx = good_projection.first;
    if(mappoints[idx] && !mappoints[idx]->IsBad()) continue;
    mappoints[idx] = good_projection.second;
  }

  std::vector<int> inliers(mappoints.size(), -1);
  int num_inliers = FramePoseOptimization(frame, mappoints, inliers, 2);

  // std::cout << "num_inliers = " << num_inliers << "  num_inlier_thr = " << num_inlier_thr << std::endl;

  // update track id
  if(num_inliers > _configs.keyframe_config.min_num_match && num_inliers > num_inlier_thr){
    for(size_t i = 0; i < mappoints.size(); i++){
      if(inliers[i] > 0){
        frame->SetTrackId(i, mappoints[i]->GetId());
        frame->InsertMappoint(i, mappoints[i]);
      }
    }
  }else{
    num_inliers = -1;
  }
  return num_inliers;
}

void MapBuilder::SaveTrajectory(){
  _map->SaveKeyframeTrajectory(_configs.saving_dir);
}

void MapBuilder::SaveMap(const std::string& map_root){
  _map->SaveMap(map_root);
}