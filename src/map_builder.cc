#include "map_builder.h"

#include <assert.h>
#include <iostream> 
#include <Eigen/Core> 
#include <Eigen/Geometry> 
#include <opencv2/core/eigen.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "super_point.h"
#include "super_glue.h"
#include "read_configs.h"
#include "imu.h"
#include "dataset.h"
#include "camera.h"
#include "frame.h"
#include "point_matcher.h"
#include "map.h"
#include "g2o_optimization/g2o_optimization.h"
#include "timer.h"
#include "debug.h"

MapBuilder::MapBuilder(VisualOdometryConfigs& configs, ros::NodeHandle nh): _shutdown(false), _feature_thread_stop(false), 
    _tracking_trhead_stop(false), _init(false), _insert_next_keyframe(false), _track_id(0), _line_track_id(0), _configs(configs){
  _camera = std::shared_ptr<Camera>(new Camera(configs.camera_config_path));
  _preinteration_keyframe.SetNoiseAndWalk(_camera->GyrNoise(), _camera->AccNoise(), _camera->GyrWalk(), _camera->AccWalk());
  _point_matcher = std::shared_ptr<PointMatcher>(new PointMatcher(configs.point_matcher_config));
  _feature_detector = std::shared_ptr<FeatureDetector>(new FeatureDetector(configs.plnet_config));
  _ros_publisher = std::shared_ptr<RosPublisher>(new RosPublisher(configs.ros_publisher_config, nh));
  _map = std::shared_ptr<Map>(new Map(_configs.backend_optimization_config, _camera, _ros_publisher));

  _feature_thread = std::thread(boost::bind(&MapBuilder::ExtractFeatureThread, this));
  _tracking_thread = std::thread(boost::bind(&MapBuilder::TrackingThread, this));
}

bool MapBuilder::UseIMU(){
  return _camera->UseIMU();
}

void MapBuilder::AddInput(InputDataPtr data){
  cv::Mat image_left_rect, image_right_rect;
  _camera->UndistortImage(data->image_left, data->image_right, image_left_rect, image_right_rect);
  data->image_left = image_left_rect;
  data->image_right = image_right_rect;

  while(_data_buffer.size() > 3 && !_shutdown){
    usleep(2000);
  }

  _buffer_mutex.lock();
  _data_buffer.push(data);
  _buffer_mutex.unlock();
}

void MapBuilder::ExtractFeatureThread(){
  while(!_shutdown || !_data_buffer.empty()){
    if(_data_buffer.empty()){
      usleep(2000);
      continue;
    }

    InputDataPtr input_data;
    _buffer_mutex.lock();
    input_data = _data_buffer.front();
    _data_buffer.pop();
    _buffer_mutex.unlock();

    int frame_id = input_data->index;
    double timestamp = input_data->time;
    cv::Mat image_left_rect = input_data->image_left.clone();
    cv::Mat image_right_rect = input_data->image_right.clone();


    // construct frame
    FramePtr frame = std::shared_ptr<Frame>(new Frame(frame_id, false, _camera, timestamp));

    Eigen::Matrix<float, 259, Eigen::Dynamic> left_features, right_features; 
    std::vector<Eigen::Vector4d> left_lines, right_lines;
    std::vector<cv::DMatch> matches, stereo_matches;
    int good_stereo_point = 0;
    FrameType frame_type;
    if(!_init || _insert_next_keyframe){
      Eigen::Matrix<float, 259, Eigen::Dynamic> junctions;
      _feature_detector->Detect(image_left_rect, image_right_rect, left_features, right_features, left_lines, right_lines, junctions);
      _point_matcher->MatchingPoints(left_features, right_features, stereo_matches, false);
      frame->AddLeftFeatures(left_features, left_lines);
      good_stereo_point = frame->AddRightFeatures(right_features, right_lines, stereo_matches);
      frame_type = _init ? FrameType::KeyFrame : FrameType::InitializationFrame;

      frame->AddJunctions(junctions);
      // SaveLineDetectionResult(image_left_rect, left_lines, _configs.saving_dir, std::to_string(frame->GetFrameId()));
    }else{
      _feature_detector->Detect(image_left_rect, left_features);
      frame->AddLeftFeatures(left_features, left_lines);
      frame_type = FrameType::NormalFrame;
    }

    if(_init){
      const Eigen::Matrix<float, 259, Eigen::Dynamic> features_last_keyframe = _last_keyframe_feature->GetAllFeatures();
      _point_matcher->MatchingPoints(features_last_keyframe, left_features, matches, true);
      int enough_match = AddKeyframeCheck(_last_keyframe_feature, frame, matches);

      if(enough_match == 0){  // try to insert this frame as keyframe
        if(frame_type == FrameType::NormalFrame){
          _feature_detector->Detect(image_right_rect, right_features);
          _point_matcher->MatchingPoints(left_features, right_features, stereo_matches, false);
          good_stereo_point = frame->AddRightFeatures(right_features, right_lines, stereo_matches);
        }

        if(good_stereo_point < 10){
          _insert_next_keyframe = true;
          frame_type = FrameType::NormalFrame;
        }else{
          frame_type = FrameType::KeyFrame;
          _insert_next_keyframe = false;
        }
      }else{
        _insert_next_keyframe = (enough_match == 1) && (frame_type == FrameType::NormalFrame);
      }
    }else{
      if(good_stereo_point < _configs.keyframe_config.min_init_stereo_feature){
        std::cout << "good_stereo_point = " << good_stereo_point << std::endl;
        std::cout << "Not enough stereo points to initialize!" << std::endl;
        continue;
      }else{
        std::cout << "Initialization is done!" << std::endl;
        _init = true;
      }
    }

    TrackingDataPtr tracking_data = std::shared_ptr<TrackingData>(new TrackingData());
    tracking_data->frame = frame;
    tracking_data->frame_type = frame_type;
    tracking_data->ref_keyframe = _last_keyframe_feature;
    tracking_data->matches = matches;
    tracking_data->input_data = input_data;

    if(frame_type != FrameType::NormalFrame){
      _last_keyframe_feature = frame;
    }

    while(_tracking_data_buffer.size() > 5){
      usleep(2000);
    }

    _tracking_mutex.lock();
    _tracking_data_buffer.push(tracking_data);
    _tracking_mutex.unlock();
  }  

  _stop_mutex.lock();
  _feature_thread_stop = true;
  _stop_mutex.unlock();
}

void MapBuilder::TrackingThread(){ 
  while(!_shutdown || !_data_buffer.empty() || !_tracking_data_buffer.empty()){
    if(_tracking_data_buffer.empty()){
      usleep(2000);
      continue;
    }

    TrackingDataPtr tracking_data;
    _tracking_mutex.lock();
    tracking_data = _tracking_data_buffer.front();
    _tracking_data_buffer.pop();
    _tracking_mutex.unlock();

    FramePtr frame = tracking_data->frame;
    FrameType frame_type = tracking_data->frame_type;
    FramePtr ref_keyframe = tracking_data->ref_keyframe;
    InputDataPtr input_data = tracking_data->input_data;
    std::vector<cv::DMatch> matches = tracking_data->matches;

    double timestamp = input_data->time;
    cv::Mat image_left_rect = input_data->image_left.clone();
    cv::Mat image_right_rect = input_data->image_right.clone();
    ImuDataList batch_imu_data = input_data->batch_imu_data;

    if(frame_type == FrameType::InitializationFrame){
      Eigen::Matrix4d init_pose;
      init_pose << 1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1, 0, 0, 0, 1;
      // init_pose = Eigen::Matrix4d::Identity();
      frame->SetPose(init_pose);
      frame->SetPoseFixed(true);
      frame->SetVelocaity(Eigen::Vector3d::Zero());

      _preinteration_keyframe.SetBias(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), false);
      frame->SetIMUPreinteration(_preinteration_keyframe);

      InsertKeyframe(frame);
      _last_keyframe_tracking = frame;
      _last_tracked_frame = frame;
      _last_keyimage = image_left_rect;

      PublishFrame(frame, image_left_rect, frame_type, matches);
      continue;
    }

    // SaveTrackingResult(_last_keyimage, image_left_rect, ref_keyframe, frame, matches, _configs.saving_dir);

    // IMU preinteration
    _preinteration_keyframe.AddBatchData(batch_imu_data, ref_keyframe->GetTimestamp(), timestamp);
    frame->SetIMUPreinteration(_preinteration_keyframe);

    int track_inliers = TrackFrame(ref_keyframe, frame, matches, _preinteration_keyframe);

    frame->SetPreviousFrame(ref_keyframe);

    if(track_inliers > _configs.keyframe_config.lost_num_match){ 
      _last_tracked_frame = frame;
    }

    if(frame_type == FrameType::KeyFrame){
      std::cout << "insert keyframe, id = " << frame->GetFrameId() << std::endl;
      InsertKeyframe(frame);
      _last_keyframe_tracking = frame;
      _last_keyimage = image_left_rect;
    }

    PublishFrame(frame, image_left_rect, frame_type, matches);
  }  

  _stop_mutex.lock();
  _tracking_trhead_stop = true;
  _stop_mutex.unlock();
}

int MapBuilder::TrackFrame(FramePtr ref_frame, FramePtr current_frame, std::vector<cv::DMatch>& matches, Preinteration& _preinteration){
  // line tracking
  Eigen::Matrix<float, 259, Eigen::Dynamic>& ref_features = ref_frame->GetAllFeatures();
  Eigen::Matrix<float, 259, Eigen::Dynamic>& current_features = current_frame->GetAllFeatures();
  std::vector<std::map<int, double>> ref_points_on_lines = ref_frame->GetPointsOnLines();
  std::vector<std::map<int, double>> current_points_on_lines = current_frame->GetPointsOnLines();
  std::vector<int> line_matches;
  MatchLines(ref_points_on_lines, current_points_on_lines, matches, ref_features.cols(), current_features.cols(), line_matches);

  std::vector<int> inliers(current_frame->FeatureNum(), -1);
  std::vector<MappointPtr> matched_mappoints(current_features.cols(), nullptr);
  std::vector<MappointPtr>& ref_frame_mappoints = ref_frame->GetAllMappoints();
  for(auto& match : matches){
    int idx0 = match.queryIdx;
    int idx1 = match.trainIdx;
    matched_mappoints[idx1] = ref_frame_mappoints[idx0];
    inliers[idx1] = ref_frame->GetTrackId(idx0);
  }

  int num_inliers = FramePoseOptimization(ref_frame, current_frame, matched_mappoints, inliers, _preinteration);

  // update track id
  if(num_inliers > _configs.keyframe_config.lost_num_match){
    for(std::vector<cv::DMatch>::iterator it = matches.begin(); it != matches.end();){
      int idx0 = (*it).queryIdx;
      int idx1 = (*it).trainIdx;
      if(inliers[idx1] > 0){
        current_frame->SetTrackId(idx1, ref_frame->GetTrackId(idx0));
        current_frame->InsertMappoint(idx1, ref_frame_mappoints[idx0]);
      }

      if(inliers[idx1] > 0){
        it++;
      }else{
        it = matches.erase(it);
      }
    }

    // update line track id
    const std::vector<MaplinePtr>& ref_frame_maplines = ref_frame->GetConstAllMaplines();
    for(size_t i = 0; i < ref_frame_maplines.size(); i++){
      int j = line_matches[i];
      if(j < 0) continue;
      int line_track_id = ref_frame->GetLineTrackId(i);

      if(line_track_id >= 0){
        current_frame->SetLineTrackId(j, line_track_id);
        current_frame->InsertMapline(j, ref_frame_maplines[i]);
      }
    }
  }

  return num_inliers;
}

int MapBuilder::FramePoseOptimization(FramePtr frame0, FramePtr frame1, std::vector<MappointPtr>& mappoints, 
    std::vector<int>& inliers, Preinteration& preinteration){

  // get initial pose
  bool imu_init = _map->IMUInit();
  Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
  Eigen::Vector3d vwb = Eigen::Vector3d::Zero();
  int frame_id1 = frame1->GetFrameId();
  
  bool predict_by_pnp = true;
  if(imu_init && preinteration.Valid() && preinteration.dT < 2.0){
    Eigen::Matrix4d Twb1;
    Eigen::Matrix4d Twb0 = frame0->IMUPose();
    Eigen::Vector3d vwb0 = frame0->GetVelocity();
    preinteration.Predict(Twb0, vwb0, Twb1, vwb);
    Twc = Twb1 * frame1->GetCamera()->CameraToBody();
    Eigen::Vector3d check_dp = Twc.block<3, 1>(0, 3) - _last_tracked_frame->GetPose().block<3, 1>(0, 3);
    if(check_dp.norm() < 1.0){
      predict_by_pnp = false;
    }
  }

  if(predict_by_pnp){
    // solve PnP using opencv to get initial pose
    std::vector<int> cv_inliers;
    int num_cv_inliers = SolvePnPWithCV(frame1, mappoints, Twc, cv_inliers);
    Eigen::Vector3d check_dp = Twc.block<3, 1>(0, 3) - _last_tracked_frame->GetPose().block<3, 1>(0, 3);
    if(check_dp.norm() > 1.0 || num_cv_inliers < _configs.keyframe_config.lost_num_match ){
      Twc = _last_tracked_frame->GetPose();
    }
  }

  frame1->SetPose(Twc);
  frame1->SetVelocaity(vwb);

  // Second, optimization
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

  // map of poses
  if(imu_init && preinteration.Valid()){
    AddFrameVertex(frame0, poses, 0, velocities, biases, imu_constraints, true, false, true);
    AddFrameVertex(frame1, poses, 0, velocities, biases, imu_constraints, false, false);

    // imu constraint
    IMUConstraintPtr imu_constraint = std::shared_ptr<ImuConstraint>(new ImuConstraint()); 
    imu_constraint->id_pose1 = frame0->GetFrameId();
    imu_constraint->id_pose2 = frame1->GetFrameId();;
    imu_constraint->id_camera1 = 0;
    imu_constraint->id_camera2 = 0;
    imu_constraint->preinteration = std::make_shared<Preinteration>(preinteration);;
    imu_constraints.emplace_back(imu_constraint);
  }else{
    AddFrameVertex(frame1, poses, 0, false);
  }

  // visual constraint construction
  std::vector<size_t> mono_indexes;
  std::vector<size_t> stereo_indexes;
  for(size_t i = 0; i < mappoints.size(); i++){
    // points
    MappointPtr mpt = mappoints[i];
    if(mpt == nullptr || !mpt->IsValid()) continue;
    Eigen::Vector3d keypoint; 
    if(!frame1->GetKeypointPosition(i, keypoint)) continue;

    int mpt_id = mpt->GetId();
    Position3d point;
    point.p = mpt->GetPosition();
    point.fixed = true;
    points.insert(std::pair<int, Position3d>(mpt_id, point));

    // visual constraint
    if(keypoint(2) > 0){
      StereoPointConstraintPtr stereo_constraint = std::shared_ptr<StereoPointConstraint>(new StereoPointConstraint()); 
      stereo_constraint->id_pose = frame_id1;
      stereo_constraint->id_point = mpt_id;
      stereo_constraint->id_camera = 0;
      stereo_constraint->inlier = true;
      stereo_constraint->keypoint = keypoint;
      stereo_constraint->pixel_sigma = 0.8;
      stereo_point_constraints.push_back(stereo_constraint);
      stereo_indexes.push_back(i);
    }else{
      MonoPointConstraintPtr mono_constraint = std::shared_ptr<MonoPointConstraint>(new MonoPointConstraint()); 
      mono_constraint->id_pose = frame_id1;
      mono_constraint->id_point = mpt_id;
      mono_constraint->id_camera = 0;
      mono_constraint->inlier = true;
      mono_constraint->keypoint = keypoint.head(2);
      mono_constraint->pixel_sigma = 0.8;
      mono_point_constraints.push_back(mono_constraint);
      mono_indexes.push_back(i);
    }
  }

  int num_inliers = FrameOptimization(poses, points, lines, velocities, biases, camera_list, 
    mono_point_constraints, stereo_point_constraints, mono_line_constraints, stereo_line_constraints,
    imu_constraints, Rwg, _configs.tracking_optimization_config);

  if(num_inliers > _configs.keyframe_config.lost_num_match ){
    // set frame pose
    Eigen::Matrix4d frame_pose = Eigen::Matrix4d::Identity();
    frame_pose.block<3, 3>(0, 0) = poses[frame_id1].R;
    frame_pose.block<3, 1>(0, 3) = poses[frame_id1].p;
    frame1->SetPose(frame_pose);

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

    if(velocities.size() > 0){
      frame1->SetVelocaity(velocities[frame_id1].velocity);
      frame1->UpdateBias(biases[frame_id1].gyr_bias, biases[frame_id1].acc_bias);
    }
  }

  return num_inliers;
}

// return value: 0 : select this frame as keyframe, 1 : select next frame as keyframe, 2 : not select keyframe
int MapBuilder::AddKeyframeCheck(FramePtr ref_keyframe, FramePtr current_frame, const std::vector<cv::DMatch>& matches){
  int match_num = matches.size();
  if(match_num < _configs.keyframe_config.min_num_match) return 0;

  const Eigen::Matrix<float, 259, Eigen::Dynamic>& ref_features = ref_keyframe->GetAllFeatures();
  const Eigen::Matrix<float, 259, Eigen::Dynamic>& current_features = current_frame->GetAllFeatures();

  float feature_tracking_thr = _configs.keyframe_config.tracking_point_rate;
  double ration_thr = _configs.keyframe_config.tracking_parallax_rate;
  if(UseIMU() && !_map->IMUInit()){
    feature_tracking_thr *= 1.1;
    ration_thr *= 0.7;
  }

  if((float)match_num/ref_features.cols() < feature_tracking_thr || (float)match_num/current_features.cols() < feature_tracking_thr || match_num < _configs.keyframe_config.max_num_match){
    return 1;
  }

  Eigen::Matrix2Xf ref_keypoints(2, match_num), current_keypoints(2, match_num);
  for(int i = 0; i < match_num; i++){
    int idx0 = matches[i].queryIdx;
    int idx1 = matches[i].trainIdx;

    ref_keypoints.col(i) = ref_features.block<2, 1>(1, idx0);
    current_keypoints.col(i) = current_features.block<2, 1>(1, idx1);
  }


  Eigen::Matrix2Xf parallax = ref_keypoints - current_keypoints;
  double average_parallax = (double)((parallax * parallax.transpose()).sum()) / match_num;
  double image_size = _camera->ImageHeight() * _camera->ImageWidth();
  
  if(average_parallax > image_size * ration_thr * ration_thr){
    return 1;
  }

  return 2;
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

  _track_id = _map->UpdateFrameTrackIds(_track_id);
  _line_track_id = _map->UpdateFrameLineTrackIds(_line_track_id);

  Eigen::Vector3d gyr_bias, acc_bias;
  frame->GetBias(gyr_bias, acc_bias);
  _preinteration_keyframe.Reset();
  _preinteration_keyframe.SetBias(gyr_bias, acc_bias, false);
}

void MapBuilder::PublishFrame(FramePtr frame, cv::Mat& image, FrameType frame_type, std::vector<cv::DMatch>& matches){

  double timestamp = frame->GetTimestamp();
  const std::vector<cv::KeyPoint>& keypoints = frame->GetAllKeypoints();
  const std::vector<Eigen::Vector4d>& lines = frame->GatAllLines();
  const std::vector<std::map<int, double>>& points_on_lines = frame->GetPointsOnLines();
  std::vector<bool> inliers_feature_message;
  frame->GetInlierFlag(inliers_feature_message);
  const Eigen::Matrix4d& pose = frame->GetPose();
  const std::vector<int>& line_track_ids = frame->GetAllLineTrackId();

  if(frame_type==InitializationFrame){
    // std::vector<bool> inliers(keypoints.size(), true);
    // key_image_pub = DrawFeatures(image, keypoints, inliers, lines, line_track_ids, points_on_lines);
    // key_image_id_pub = frame->GetFrameId();
    key_image_pub = DrawFeatures(image, keypoints, lines, false);
    key_image_id_pub = 1;
    keyframe_keypoints_pub = frame->GetAllKeypoints();
    return;
  } 

  FeatureMessgaePtr feature_message = std::shared_ptr<FeatureMessgae>(new FeatureMessgae);
  FramePoseMessagePtr frame_pose_message = std::shared_ptr<FramePoseMessage>(new FramePoseMessage);

  feature_message->time = timestamp;
  feature_message->image = image;
  feature_message->key_image = key_image_pub;
  feature_message->frame_id = frame->GetFrameId();
  feature_message->keyfrmae_id = key_image_id_pub;
  feature_message->keypoints = keypoints;
  feature_message->keyframe_keypoints = keyframe_keypoints_pub;
  feature_message->matches = matches;
  feature_message->fm_type = FeatureMessgaeType::VOFeature;
  // feature_message->lines = lines;
  // feature_message->points_on_lines = points_on_lines;
  feature_message->inliers = inliers_feature_message;
  
  frame_pose_message->time = timestamp;
  frame_pose_message->pose = pose;
  // feature_message->line_track_ids = line_track_ids;

  _ros_publisher->PublishFeature(feature_message);
  _ros_publisher->PublishFramePose(frame_pose_message);

  if(frame_type==KeyFrame){
    std::vector<bool> inliers(keypoints.size(), true);
    key_image_pub = DrawFeatures(image, keypoints, lines, false);
    // key_image_id_pub = frame->GetFrameId();
    key_image_id_pub++;
    keyframe_keypoints_pub = frame->GetAllKeypoints();
  }
}

void MapBuilder::SaveTrajectory(){
  std::string file_path = ConcatenateFolderAndFileName(_configs.saving_dir, "keyframe_trajectory.txt");
  _map->SaveKeyframeTrajectory(file_path);
}

void MapBuilder::SaveTrajectory(std::string file_path){
  _map->SaveKeyframeTrajectory(file_path);
}

void MapBuilder::SaveMap(const std::string& map_root){
  // _map->SaveMap(map_root);
  std::string map_path = ConcatenateFolderAndFileName(map_root, "AirSLAM_mapv0.bin");
  std::ofstream ofs(map_path, std::ios::binary);
  std::cout << "map_path = " << map_path << std::endl;
  boost::archive::binary_oarchive oa(ofs);

  _map->CheckMap();

  std::cout << "Map saveing..... " << std::endl;
  oa << _map;
  std::cout << "Map saveing done! " << std::endl;

}

void MapBuilder::Stop(){
  _stop_mutex.lock();
  _shutdown = true;
  _stop_mutex.unlock();
  _ros_publisher->ShutDown();
  _feature_thread.join();
  _tracking_thread.join();
}

bool MapBuilder::IsStopped(){
  bool have_stopped = (_feature_thread_stop && _tracking_trhead_stop);
  return have_stopped;
}