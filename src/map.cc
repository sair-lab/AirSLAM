#include <cmath> 
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "map.h"
#include "utils.h"
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
      if(track_ids[i] < 0) continue;
      mpt = std::shared_ptr<Mappoint>(new Mappoint(track_ids[i]));
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

  // optimization
  if(_keyframes.size() > 2){
    SlidingWindowOptimization();
  }
}

void Map::InsertMappoint(MappointPtr mappoint){
  int mappoint_id = mappoint->GetId();
  _mappoints[mappoint_id] = mappoint;
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

void Map::SlidingWindowOptimization(){
  const size_t WindowSize = 10;
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
      Position3d point;
      point.p = mpt->GetPosition();
      point.fixed = false;
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

// void Map::LoadMap(const std::string& map_root){

//   // load camera file
//   std::string camera_file = map_root + "/camera.yaml";
//   camera = std::shared_ptr<Camera>(new Camera(camera_file));

//   // load frame files
//   std::string frame_root = map_root + "/frames";
//   std::vector<std::string> frame_files;
//   GetFiles(frame_root, frame_files);
//   for(std::string& frame_file : frame_files){

//     std::vector<std::string> keyframe_files;
//     std::string keyframe_root = frame_root + "/" + frame_file;

//     // load metadata
//     std::string metadata_path = keyframe_root + "/metadata.txt";
//     std::vector<std::vector<std::string> > metadata;
//     ReadTxt(metadata_path, metadata, ",");
//     int frame_id = atoi(metadata[0][0].c_str());
//     int64_t timestamp = (int64_t)(atoll(metadata[0][1].c_str()));
//     int num_kps = atoi(metadata[0][2].c_str());
//     int num_lines = atoi(metadata[0][3].c_str());
//     Eigen::Matrix<double, 7, 1> odom_pose;
//     for(int i = 0; i < 7; ++i){
//       odom_pose(i, 0) = atof(metadata[0][(i+4)].c_str());
//     }
//     FramePtr keyframe = std::shared_ptr<Keyframe>(new Keyframe(frame_id, timestamp, odom_pose));

//     // load points
//     std::string points_root = keyframe_root + "/points";
//     std::string keypoints_file = points_root + "/keypoints.npy";
//     cnpy::NpyArray keypoints_arr = cnpy::npy_load(keypoints_file);
//     float* keypoints_data = keypoints_arr.data<float>();
//     std::string descriptors_file = points_root + "/descriptors.npy";
//     cnpy::NpyArray descriptors_arr = cnpy::npy_load(descriptors_file);
//     float* descriptors_data = descriptors_arr.data<float>();
//     std::string scores_file = points_root + "/scores.npy";
//     cnpy::NpyArray scores_arr = cnpy::npy_load(scores_file);
//     float* scores_data = scores_arr.data<float>();
//     std::string track_ids_file = points_root + "/track_ids.npy";
//     cnpy::NpyArray track_ids_arr = cnpy::npy_load(track_ids_file);
//     int* track_ids_data = track_ids_arr.data<int>();

//     const int kNumDesc = 256;
//     for(int i = 0; i < num_kps; i++){
//       float x = keypoints_data[(2*i)];
//       float y = keypoints_data[(2*i+1)];
//       float score = scores_data[i];
//       cv::KeyPoint keypoint(x, y, 1.0, -1, score);
//       int track_id = *(track_ids_data + i);

//       Eigen::VectorXf descriptor = 
//           Eigen::Map<Eigen::VectorXf>((descriptors_data+kNumDesc*i), kNumDesc, 1);
//       keyframe->AddPoint(keypoint, descriptor, track_id);
//     }

//     // load line data
//     std::string line_file_path = keyframe_root + "/lines.txt";
//     std::vector<std::vector<std::string> > lines;
//     ReadTxt(line_file_path, lines, ",");
//     for(std::vector<std::string> line_data : lines){
//       Line2D line2d;
//       line2d.x1 = atof(line_data[0].c_str());
//       line2d.y1 = atof(line_data[1].c_str());
//       line2d.x2 = atof(line_data[2].c_str());
//       line2d.y2 = atof(line_data[3].c_str());
//       for(int i = 4; i < line_data.size(); ++i){
//         int point_id = static_cast<int>(atof(line_data[i].c_str()));
//         line2d.point_ids.emplace_back(point_id);
//       }
//       keyframe->AddLine(line2d);
//     }
//     keyframes[frame_id] = keyframe;
//   }

//   optimizer = std::shared_ptr<Optimization3d>(new Optimization3d());
// }


// ros::Time ConvertToRosTime(int64_t& t){
//   const uint32_t kNanosecondsToSecond = 1e9;
//   const uint64_t timestamp_u64 = static_cast<uint64_t>(t);
//   const uint32_t ros_timestamp_sec = timestamp_u64 / kNanosecondsToSecond;
//   const uint32_t ros_timestamp_nsec =
//       timestamp_u64 - (ros_timestamp_sec * kNanosecondsToSecond);
//   return ros::Time(ros_timestamp_sec, ros_timestamp_nsec);
// }

// void AddNewPoseToPath(
//     Eigen::Vector3d& pose, nav_msgs::Path& path, std::string& frame_id){
//   ros::Time current_time = ros::Time::now();

//   geometry_msgs::PoseStamped pose_stamped; 
//   pose_stamped.pose.position.x = pose(0); 
//   pose_stamped.pose.position.y = pose(1); 

//   geometry_msgs::Quaternion q = tf::createQuaternionMsgFromYaw(pose(2)); 
//   pose_stamped.pose.orientation.x = q.x; 
//   pose_stamped.pose.orientation.y = q.y; 
//   pose_stamped.pose.orientation.z = q.z; 
//   pose_stamped.pose.orientation.w = q.w; 

//   pose_stamped.header.stamp = current_time; 
//   pose_stamped.header.frame_id = frame_id; 
//   path.poses.push_back(pose_stamped); 
// }

// void Map::VisualizeMap(){
//   ros::Time current_time = ros::Time::now();
//   std::string frame_id = "map";

//   frame_poses_pub = nh.advertise<visualization_msgs::MarkerArray>("/map/frame_pose", 10);
//   visualization_msgs::MarkerArray frame_pose_msgs;

//   path_pub = nh.advertise<nav_msgs::Path>("/map/frame_path", 10);
//   nav_msgs::Path path_msgs;
//   path_msgs.header.stamp = current_time; 
// 	path_msgs.header.frame_id = frame_id; 
//   for(auto& kv : keyframes){
//     FramePtr kf = kv.second;
//     int64_t time_int = kf->GetTimestamp();
//     ros::Time timestamp = ConvertToRosTime(time_int);
//     Eigen::Matrix<double, 7, 1> pose = kf->GetOdomPose();

//     // path
//     geometry_msgs::PoseStamped pose_stamped; 
//     pose_stamped.pose.orientation.w = pose(0); 
//     pose_stamped.pose.orientation.x = pose(1); 
//     pose_stamped.pose.orientation.y = pose(2); 
//     pose_stamped.pose.orientation.z = pose(3); 
//     pose_stamped.pose.position.x = pose(4); 
//     pose_stamped.pose.position.y = pose(5); 
//     pose_stamped.pose.position.z = pose(6); 

//     pose_stamped.header.stamp = timestamp; 
//     pose_stamped.header.frame_id = frame_id; 
//     path_msgs.poses.push_back(pose_stamped); 

//     // marker
//     visualization_msgs::Marker marker;
//     marker.header.frame_id = frame_id;
//     marker.header.stamp = timestamp;
//     marker.ns = "frame_pose";
//     marker.action = visualization_msgs::Marker::ADD;
//     marker.id = kf->GetFrameId();
//     marker.type = visualization_msgs::Marker::ARROW;
//     marker.scale.x = 0.2;
//     marker.scale.y = 0.03;
//     marker.scale.z = 0.03;
//     marker.color.b = 0;
//     marker.color.g = 0;
//     marker.color.r = 255;
//     marker.color.a = 1;
//     marker.pose = pose_stamped.pose;

//     frame_pose_msgs.markers.push_back(marker);
//   }

//   // map pointcloud
//   mappoints_pub = nh.advertise<sensor_msgs::PointCloud> ("/map/mappoints", 1);
//   sensor_msgs::PointCloud mappoints_msgs;
//   mappoints_msgs.header.stamp = current_time; 
// 	mappoints_msgs.header.frame_id = frame_id; 

//   int num_points = 0;
//   for(auto& kv : mappoints){
//     if(kv.second->IsValid()) num_points++;
//   }

//   mappoints_msgs.points.resize(num_points);
//   mappoints_msgs.channels.resize(3);
//   mappoints_msgs.channels[0].name = "r";
//   mappoints_msgs.channels[0].values.resize(num_points);
//   mappoints_msgs.channels[1].name = "g";
//   mappoints_msgs.channels[1].values.resize(num_points);
//   mappoints_msgs.channels[2].name = "b";
//   mappoints_msgs.channels[2].values.resize(num_points);


//   int i = 0;
//   for(auto& kv : mappoints){
//     if(!kv.second->IsValid()) continue;
//     Eigen::Vector3d position = kv.second->GetPosition();
//     mappoints_msgs.points[i].x = position(0);
//     mappoints_msgs.points[i].y = position(1);
//     mappoints_msgs.points[i].z = position(2);
//     mappoints_msgs.channels[0].values[i] = static_cast<double>(static_cast<int>((position(0)+0.5))%10)/10.0;
//     mappoints_msgs.channels[1].values[i] = static_cast<double>(static_cast<int>((position(1)+0.5))%10)/10.0;
//     mappoints_msgs.channels[2].values[i] = static_cast<double>(static_cast<int>((position(2)+0.5))%10)/10.0;
    
//     i++;
//   }

//   ros::Rate loop_rate(1);
//   while(ros::ok()){
//     frame_poses_pub.publish(frame_pose_msgs);
//     path_pub.publish(path_msgs);
//     mappoints_pub.publish(mappoints_msgs);
//     ros::spinOnce(); 
//     loop_rate.sleep(); 
//   }

//   // frame_poses_pub.publish(frame_pose_msgs);
//   // path_pub.publish(path_msgs);
//   // mappoints_pub.publish(mappoints_msgs); 
//   // ros::Duration(5).sleep();
// }