#include "ros_publisher.h"

#include <Eigen/Geometry>

#include "utils.h"

RosPublisher::RosPublisher(const RosPublisherConfig& ros_publisher_config, ros::NodeHandle nh): _config(ros_publisher_config){

  if(_config.feature){
    _ros_feature_pub = nh.advertise<sensor_msgs::Image>(_config.feature_topic, 10);
    std::function<void(const FeatureMessgaeConstPtr&)> publish_feature_function = 
        [&](const FeatureMessgaeConstPtr& feature_message){
      cv::Mat result;
      if(feature_message->fm_type == FeatureMessgaeType::RelocFeature){
        result = DrawFeatures(feature_message->image, feature_message->keypoints, feature_message->lines, true);
      }else{
        cv::Mat drawed_image = DrawFeatures(feature_message->image, feature_message->keypoints, 
            feature_message->lines, true);

        result = DrawMatches(feature_message->key_image, drawed_image, 
            feature_message->keyframe_keypoints, feature_message->keypoints, feature_message->matches);

        cv::copyMakeBorder(result, result, 50, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 255));
        cv::putText(result, "Feature Detection", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        std::string keyframe_info = "Keyframe Num : " + std::to_string(feature_message->keyfrmae_id);
        cv::putText(result, keyframe_info, cv::Point(10+drawed_image.cols, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        std::string frame_info = "Current Frame Id: " + std::to_string(feature_message->frame_id);
        cv::putText(result, frame_info, cv::Point(10+drawed_image.cols*2, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
      }

      sensor_msgs::ImagePtr ros_feature_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", result).toImageMsg();
      ros_feature_msg->header.stamp = ros::Time().fromSec(feature_message->time);
      _ros_feature_pub.publish(ros_feature_msg);
    };
    _feature_publisher.Register(publish_feature_function);
    _feature_publisher.Start();
  }

  if(_config.frame_pose){
    _ros_frame_pose_pub = nh.advertise<geometry_msgs::PoseStamped>(_config.frame_pose_topic, 10);
    _pub_latest_odometry = nh.advertise<nav_msgs::Odometry>(_config.frame_odometry_topic, 1000);

    std::function<void(const FramePoseMessageConstPtr&)> publish_frame_pose_function = 
        [&](const FramePoseMessageConstPtr& frame_pose_message){
      geometry_msgs::PoseStamped pose_stamped;
      pose_stamped.header.stamp = ros::Time().fromSec(frame_pose_message->time);
      pose_stamped.header.frame_id = "map";
      pose_stamped.pose.position.x = frame_pose_message->pose(0, 3);
      pose_stamped.pose.position.y = frame_pose_message->pose(1, 3);
      pose_stamped.pose.position.z = frame_pose_message->pose(2, 3);
      Eigen::Quaterniond q(frame_pose_message->pose.block<3, 3>(0, 0));
      pose_stamped.pose.orientation.x = q.x();
      pose_stamped.pose.orientation.y = q.y();
      pose_stamped.pose.orientation.z = q.z();
      pose_stamped.pose.orientation.w = q.w();
      _ros_frame_pose_pub.publish(pose_stamped);

      nav_msgs::Odometry odometry;
      odometry.header.stamp = ros::Time().fromSec(frame_pose_message->time);
      odometry.header.frame_id = "map";
      odometry.pose.pose.position.x = frame_pose_message->pose(0, 3);
      odometry.pose.pose.position.y = frame_pose_message->pose(1, 3);
      odometry.pose.pose.position.z = frame_pose_message->pose(2, 3);
      odometry.pose.pose.orientation.x = q.x();
      odometry.pose.pose.orientation.y = q.y();
      odometry.pose.pose.orientation.z = q.z();
      odometry.pose.pose.orientation.w = q.w();
      _pub_latest_odometry.publish(odometry);

      static tf::TransformBroadcaster br;
      tf::Transform transform;
      tf::Quaternion tf_q;
      transform.setOrigin(tf::Vector3(frame_pose_message->pose(0, 3),
                                      frame_pose_message->pose(1, 3),
                                      frame_pose_message->pose(2, 3)));
      tf_q.setW(q.w());
      tf_q.setX(q.x());
      tf_q.setY(q.y());
      tf_q.setZ(q.z());
      transform.setRotation(tf_q);
      br.sendTransform(tf::StampedTransform(transform, pose_stamped.header.stamp, "map", "camera"));
    };
    _frame_pose_publisher.Register(publish_frame_pose_function);
    _frame_pose_publisher.Start();
  }

  if(_config.keyframe){
    _ros_keyframe_pub = nh.advertise<geometry_msgs::PoseArray>(_config.keyframe_topic, 10);
    // _ros_keyframe_array.header.stamp = ros::Time::now();
    _ros_keyframe_array.header.frame_id = "map";

    _ros_path_pub = nh.advertise<nav_msgs::Path>(_config.path_topic, 10);
    // _ros_path.header.stamp = ros::Time::now();
    _ros_path.header.frame_id = "map";

    std::function<void(const KeyframeMessageConstPtr&)> publish_keyframe_function = 
        [&](const KeyframeMessageConstPtr& keyframe_message){
      _ros_keyframe_array.header.stamp = ros::Time().fromSec(keyframe_message->time); 
      _ros_path.header.stamp = ros::Time().fromSec(keyframe_message->time); 

      std::map<int, int>::iterator it;
      for(int i = 0; i < keyframe_message->ids.size(); i++){
        int keyframe_id = keyframe_message->ids[i];

        geometry_msgs::Pose pose;
        pose.position.x = keyframe_message->poses[i](0, 3);
        pose.position.y = keyframe_message->poses[i](1, 3);
        pose.position.z = keyframe_message->poses[i](2, 3);
        Eigen::Quaterniond q(keyframe_message->poses[i].block<3, 3>(0, 0));
        pose.orientation.x = q.x();
        pose.orientation.y = q.y();
        pose.orientation.z = q.z();
        pose.orientation.w = q.w();

        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(keyframe_message->times[i]);
        pose_stamped.pose = pose;
        
        it = _keyframe_id_to_index.find(keyframe_id);
        if(it == _keyframe_id_to_index.end()){
          _ros_keyframe_array.poses.push_back(pose);
          _ros_path.poses.push_back(pose_stamped);
          _keyframe_id_to_index[keyframe_id] = _ros_keyframe_array.poses.size()-1;
        }else{
          int idx = it->second;
          _ros_keyframe_array.poses[idx] = pose;
          _ros_path.poses[idx] = pose_stamped;
        }
      }
      _ros_keyframe_pub.publish(_ros_keyframe_array);
      _ros_path_pub.publish(_ros_path);
    };
    _keyframe_publisher.Register(publish_keyframe_function);    
    _keyframe_publisher.Start();
  }

  if(_config.map){
    // for mappoints
    _ros_map_pub = nh.advertise<sensor_msgs::PointCloud> (_config.map_topic, 1);
    _ros_mappoints.header.stamp = ros::Time::now(); 
    _ros_mappoints.header.frame_id = "map"; 

    std::function<void(const MapMessageConstPtr&)> publish_map_function = 
        [&](const MapMessageConstPtr& map_message){
      _ros_mappoints.header.stamp = ros::Time().fromSec(map_message->time);

      std::unordered_map<int, int>::iterator it;
      for(int i = 0; i < map_message->ids.size(); i++){
        int mappoint_id = map_message->ids[i];
        it = _mappoint_id_to_index.find(mappoint_id);
        if(it == _mappoint_id_to_index.end()){
          geometry_msgs::Point32 point;
          point.x = map_message->points[i](0);
          point.y = map_message->points[i](1);
          point.z = map_message->points[i](2);
          _ros_mappoints.points.push_back(point);
          _mappoint_id_to_index[mappoint_id] = _ros_mappoints.points.size()-1;
        }else{
          int idx = it->second;
          _ros_mappoints.points[idx].x = map_message->points[i](0);
          _ros_mappoints.points[idx].y = map_message->points[i](1);
          _ros_mappoints.points[idx].z = map_message->points[i](2);
        }
      }
      _ros_map_pub.publish(_ros_mappoints);
    };
    _map_publisher.Register(publish_map_function); 
    _map_publisher.Start();   

    // for maplines
    _ros_mapline_pub = nh.advertise<visualization_msgs::Marker>(_config.mapline_topic, 1);
    _ros_maplines.header.stamp = ros::Time::now(); 
    _ros_maplines.header.frame_id = "map"; 
    _ros_maplines.pose.position.x = 0;
    _ros_maplines.pose.position.y = 0;
    _ros_maplines.pose.position.z = 0;
    _ros_maplines.pose.orientation.x = 0;
    _ros_maplines.pose.orientation.y = 0;
    _ros_maplines.pose.orientation.z = 0;
    _ros_maplines.pose.orientation.w = 1.0;
    _ros_maplines.type = visualization_msgs::Marker::LINE_LIST;
    _ros_maplines.scale.x = 0.05;  
    _ros_maplines.color.b = 1.0;
    _ros_maplines.color.a = 1.0;

    std::function<void(const MapLineMessageConstPtr&)> publish_mapline_function = 
        [&](const MapLineMessageConstPtr& mapline_message){
      _ros_maplines.header.stamp = ros::Time().fromSec(mapline_message->time);

      std::unordered_map<int, int>::iterator it;
      for(int i = 0; i < mapline_message->ids.size(); i++){
        int mapline_id = mapline_message->ids[i];
        it = _mapline_id_to_index.find(mapline_id);
        if(it == _mapline_id_to_index.end()){
          geometry_msgs::Point point1, point2;
          point1.x = mapline_message->lines[i](0);
          point1.y = mapline_message->lines[i](1);
          point1.z = mapline_message->lines[i](2);
          point2.x = mapline_message->lines[i](3);
          point2.y = mapline_message->lines[i](4);
          point2.z = mapline_message->lines[i](5);
          _ros_maplines.points.push_back(point1);
          _ros_maplines.points.push_back(point2);

          std_msgs::ColorRGBA color;
          Eigen::Vector3d color_vector;
          GenerateColor(mapline_id, color_vector);
          color.r = 0.0;
          color.g = 0.0;
          color.b = 1.0;   
          color.a = 1.0;   
          _ros_maplines.colors.push_back(color);
          _ros_maplines.colors.push_back(color);
          _mapline_id_to_index[mapline_id] = _ros_maplines.points.size()-2;
        }else{
          int idx = it->second;
          _ros_maplines.points[idx].x = mapline_message->lines[i](0);
          _ros_maplines.points[idx].y = mapline_message->lines[i](1);
          _ros_maplines.points[idx].z = mapline_message->lines[i](2);
          _ros_maplines.points[idx+1].x = mapline_message->lines[i](3);
          _ros_maplines.points[idx+1].y = mapline_message->lines[i](4);
          _ros_maplines.points[idx+1].z = mapline_message->lines[i](5);
        }
      }
      _ros_mapline_pub.publish(_ros_maplines);
    };
    _mapline_publisher.Register(publish_mapline_function); 
    _mapline_publisher.Start();  
  }

  if(_config.reloc){
    // for relocalization trajectory
    _ros_reloc_traj_pub = nh.advertise<visualization_msgs::Marker>(_config.reloc_topic+"/trajectory", 1);
    _ros_reloc_traj.header.stamp = ros::Time::now(); 
    _ros_reloc_traj.header.frame_id = "map"; 
    _ros_reloc_traj.pose.position.x = 0;
    _ros_reloc_traj.pose.position.y = 0;
    _ros_reloc_traj.pose.position.z = 0;
    _ros_reloc_traj.pose.orientation.x = 0;
    _ros_reloc_traj.pose.orientation.y = 0;
    _ros_reloc_traj.pose.orientation.z = 0;
    _ros_reloc_traj.pose.orientation.w = 1.0;
    _ros_reloc_traj.type = visualization_msgs::Marker::SPHERE_LIST;
    // _ros_reloc_traj.scale.x = 0.8;  
    // _ros_reloc_traj.scale.y = 0.8;
    // _ros_reloc_traj.scale.z = 0.8;

    _ros_reloc_traj.color.a = 0.5; 
    _ros_reloc_traj.color.r = 0.0;
    _ros_reloc_traj.color.g = 1.0;
    _ros_reloc_traj.color.b = 0.0;  

    std::function<void(const RelocMessageConstPtr&)> publish_reloc_traj_function = 
        [&](const RelocMessageConstPtr& reloc_message){
      _ros_reloc_traj.scale.x = 0.8 * reloc_message->map_scale;  
      _ros_reloc_traj.scale.y = 0.8 * reloc_message->map_scale;
      _ros_reloc_traj.scale.z = 0.8 * reloc_message->map_scale;

      int idx = reloc_message->poses.size() - 1;
      if(idx < 0){
        return;
      }
      geometry_msgs::Point p;
      p.x = reloc_message->poses[idx](0, 3);
      p.y = reloc_message->poses[idx](1, 3);
      p.z = reloc_message->poses[idx](2, 3);
      _ros_reloc_traj.points.push_back(p);

      _ros_reloc_traj_pub.publish(_ros_reloc_traj);
    };
    _reloc_traj_publisher.Register(publish_reloc_traj_function); 
    _reloc_traj_publisher.Start();  


    // for current pose
    _ros_reloc_pose_pub = nh.advertise<geometry_msgs::PoseStamped>(_config.reloc_topic+"/pose", 10);
    std::function<void(const RelocMessageConstPtr&)> publish_reloc_pose_function = 
        [&](const RelocMessageConstPtr& reloc_message){
      int idx = reloc_message->times.size() - 1;
      geometry_msgs::PoseStamped pose_stamped;
      pose_stamped.header.stamp = ros::Time().fromSec(reloc_message->times[idx]);
      pose_stamped.header.frame_id = "map";
      pose_stamped.pose.position.x = reloc_message->poses[idx](0, 3);
      pose_stamped.pose.position.y = reloc_message->poses[idx](1, 3);
      pose_stamped.pose.position.z = reloc_message->poses[idx](2, 3);
      Eigen::Quaterniond q(reloc_message->poses[idx].block<3, 3>(0, 0));
      pose_stamped.pose.orientation.x = q.x();
      pose_stamped.pose.orientation.y = q.y();
      pose_stamped.pose.orientation.z = q.z();
      pose_stamped.pose.orientation.w = q.w();
      _ros_reloc_pose_pub.publish(pose_stamped);
    };
    _reloc_pose_publisher.Register(publish_reloc_pose_function);
    _reloc_pose_publisher.Start();


    // for matches
    _ros_reloc_mpts_pub = nh.advertise<visualization_msgs::Marker>(_config.reloc_topic+"/matches", 1);
    std::function<void(const RelocMessageConstPtr&)> publish_reloc_mpts_function = 
        [&](const RelocMessageConstPtr& reloc_message){
      int idx = reloc_message->times.size() - 1;
      visualization_msgs::Marker ros_reloc_mpts;
      ros_reloc_mpts.header.stamp = ros::Time().fromSec(reloc_message->times[idx]);
      ros_reloc_mpts.header.frame_id = "map"; 
      ros_reloc_mpts.pose.position.x = 0;
      ros_reloc_mpts.pose.position.y = 0;
      ros_reloc_mpts.pose.position.z = 0;
      ros_reloc_mpts.pose.orientation.x = 0;
      ros_reloc_mpts.pose.orientation.y = 0;
      ros_reloc_mpts.pose.orientation.z = 0;
      ros_reloc_mpts.pose.orientation.w = 1.0;
      ros_reloc_mpts.type = visualization_msgs::Marker::LINE_LIST;
      ros_reloc_mpts.scale.x = 0.1 * reloc_message->map_scale;  
      ros_reloc_mpts.color.b = 1.0;
      ros_reloc_mpts.color.a = 0.5;

      geometry_msgs::Point point1;
      point1.x = reloc_message->poses[idx](0, 3);
      point1.y = reloc_message->poses[idx](1, 3);
      point1.z = reloc_message->poses[idx](2, 3);

      for(int i = 0; i < reloc_message->mappoints.size(); i++){

        geometry_msgs::Point point2;
        point2.x = reloc_message->mappoints[i](0);
        point2.y = reloc_message->mappoints[i](1);
        point2.z = reloc_message->mappoints[i](2);
        ros_reloc_mpts.points.push_back(point1);
        ros_reloc_mpts.points.push_back(point2);

        std_msgs::ColorRGBA color;
        color.r = 0.0;
        color.g = 1.0;
        color.b = 0.0;   
        color.a = 1.0;   
        ros_reloc_mpts.colors.push_back(color);
        ros_reloc_mpts.colors.push_back(color);
      }
      _ros_reloc_mpts_pub.publish(ros_reloc_mpts);
    };
    _reloc_mpts_publisher.Register(publish_reloc_mpts_function); 
    _reloc_mpts_publisher.Start();  
  }

}

void RosPublisher::PublishFeature(FeatureMessgaePtr feature_message){
  _feature_publisher.Publish(feature_message);
}

void RosPublisher::PublishFramePose(FramePoseMessagePtr frame_pose_message){
  _frame_pose_publisher.Publish(frame_pose_message);
}

void RosPublisher::PublisheKeyframe(KeyframeMessagePtr keyframe_message){
  _keyframe_publisher.Publish(keyframe_message);
}

void RosPublisher::PublishMap(MapMessagePtr map_message){
  _map_publisher.Publish(map_message);
}

void RosPublisher::PublishMapLine(MapLineMessagePtr mapline_message){
  _mapline_publisher.Publish(mapline_message);
} 

void RosPublisher::PubRelocResults(RelocMessagePtr reloc_message){
  _reloc_traj_publisher.Publish(reloc_message);
  _reloc_pose_publisher.Publish(reloc_message);
  _reloc_mpts_publisher.Publish(reloc_message);
}

void RosPublisher::Clear(){
  _keyframe_id_to_index.clear();
  _ros_keyframe_array.poses.clear();
  _ros_path.poses.clear();

  _mappoint_id_to_index.clear();
  _ros_mappoints.points.clear();

  _mapline_id_to_index.clear();
  _ros_maplines.colors.clear();
  _ros_maplines.points.clear();
}

void RosPublisher::ShutDown(){
  if(_config.feature){
    _feature_publisher.ShutDown();
  }

  if(_config.frame_pose){
    _frame_pose_publisher.ShutDown();
  }

  if(_config.keyframe){
    _keyframe_publisher.ShutDown();
  }

  if(_config.map){
    _map_publisher.ShutDown();
    _mapline_publisher.ShutDown();
  }

  if(_config.reloc){
    _reloc_traj_publisher.ShutDown();
    _reloc_pose_publisher.ShutDown();
    _reloc_mpts_publisher.ShutDown();
  }
}