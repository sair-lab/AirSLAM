#ifndef READ_CONFIGS_H_
#define READ_CONFIGS_H_

#include <iostream>
#include <yaml-cpp/yaml.h>

#include "utils.h"

struct PLNetConfig{
  std::string superpoint_onnx;
  std::string superpoint_engine;

  std::string plnet_s0_onnx;
  std::string plnet_s0_engine;
  std::string plnet_s1_onnx;
  std::string plnet_s1_engine;

  int use_superpoint;

  int max_keypoints;
  float keypoint_threshold;
  int remove_borders;

  float line_threshold;
  float line_length_threshold;

  PLNetConfig() {}
  void Load(const YAML::Node& plnet_node){
    use_superpoint = plnet_node["use_superpoint"].as<int>();

    max_keypoints = plnet_node["max_keypoints"].as<int>();
    keypoint_threshold = plnet_node["keypoint_threshold"].as<float>();
    remove_borders = plnet_node["remove_borders"].as<int>();

    line_threshold = plnet_node["line_threshold"].as<float>();
    line_length_threshold = plnet_node["line_length_threshold"].as<float>();
  }

  void SetModelPath(std::string model_dir){
    if(use_superpoint){
      superpoint_onnx = ConcatenateFolderAndFileName(model_dir, "superpoint_v1_sim_int32.onnx");
      superpoint_engine = ConcatenateFolderAndFileName(model_dir, "superpoint_v1_sim_int32.engine");
    }

    plnet_s0_onnx = ConcatenateFolderAndFileName(model_dir, "plnet_s0.onnx");
    plnet_s0_engine = ConcatenateFolderAndFileName(model_dir, "plnet_s0.engine");
    plnet_s1_onnx = ConcatenateFolderAndFileName(model_dir, "plnet_s1.onnx");
    plnet_s1_engine = ConcatenateFolderAndFileName(model_dir, "plnet_s1.engine");
  }

};


struct SuperPointConfig {
  SuperPointConfig() {}
  void Load(const YAML::Node& superpoint_node){
    max_keypoints = superpoint_node["max_keypoints"].as<int>();
    keypoint_threshold = superpoint_node["keypoint_threshold"].as<float>();
    remove_borders = superpoint_node["remove_borders"].as<int>();
    dla_core = superpoint_node["dla_core"].as<int>();
    const YAML::Node superpoint_input_tensor_names_node = superpoint_node["input_tensor_names"];
    size_t superpoint_num_input_tensor_names = superpoint_input_tensor_names_node.size();
    for(size_t i = 0; i < superpoint_num_input_tensor_names; i++){
      input_tensor_names.push_back(superpoint_input_tensor_names_node[i].as<std::string>());
    }
    YAML::Node superpoint_output_tensor_names_node = superpoint_node["output_tensor_names"];
    size_t superpoint_num_output_tensor_names = superpoint_output_tensor_names_node.size();
    for(size_t i = 0; i < superpoint_num_output_tensor_names; i++){
      output_tensor_names.push_back(superpoint_output_tensor_names_node[i].as<std::string>());
    }
    onnx_file = superpoint_node["onnx_file"].as<std::string>();
    engine_file= superpoint_node["engine_file"].as<std::string>();
  }

  int max_keypoints;
  float keypoint_threshold;
  int remove_borders;
  int dla_core;
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_file;
  std::string engine_file;
};

struct PointMatcherConfig {
  PointMatcherConfig() {}
  void Load(const YAML::Node& point_matcher_node){
    matcher = point_matcher_node["matcher"].as<int>();
    image_width = point_matcher_node["image_width"].as<int>();
    image_height = point_matcher_node["image_height"].as<int>();
    onnx_file = point_matcher_node["onnx_file"].as<std::string>();
    engine_file = point_matcher_node["engine_file"].as<std::string>();
  }

  int matcher;
  int image_width;
  int image_height;
  int dla_core;
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_file;
  std::string engine_file;
};

struct LineDetectorConfig{
  LineDetectorConfig() {}
  void Load(const YAML::Node& line_detector_node){
    length_threshold = line_detector_node["length_threshold"].as<int>();
    distance_threshold = line_detector_node["distance_threshold"].as<float>();
    canny_th1 = line_detector_node["canny_th1"].as<double>();
    canny_th2 = line_detector_node["canny_th2"].as<double>();
    canny_aperture_size = line_detector_node["canny_aperture_size"].as<int>();
    do_merge = line_detector_node["do_merge"].as<int>();
    angle_thr = line_detector_node["angle_thr"].as<float>();
    distance_thr = line_detector_node["distance_thr"].as<float>();
    ep_thr = line_detector_node["ep_thr"].as<float>();
  }

  int length_threshold;
  float distance_threshold;
  double canny_th1;
  double canny_th2;
  int canny_aperture_size;
  int do_merge;
  float angle_thr;
  float distance_thr;
  float ep_thr;
};

struct KeyframeConfig {
  KeyframeConfig() {}
  void Load(const YAML::Node& keyframe_node){
    min_init_stereo_feature = keyframe_node["min_init_stereo_feature"].as<int>();
    lost_num_match = keyframe_node["lost_num_match"].as<int>();
    min_num_match = keyframe_node["min_num_match"].as<int>();
    max_num_match = keyframe_node["max_num_match"].as<int>();
    tracking_point_rate = keyframe_node["tracking_point_rate"].as<float>();
    tracking_parallax_rate = keyframe_node["tracking_parallax_rate"].as<double>();
  }

  int min_init_stereo_feature;
  int lost_num_match;
  int min_num_match;
  int max_num_match;
  float tracking_point_rate;
  double tracking_parallax_rate;
};

struct OptimizationConfig{
  OptimizationConfig() {}
  void Load(const YAML::Node& optimization_node){
    mono_point = optimization_node["mono_point"].as<double>();
    stereo_point = optimization_node["stereo_point"].as<double>();
    mono_line = optimization_node["mono_line"].as<double>();
    stereo_line = optimization_node["stereo_line"].as<double>();
    rate = optimization_node["rate"].as<double>();    
  }

  double mono_point;
  double stereo_point;
  double mono_line;
  double stereo_line;
  double rate;
};

struct RosPublisherConfig{
  RosPublisherConfig() {}
  void Load(const YAML::Node& ros_publisher_node){
    feature = ros_publisher_node["feature"].as<int>();
    feature_topic = ros_publisher_node["feature_topic"].as<std::string>();   
    frame_pose = ros_publisher_node["frame_pose"].as<int>();
    frame_pose_topic = ros_publisher_node["frame_pose_topic"].as<std::string>();
    frame_odometry_topic = ros_publisher_node["frame_odometry_topic"].as<std::string>();
    keyframe = ros_publisher_node["keyframe"].as<int>();
    keyframe_topic = ros_publisher_node["keyframe_topic"].as<std::string>();
    path_topic = ros_publisher_node["path_topic"].as<std::string>();
    map = ros_publisher_node["map"].as<int>();
    map_topic = ros_publisher_node["map_topic"].as<std::string>();
    mapline = ros_publisher_node["mapline"].as<int>();
    mapline_topic = ros_publisher_node["mapline_topic"].as<std::string>();
    reloc = ros_publisher_node["reloc"].as<int>();
    reloc_topic = ros_publisher_node["reloc_topic"].as<std::string>();
  }

  int feature;
  std::string feature_topic;
  int frame_pose;
  std::string frame_pose_topic;
  std::string frame_odometry_topic;
  int keyframe;
  std::string keyframe_topic;
  std::string path_topic;
  int map;
  std::string map_topic;
  int mapline;
  std::string mapline_topic;
  int reloc;
  std::string reloc_topic;
};


struct VisualOdometryConfigs{
  std::string dataroot;
  std::string camera_config_path;
  std::string model_dir;
  std::string saving_dir;

  PLNetConfig plnet_config;
  SuperPointConfig superpoint_config;
  PointMatcherConfig point_matcher_config;
  LineDetectorConfig line_detector_config;
  KeyframeConfig keyframe_config;
  OptimizationConfig tracking_optimization_config;
  OptimizationConfig backend_optimization_config;
  RosPublisherConfig ros_publisher_config;

  VisualOdometryConfigs() {}

  VisualOdometryConfigs(const std::string& config_file_, const std::string& model_dir_){
    model_dir = model_dir_;

    std::cout << "config_file = " << config_file_ << std::endl;
    if(!FileExists(config_file_)){
      std::cout << "config file: " << config_file_ << " doesn't exist" << std::endl;
      return;
    }
    YAML::Node file_node = YAML::LoadFile(config_file_);

    plnet_config.Load(file_node["plnet"]);
    plnet_config.SetModelPath(model_dir);

    point_matcher_config.Load(file_node["point_matcher"]);
    point_matcher_config.onnx_file = ConcatenateFolderAndFileName(model_dir, point_matcher_config.onnx_file);
    point_matcher_config.engine_file = ConcatenateFolderAndFileName(model_dir, point_matcher_config.engine_file);

    keyframe_config.Load(file_node["keyframe"]);
    tracking_optimization_config.Load(file_node["optimization"]["tracking"]);
    backend_optimization_config.Load(file_node["optimization"]["backend"]);
    ros_publisher_config.Load(file_node["ros_publisher"]);
  }
};

struct MapRefinementConfigs{
  PointMatcherConfig point_matcher_config;
  OptimizationConfig map_optimization_config;
  RosPublisherConfig ros_publisher_config;

  MapRefinementConfigs() {}

  MapRefinementConfigs(const std::string& config_file_, const std::string& model_dir_){
    std::cout << "config_file = " << config_file_ << std::endl;
    if(!FileExists(config_file_)){
      std::cout << "config file: " << config_file_ << " doesn't exist" << std::endl;
      return;
    }
    YAML::Node file_node = YAML::LoadFile(config_file_);

    point_matcher_config.Load(file_node["point_matcher"]);
    point_matcher_config.onnx_file = ConcatenateFolderAndFileName(model_dir_, point_matcher_config.onnx_file);
    point_matcher_config.engine_file = ConcatenateFolderAndFileName(model_dir_, point_matcher_config.engine_file);
    map_optimization_config.Load(file_node["optimization"]);
    ros_publisher_config.Load(file_node["ros_publisher"]);
  }
};

struct RelocalizationConfigs{
  std::string dataroot;
  std::string camera_config_path;
  std::string model_dir;
  std::string saving_dir;

  int min_inlier;
  int pose_refinement;
  PLNetConfig plnet_config;
  PointMatcherConfig point_matcher_config;
  OptimizationConfig pose_estimation_config;
  RosPublisherConfig ros_publisher_config;

  RelocalizationConfigs() {}

  RelocalizationConfigs(const std::string& config_file_, const std::string& model_dir_){
    model_dir = model_dir_;

    std::cout << "config_file = " << config_file_ << std::endl;
    if(!FileExists(config_file_)){
      std::cout << "config file: " << config_file_ << " doesn't exist" << std::endl;
      return;
    }

    YAML::Node file_node = YAML::LoadFile(config_file_);

    min_inlier = file_node["min_inlier_num"].as<int>();
    pose_refinement = file_node["pose_refinement"].as<int>();

    plnet_config.Load(file_node["plnet"]);
    plnet_config.SetModelPath(model_dir);

    point_matcher_config.Load(file_node["point_matcher"]);
    point_matcher_config.onnx_file = ConcatenateFolderAndFileName(model_dir, point_matcher_config.onnx_file);
    point_matcher_config.engine_file = ConcatenateFolderAndFileName(model_dir, point_matcher_config.engine_file);

    pose_estimation_config.Load(file_node["pose_estimation"]);
    ros_publisher_config.Load(file_node["ros_publisher"]);
  }
};


#endif  // READ_CONFIGS_H_
