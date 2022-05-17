#ifndef READ_CONFIGS_H_
#define READ_CONFIGS_H_

#include <iostream>
#include <yaml-cpp/yaml.h>

#include "utils.h"

struct SuperPointConfig {
  int max_keypoints;
  double keypoint_threshold;
  int remove_borders;
  int dla_core;
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_file;
  std::string engine_file;
};

struct SuperGlueConfig {
  std::vector<int> image_shape;
  int dla_core;
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_file;
  std::string engine_file;
};

struct KeyframeConfig {
  int min_num_match;
  int max_num_match;
  double max_distance;
  double max_angle;
};

struct Configs{
  std::string dataroot;
  std::string camera_config_path;
  std::string model_dir;
  std::string saving_dir;

  SuperPointConfig superpoint_config;
  SuperGlueConfig superglue_config;
  KeyframeConfig keyframe_config;

  Configs(const std::string& config_file){
    std::cout << "config_file = " << config_file << std::endl;
    if(!FileExists(config_file)){
      std::cout << "config file: " << config_file << " doesn't exist" << std::endl;
      return;
    }

    YAML::Node file_node = YAML::LoadFile(config_file);
    dataroot = file_node["dataroot"].as<std::string>();
    camera_config_path = file_node["camera_config_path"].as<std::string>();
    model_dir = file_node["model_dir"].as<std::string>();
    saving_dir = file_node["saving_dir"].as<std::string>();

    YAML::Node superpoint_node = file_node["superpoint"];
    superpoint_config.max_keypoints = superpoint_node["max_keypoints"].as<int>();
    superpoint_config.keypoint_threshold = superpoint_node["keypoint_threshold"].as<double>();
    superpoint_config.remove_borders = superpoint_node["remove_borders"].as<int>();
    superpoint_config.dla_core = superpoint_node["dla_core"].as<int>();
    YAML::Node superpoint_input_tensor_names_node = superpoint_node["input_tensor_names"];
    size_t superpoint_num_input_tensor_names = superpoint_input_tensor_names_node.size();
    for(size_t i = 0; i < superpoint_num_input_tensor_names; i++){
      superpoint_config.input_tensor_names.push_back(superpoint_input_tensor_names_node[i].as<std::string>());
    }
    YAML::Node superpoint_output_tensor_names_node = superpoint_node["output_tensor_names"];
    size_t superpoint_num_output_tensor_names = superpoint_output_tensor_names_node.size();
    for(size_t i = 0; i < superpoint_num_output_tensor_names; i++){
      superpoint_config.output_tensor_names.push_back(superpoint_output_tensor_names_node[i].as<std::string>());
    }
    std::string superpoint_onnx_file = superpoint_node["onnx_file"].as<std::string>();
    std::string superpoint_engine_file= superpoint_node["engine_file"].as<std::string>();
    superpoint_config.onnx_file = ConcatenateFolderAndFileName(model_dir, superpoint_onnx_file);
    superpoint_config.engine_file = ConcatenateFolderAndFileName(model_dir, superpoint_engine_file);

    
    YAML::Node superglue_node = file_node["superglue"];
    int image_width = superglue_node["image_width"].as<int>();
    int image_height = superglue_node["image_height"].as<int>();
    superglue_config.image_shape.push_back(image_height);
    superglue_config.image_shape.push_back(image_width);
    superglue_config.dla_core = superglue_node["dla_core"].as<int>();
    YAML::Node superglue_input_tensor_names_node = superglue_node["input_tensor_names"];
    size_t superglue_num_input_tensor_names = superglue_input_tensor_names_node.size();
    for(size_t i = 0; i < superglue_num_input_tensor_names; i++){
      superglue_config.input_tensor_names.push_back(superglue_input_tensor_names_node[i].as<std::string>());
    }
    YAML::Node superglue_output_tensor_names_node = superglue_node["output_tensor_names"];
    size_t superglue_num_output_tensor_names = superglue_output_tensor_names_node.size();
    for(size_t i = 0; i < superglue_num_output_tensor_names; i++){
      superglue_config.output_tensor_names.push_back(superglue_output_tensor_names_node[i].as<std::string>());
    }
    std::string superglue_onnx_file = superglue_node["onnx_file"].as<std::string>();
    std::string superglue_engine_file= superglue_node["engine_file"].as<std::string>();
    superglue_config.onnx_file = ConcatenateFolderAndFileName(model_dir, superglue_onnx_file);
    superglue_config.engine_file = ConcatenateFolderAndFileName(model_dir, superglue_engine_file); 

    YAML::Node keyframe_node = file_node["keyframe"];
    keyframe_config.min_num_match = keyframe_node["min_num_match"].as<int>();
    keyframe_config.max_num_match = keyframe_node["max_num_match"].as<int>();
    keyframe_config.max_distance = keyframe_node["max_distance"].as<double>();
    keyframe_config.max_angle = keyframe_node["max_angle"].as<double>();
  }
};



#endif  // READ_CONFIGS_H_