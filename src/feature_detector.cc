#include <opencv2/opencv.hpp>

#include "plnet.h"
#include "feature_detector.h"
#include "utils.h"

FeatureDetector::FeatureDetector(const PLNetConfig& plnet_config) : _plnet_config(plnet_config){
  if(_plnet_config.use_superpoint){
    SuperPointConfig superpoint_config;
    superpoint_config.max_keypoints = plnet_config.max_keypoints;
    superpoint_config.keypoint_threshold = plnet_config.keypoint_threshold;
    superpoint_config.remove_borders = plnet_config.remove_borders;
    superpoint_config.dla_core = -1;

    superpoint_config.input_tensor_names.push_back("input");
    superpoint_config.output_tensor_names.push_back("scores");
    superpoint_config.output_tensor_names.push_back("descriptors");

    superpoint_config.onnx_file = plnet_config.superpoint_onnx;
    superpoint_config.engine_file = plnet_config.superpoint_engine;

    _superpoint = std::shared_ptr<SuperPoint>(new SuperPoint(superpoint_config));
    if (!_superpoint->build()){
      std::cout << "Error in SuperPoint building" << std::endl;
      exit(0);
    }
  }

  _plnet = std::shared_ptr<PLNet>(new PLNet(_plnet_config));
  if (!_plnet->build()){
    std::cout << "Error in FeatureDetector building" << std::endl;
    // exit(0);
  }
}

bool FeatureDetector::Detect(cv::Mat& image, Eigen::Matrix<float, 259, Eigen::Dynamic> &features){
  bool good_infer = false;
  if(_plnet_config.use_superpoint){
    good_infer = _superpoint->infer(image, features);
  }else{
    std::vector<Eigen::Vector4d> lines;
    good_infer = Detect(image, features, lines);
  }


  if(!good_infer){
    std::cout << "Failed when extracting point features !" << std::endl;
  }
  return good_infer; 
}

bool FeatureDetector::Detect(cv::Mat& image, Eigen::Matrix<float, 259, Eigen::Dynamic> &features, 
    std::vector<Eigen::Vector4d>& lines){
  Eigen::Matrix<float, 259, Eigen::Dynamic> junctions;
  bool good_infer = _plnet->infer(image, features, lines, junctions);
  if(!good_infer){
    std::cout << "Failed when extracting point features !" << std::endl;
  }
  return good_infer; 
}

bool FeatureDetector::Detect(cv::Mat& image, Eigen::Matrix<float, 259, Eigen::Dynamic> &features, 
    std::vector<Eigen::Vector4d>& lines, Eigen::Matrix<float, 259, Eigen::Dynamic>& junctions){
  bool good_infer = _plnet->infer(image, features, lines, junctions, true);
  if(!good_infer){
    std::cout << "Failed when extracting point features !" << std::endl;
  }
  return good_infer; 
}

bool FeatureDetector::Detect(cv::Mat& image_left, cv::Mat& image_right, 
    Eigen::Matrix<float, 259, Eigen::Dynamic> & left_features, 
    Eigen::Matrix<float, 259, Eigen::Dynamic> & right_features){
  bool good_infer_left = Detect(image_left, left_features);
  bool good_infer_right = Detect(image_right, right_features);
  bool good_infer = good_infer_left & good_infer_right;
  if(!good_infer){
    std::cout << "Failed when extracting point features !" << std::endl;
  }
  return good_infer; 
}

bool FeatureDetector::Detect(cv::Mat& image_left, cv::Mat& image_right, 
    Eigen::Matrix<float, 259, Eigen::Dynamic> & left_features, 
    Eigen::Matrix<float, 259, Eigen::Dynamic> & right_features, 
    std::vector<Eigen::Vector4d>& left_lines, 
    std::vector<Eigen::Vector4d>& right_lines){
  bool good_infer_left = Detect(image_left, left_features, left_lines);
  bool good_infer_right = Detect(image_right, right_features, right_lines);
  bool good_infer = good_infer_left & good_infer_right;
  if(!good_infer){
    std::cout << "Failed when extracting point features !" << std::endl;
  }
  return good_infer; 
}

bool FeatureDetector::Detect(cv::Mat& image_left, cv::Mat& image_right, Eigen::Matrix<float, 259, Eigen::Dynamic> & left_features, 
    Eigen::Matrix<float, 259, Eigen::Dynamic> & right_features, std::vector<Eigen::Vector4d>& left_lines, 
    std::vector<Eigen::Vector4d>& right_lines, Eigen::Matrix<float, 259, Eigen::Dynamic>& junctions){
  bool good_infer_left = Detect(image_left, left_features, left_lines, junctions);
  bool good_infer_right = Detect(image_right, right_features, right_lines);

  bool good_infer = good_infer_left & good_infer_right;
  if(!good_infer){
    std::cout << "Failed when extracting point features !" << std::endl;
  }
  return good_infer; 
}