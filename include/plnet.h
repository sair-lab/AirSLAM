#ifndef PLNET_PLNET_H
#define PLNET_PLNET_H

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <Eigen/Core>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

#include "3rdparty/tensorrtbuffer/include/buffers.h"
#include "read_configs.h"

using tensorrt_buffer::TensorRTUniquePtr;

class PLNet {
 public:
  PLNet(PLNetConfig& plnet_config);

  bool build();

  bool infer(const cv::Mat &image, Eigen::Matrix<float, 259, Eigen::Dynamic> &features, 
      std::vector<Eigen::Vector4d>& lines, Eigen::Matrix<float, 259, Eigen::Dynamic>& junctions, bool junction_detection = false);

  void save_engine();

  bool deserialize_engine();

 private:
  PLNetConfig plnet_config_;

  std::shared_ptr<nvinfer1::ICudaEngine> engine0_;
  std::shared_ptr<nvinfer1::IExecutionContext> context0_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine1_;
  std::shared_ptr<nvinfer1::IExecutionContext> context1_;

  int input_width;
  int input_height;
  int resized_width;
  int resized_height;

  float w_scale;
  float h_scale; 

  int feature_width;
  int feature_height;


  int image_input_index_;
  int juncs_pred_index_;
  int lines_pred_index_;
  int idx_lines_for_junctions_index_;
  int inverse_index_;
  int is_keep_index_index_;
  int loi_features_index_;
  int loi_features_thin_index_;
  int loi_features_aux_index_;


  std::vector<int> is_keep_index_;
  std::vector<int> inverse_;
  std::vector<std::pair<int, int>> idx_lines_for_junctions_unique_;

  bool construct_network_stage1(TensorRTUniquePtr<nvinfer1::IBuilder> &builder, TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                         TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config, TensorRTUniquePtr<nvonnxparser::IParser> &parser) const;

  bool construct_network_stage2(TensorRTUniquePtr<nvinfer1::IBuilder> &builder, TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                             TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config, TensorRTUniquePtr<nvonnxparser::IParser> &parser) const;

  bool process_image(const tensorrt_buffer::BufferManager &buffers, const cv::Mat &image);

  bool process_output(const tensorrt_buffer::BufferManager &buffers, Eigen::Matrix<float, 259, Eigen::Dynamic> &features, 
      std::vector<Eigen::Vector4d>& lines, Eigen::Matrix<float, 259, Eigen::Dynamic>& junctions, bool junction_detection);

  bool wireframe_matcher(const float* iskeep, const float* idx_junc_to_end_min, const float* idx_junc_to_end_max);

  bool keypoints_decoder(const float* scores, const float* descriptors, Eigen::Matrix<float, 259, Eigen::Dynamic> &features);

  bool junction_detector(const float* scores, const float* descriptors, 
      std::vector<std::vector<bool>>& junction_map, Eigen::Matrix<float, 259, Eigen::Dynamic> &junctions);

  std::vector<size_t> sort_indexes(std::vector<float> &data);
  int clip(int val, int max);

  void detect_point(const float* heat_map, Eigen::Matrix<float, 259, Eigen::Dynamic>& features, int h, int w, float threshold, int border, int top_k);

  void extract_descriptors(const float *descriptors, Eigen::Matrix<float, 259, Eigen::Dynamic> &features, int h, int w, int s);
};

typedef std::shared_ptr<PLNet> PLNetPtr;
#endif  // PLNET_PLNET_H