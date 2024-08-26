//
// Created by haoyuefan on 2021/9/22.
//
#include "plnet.h"

#include <memory>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <utility>
#include <chrono>

#include "NvInferPlugin.h"

using namespace tensorrt_log;
using namespace tensorrt_buffer;

PLNet::PLNet(PLNetConfig& plnet_config) : plnet_config_(plnet_config), engine0_(nullptr), 
    engine1_(nullptr), resized_width(512), resized_height(512){
  setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
  feature_width = resized_width / 4;
  feature_height = resized_height / 4;
}

bool PLNet::build() {
  if (deserialize_engine()) {
    if (!context0_) {
      context0_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(engine0_->createExecutionContext());
      if (!context0_) {
        return false;
      }
    }

    image_input_index_ = engine0_->getBindingIndex("input");
    if (!context1_) {
      context1_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(engine1_->createExecutionContext());
      if (!context1_) {
        return false;
      }
    }

    juncs_pred_index_ = engine1_->getBindingIndex("juncs_pred");
    lines_pred_index_ = engine1_->getBindingIndex("lines_pred");
    idx_lines_for_junctions_index_ = engine1_->getBindingIndex("idx_lines_for_junctions");
    inverse_index_ = engine1_->getBindingIndex("inverse");
    is_keep_index_index_ = engine1_->getBindingIndex("iskeep_index");
    loi_features_index_ = engine1_->getBindingIndex("loi_features");
    loi_features_thin_index_ = engine1_->getBindingIndex("loi_features_thin");
    loi_features_aux_index_ = engine1_->getBindingIndex("loi_features_aux");
    return true;
  }
  auto builder_stage1 = TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
  if (!builder_stage1) {
    return false;
  }
  const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network_stage1 = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder_stage1->createNetworkV2(explicit_batch));
  if (!network_stage1) {
    return false;
  }
  auto config_stage1 = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder_stage1->createBuilderConfig());
  if (!config_stage1) {
    return false;
  }
  auto parser_stage1 = TensorRTUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network_stage1, gLogger.getTRTLogger()));
  if (!parser_stage1) {
    return false;
  }

  auto profile_stage1 = builder_stage1->createOptimizationProfile();
  if (!profile_stage1) {
    return false;
  }
  profile_stage1->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 1, 100, 100));
  profile_stage1->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 1, 512, 512));
  profile_stage1->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 1, 1500, 1500));
  config_stage1->addOptimizationProfile(profile_stage1);

  auto constructed_stage1 = construct_network_stage1(builder_stage1, network_stage1, config_stage1, parser_stage1);
  if (!constructed_stage1) {
    return false;
  }
  auto profile_stream_stage1 = makeCudaStream();
  if (!profile_stream_stage1) {
    return false;
  }
  config_stage1->setProfileStream(*profile_stream_stage1);
  TensorRTUniquePtr<nvinfer1::IHostMemory> plan_stage1{builder_stage1->buildSerializedNetwork(*network_stage1, *config_stage1)};
  if (!plan_stage1) {
    return false;
  }
  TensorRTUniquePtr<nvinfer1::IRuntime> runtime_stage1{nvinfer1::createInferRuntime(gLogger.getTRTLogger())};
  if (!runtime_stage1) {
    return false;
  }
  engine0_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_stage1->deserializeCudaEngine(plan_stage1->data(), plan_stage1->size()));
  if (!engine0_) {
    return false;
  }

  if (!context0_) {
    context0_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(engine0_->createExecutionContext());
    if (!context0_) {
      return false;
    }
  }

  image_input_index_ = engine0_->getBindingIndex("input");

  auto builder_stage2 = TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
  if (!builder_stage2) {
    return false;
  }
  auto network_stage2 = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder_stage2->createNetworkV2(explicit_batch));
  if (!network_stage2) {
    return false;
  }
  auto config_stage2 = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder_stage2->createBuilderConfig());
  if (!config_stage2) {
    return false;
  }
  auto parser_stage2 = TensorRTUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network_stage2, gLogger.getTRTLogger()));
  if (!parser_stage2) {
    return false;
  }

  auto profile_stage2 = builder_stage2->createOptimizationProfile();
  if (!profile_stage2) {
    return false;
  }
  profile_stage2->setDimensions("juncs_pred", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 2));
  profile_stage2->setDimensions("juncs_pred", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(250, 2));
  profile_stage2->setDimensions("juncs_pred", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(500, 2));
  profile_stage2->setDimensions("lines_pred", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 4));
  profile_stage2->setDimensions("lines_pred", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(20000, 4));
  profile_stage2->setDimensions("lines_pred", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(50000, 4));
  profile_stage2->setDimensions("idx_lines_for_junctions", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 2));
  profile_stage2->setDimensions("idx_lines_for_junctions", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(20000, 2));
  profile_stage2->setDimensions("idx_lines_for_junctions", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(50000, 2));
  profile_stage2->setDimensions("inverse", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 1));
  profile_stage2->setDimensions("inverse", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(20000, 1));
  profile_stage2->setDimensions("inverse", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(50000, 1));
  profile_stage2->setDimensions("iskeep_index", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2(1, 1));
  profile_stage2->setDimensions("iskeep_index", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2(20000, 1));
  profile_stage2->setDimensions("iskeep_index", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2(50000, 1));
  profile_stage2->setDimensions("loi_features", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 16, 16, 16));
  profile_stage2->setDimensions("loi_features", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 128, 128, 128));
  profile_stage2->setDimensions("loi_features", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 512, 512, 512));
  profile_stage2->setDimensions("loi_features_thin", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 4, 16, 16));
  profile_stage2->setDimensions("loi_features_thin", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 4, 128, 128));
  profile_stage2->setDimensions("loi_features_thin", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 4, 512, 512));
  profile_stage2->setDimensions("loi_features_aux", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 4, 16, 16));
  profile_stage2->setDimensions("loi_features_aux", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 4, 128, 128));
  profile_stage2->setDimensions("loi_features_aux", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 4, 512, 512));
  config_stage2->addOptimizationProfile(profile_stage2);

  auto constructed_stage2 = construct_network_stage2(builder_stage2, network_stage2, config_stage2, parser_stage2);
  if (!constructed_stage2) {
    return false;
  }
  auto profile_stream_stage2 = makeCudaStream();
  if (!profile_stream_stage2) {
    return false;
  }
  config_stage2->setProfileStream(*profile_stream_stage2);
  TensorRTUniquePtr<nvinfer1::IHostMemory> plan_stage2{builder_stage2->buildSerializedNetwork(*network_stage2, *config_stage2)};
  if (!plan_stage2) {
    return false;
  }
  TensorRTUniquePtr<nvinfer1::IRuntime> runtime_stage2{nvinfer1::createInferRuntime(gLogger.getTRTLogger())};
  if (!runtime_stage2) {
    return false;
  }
  engine1_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_stage2->deserializeCudaEngine(plan_stage2->data(), plan_stage2->size()));
  if (!engine1_) {
    return false;
  }
  save_engine();

  if (!context1_) {
    context1_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(engine1_->createExecutionContext());
    if (!context1_) {
      return false;
    }
  }

  juncs_pred_index_ = engine1_->getBindingIndex("juncs_pred");
  lines_pred_index_ = engine1_->getBindingIndex("lines_pred");
  idx_lines_for_junctions_index_ = engine1_->getBindingIndex("idx_lines_for_junctions");
  inverse_index_ = engine1_->getBindingIndex("inverse");
  is_keep_index_index_ = engine1_->getBindingIndex("iskeep_index");
  loi_features_index_ = engine1_->getBindingIndex("loi_features");
  loi_features_thin_index_ = engine1_->getBindingIndex("loi_features_thin");
  loi_features_aux_index_ = engine1_->getBindingIndex("loi_features_aux");

  return true;
}


bool PLNet::construct_network_stage1(TensorRTUniquePtr<nvinfer1::IBuilder> &builder, TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                              TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config, TensorRTUniquePtr<nvonnxparser::IParser> &parser) const {
  auto parsed = parser->parseFromFile(plnet_config_.plnet_s0_onnx.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
  if (!parsed) {
    return false;
  }
  config->setFlag(nvinfer1::BuilderFlag::kTF32);
  enableDLA(builder.get(), config.get(), -1);
  return true;
}

bool PLNet::construct_network_stage2(TensorRTUniquePtr<nvinfer1::IBuilder> &builder, TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                                  TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config, TensorRTUniquePtr<nvonnxparser::IParser> &parser) const {
  auto parsed = parser->parseFromFile(plnet_config_.plnet_s1_onnx.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
  if (!parsed) {
    return false;
  }
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
  enableDLA(builder.get(), config.get(), -1);
  return true;
}

bool PLNet::infer(const cv::Mat &image, Eigen::Matrix<float, 259, Eigen::Dynamic> &features, 
    std::vector<Eigen::Vector4d>& lines, Eigen::Matrix<float, 259, Eigen::Dynamic>& junctions, bool junction_detection) {

  context0_->setBindingDimensions(image_input_index_, nvinfer1::Dims4(1, 1, resized_height, resized_width));

  BufferManager buffers0(engine0_, 0, context0_.get());

  if (!process_image(buffers0, image)) {
    return false;
  }
  buffers0.copyInputToDevice();

  bool status = context0_->executeV2(buffers0.getDeviceBindings().data());
  if (!status) {
    return false;
  }
  buffers0.copyOutputToHost();

  if (!process_output(buffers0, features, lines, junctions, junction_detection)) {
    return false;
  }
           
  return true;
}

bool PLNet::process_image(const BufferManager &buffers, const cv::Mat &image) {
  if (image.empty()) return false;

  input_width = image.cols;
  input_height = image.rows;

  w_scale = (float)input_width / resized_width;
  h_scale = (float)input_height / resized_height;

  auto *host_data_buffer = static_cast<float *>(buffers.getHostBuffer("input"));

  cv::Mat resized_image;
  cv::resize(image, resized_image, cv::Size(resized_width, resized_height));

  for (int row = 0; row < resized_height; ++row) {
    const uchar *ptr = resized_image.ptr(row);
    int row_shift = row * resized_width;
    for (int col = 0; col < resized_width; ++col) {
      host_data_buffer[row_shift + col] = float(ptr[0]) / 255.0;
      ptr++;
    }
  }

  return true;
}

bool PLNet::wireframe_matcher(const float* iskeep, const float* idx_junc_to_end_min, const float* idx_junc_to_end_max){
  is_keep_index_.clear();
  idx_lines_for_junctions_unique_.clear();
  inverse_.clear();

  for (int i = 0; i < 3 * feature_height * feature_width; ++i) {
    if(iskeep[i] > 0){
      is_keep_index_.push_back(i);
    }
  }

  int unique_id = 1;
  const int JN = 300; // top k junction number
  std::vector<std::vector<int> > unique_map(JN, std::vector<int>(JN, 0));
  for(auto &index : is_keep_index_) {
    int x = (int)idx_junc_to_end_min[index];
    int y = (int)idx_junc_to_end_max[index];

    if(unique_map[x][y] <= 0){
      unique_map[x][y] = unique_id++;
    }
    inverse_.push_back(unique_map[x][y]-1);
  }

  idx_lines_for_junctions_unique_.resize((unique_id-1));
  for (int i = 0; i < JN; ++i) {
    for (int j = 0; j < JN; ++j) {
      int value = unique_map[i][j];
      if(value){
        idx_lines_for_junctions_unique_[value-1] = std::make_pair(j, i);
      }
    }
  }

  return true;
}

void PLNet::detect_point(const float* heat_map, Eigen::Matrix<float, 259, Eigen::Dynamic>& features, 
    int h, int w, float threshold, int border, int top_k) {
  std::vector<float> scores_v;
  std::vector<float> kpt_xs, kpt_ys;

  scores_v.reserve(top_k);
  kpt_xs.reserve(top_k);
  kpt_ys.reserve(top_k);

  int heat_map_size = w * h;

  int min_x = border;
  int min_y = border;
  int max_x = w - border;
  int max_y = h - border;

  for(int i = 0; i < heat_map_size; ++i){
    float score = *(heat_map+i);
    if(score < threshold) continue;

    int y = int(i / w);
    int x = i - y * w;

    if(x < min_x || x > max_x || y < min_y || y > max_y) continue;

    scores_v.emplace_back(*(heat_map+i));
    kpt_xs.emplace_back(float(x));
    kpt_ys.emplace_back(float(y));
  }


  if(scores_v.size() > top_k){
    std::vector<size_t> indexes = sort_indexes(scores_v);
    features.resize(259, top_k);
    for(int i = 0; i < top_k; ++i){
      features(0, i) = scores_v[indexes[i]];
      features(1, i) = kpt_xs[indexes[i]];
      features(2, i) = kpt_ys[indexes[i]];
    }
  }else{
    features.resize(259, scores_v.size());
    features.row(0) = Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic>>(scores_v.data(), 1, scores_v.size());
    features.row(1) = Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic>>(kpt_xs.data(), 1, kpt_xs.size());
    features.row(2) = Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic>>(kpt_ys.data(), 1, kpt_ys.size());
  }

}

std::vector<size_t> PLNet::sort_indexes(std::vector<float> &data) {
  std::vector<size_t> indexes(data.size());
  iota(indexes.begin(), indexes.end(), 0);
  sort(indexes.begin(), indexes.end(), [&data](size_t i1, size_t i2) { return data[i1] > data[i2]; });
  return indexes;
}

int PLNet::clip(int val, int max) {
  if (val < 0) return 0;
  return std::min(val, max - 1);
}

void PLNet::extract_descriptors(const float *descriptors, Eigen::Matrix<float, 259, Eigen::Dynamic> &features, int h, int w, int s){
  float sx = 2.f / (w * s - s / 2 - 0.5);
  float bx = (1 - s) / (w * s - s / 2 - 0.5) - 1;

  float sy = 2.f / (h * s - s / 2 - 0.5);
  float by = (1 - s) / (h * s - s / 2 - 0.5) - 1;

  Eigen::Array<float, 1, Eigen::Dynamic> kpts_x_norm = features.block(1, 0, 1, features.cols()).array() * sx + bx;
  Eigen::Array<float, 1, Eigen::Dynamic> kpts_y_norm = features.block(2, 0, 1, features.cols()).array() * sy + by;

  kpts_x_norm = (kpts_x_norm + 1) * 0.5;
  kpts_y_norm = (kpts_y_norm + 1) * 0.5;

  for(int j = 0; j < features.cols(); ++j){
    float ix = kpts_x_norm(0, j) * (w - 1);
    float iy = kpts_y_norm(0, j) * (h - 1);

    int ix_nw = clip(std::floor(ix), w);
    int iy_nw = clip(std::floor(iy), h);

    int ix_ne = clip(ix_nw + 1, w);
    int iy_ne = clip(iy_nw, h);

    int ix_sw = clip(ix_nw, w);
    int iy_sw = clip(iy_nw + 1, h);

    int ix_se = clip(ix_nw + 1, w);
    int iy_se = clip(iy_nw + 1, h);

    float nw = (ix_se - ix) * (iy_se - iy);
    float ne = (ix - ix_sw) * (iy_sw - iy);
    float sw = (ix_ne - ix) * (iy - iy_ne);
    float se = (ix - ix_nw) * (iy - iy_nw);

    for (int i = 0; i < 256; ++i) {
      // 256x60x106 dhw
      // x * height * depth + y * depth + z
      float nw_val = descriptors[i * h * w + iy_nw * w + ix_nw];
      float ne_val = descriptors[i * h * w + iy_ne * w + ix_ne];
      float sw_val = descriptors[i * h * w + iy_sw * w + ix_sw];
      float se_val = descriptors[i * h * w + iy_se * w + ix_se];
      features(i+3, j) = nw_val * nw + ne_val * ne + sw_val * sw + se_val * se;
    }
  }

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> descriptor_matrix = features.block(3, 0, features.rows() - 3, features.cols());
  descriptor_matrix.colwise().normalize();
  features.block(3, 0, features.rows() - 3, features.cols()) = descriptor_matrix;
}

bool PLNet::keypoints_decoder(const float* scores, const float* descriptors, Eigen::Matrix<float, 259, Eigen::Dynamic> &features){
  detect_point(scores, features, resized_height, resized_width, plnet_config_.keypoint_threshold, plnet_config_.remove_borders, plnet_config_.max_keypoints);
  extract_descriptors(descriptors, features, resized_height / 8, resized_width / 8, 8);
  return true;
}

bool PLNet::junction_detector(const float* scores, const float* descriptors, 
    std::vector<std::vector<bool>>& junction_map, Eigen::Matrix<float, 259, Eigen::Dynamic> &junctions){
  int border = std::max(plnet_config_.remove_borders, 0);
  std::vector<float> js, jx, jy;
  for(int y = border; y < resized_height-border; y++){
    for(int x = border; x < resized_width-border; x++){
      if(junction_map[y][x]){
        js.push_back(*(scores+x+y*resized_width));
        jx.push_back(x);
        jy.push_back(y);
      }
    }
  }


  junctions.resize(259, js.size());
  junctions.row(0) = Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic>>(js.data(), 1, js.size());
  junctions.row(1) = Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic>>(jx.data(), 1, jx.size());
  junctions.row(2) = Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic>>(jy.data(), 1, jy.size());

  extract_descriptors(descriptors, junctions, resized_height / 8, resized_width / 8, 8);

  return true;
}

bool PLNet::process_output(const BufferManager &buffers, Eigen::Matrix<float, 259, Eigen::Dynamic> &features, 
    std::vector<Eigen::Vector4d>& lines, Eigen::Matrix<float, 259, Eigen::Dynamic>& junctions, bool junction_detection) {

  auto *iskeep = static_cast<float *>(buffers.getHostBuffer("iskeep"));                            // 1x3x128x128
  auto *idx_junc_to_end_min = static_cast<float *>(buffers.getHostBuffer("idx_junc_to_end_min"));  // 1x3x128x128
  auto *idx_junc_to_end_max = static_cast<float *>(buffers.getHostBuffer("idx_junc_to_end_max"));  // 1x3x128x128
  auto *juncs_pred = static_cast<float *>(buffers.getHostBuffer("juncs_pred"));                    // 1x3x128x128
  auto *lines_pred = static_cast<float *>(buffers.getHostBuffer("lines_pred"));                    // 1x3x128x128
  auto *loi_features = static_cast<float *>(buffers.getHostBuffer("loi_features"));                // 1x3x128x128
  auto *loi_features_thin = static_cast<float *>(buffers.getHostBuffer("loi_features_thin"));      // 1x3x128x128
  auto *loi_features_aux = static_cast<float *>(buffers.getHostBuffer("loi_features_aux"));        // 1x3x128x128
  auto *scores = static_cast<float *>(buffers.getHostBuffer("scores"));                            // 1x512x512
  auto *descriptors = static_cast<float *>(buffers.getHostBuffer("descriptors"));                  // 1x256x64x64

  if(!wireframe_matcher(iskeep, idx_junc_to_end_min, idx_junc_to_end_max)){
    return false;
  }

  context1_->setBindingDimensions(juncs_pred_index_, nvinfer1::Dims2(300, 2));
  context1_->setBindingDimensions(lines_pred_index_, nvinfer1::Dims2(128 * 128 * 3, 4));
  context1_->setBindingDimensions(idx_lines_for_junctions_index_, nvinfer1::Dims2((int)idx_lines_for_junctions_unique_.size(), 2));
  context1_->setBindingDimensions(inverse_index_, nvinfer1::Dims2((int)inverse_.size(), 1));
  context1_->setBindingDimensions(is_keep_index_index_, nvinfer1::Dims2((int)is_keep_index_.size(), 1));
  context1_->setBindingDimensions(loi_features_index_, nvinfer1::Dims4(1, 128, 128, 128));
  context1_->setBindingDimensions(loi_features_thin_index_, nvinfer1::Dims4(1, 4, 128, 128));
  context1_->setBindingDimensions(loi_features_aux_index_, nvinfer1::Dims4(1, 4, 128, 128));

  BufferManager buffers1(engine1_, 0, context1_.get());

  auto *juncs_pred_hbuffer = static_cast<float *>(buffers1.getHostBuffer("juncs_pred"));
  auto *lines_pred_hbuffer = static_cast<float *>(buffers1.getHostBuffer("lines_pred"));
  auto *idx_lines_for_junctions_hbuffer = static_cast<float *>(buffers1.getHostBuffer("idx_lines_for_junctions"));
  auto *inverse_hbuffer = static_cast<float *>(buffers1.getHostBuffer("inverse"));
  auto *iskeep_index_hbuffer = static_cast<float *>(buffers1.getHostBuffer("iskeep_index"));
  auto *loi_features_hbuffer = static_cast<float *>(buffers1.getHostBuffer("loi_features"));
  auto *loi_features_thin_hbuffer = static_cast<float *>(buffers1.getHostBuffer("loi_features_thin"));
  auto *loi_features_aux_hbuffer = static_cast<float *>(buffers1.getHostBuffer("loi_features_aux"));

  memcpy(juncs_pred_hbuffer, juncs_pred, 300 * 2 * sizeof(float));
  memcpy(lines_pred_hbuffer, lines_pred, 128 * 128 * 3 * 4 * sizeof(float));
  memcpy(loi_features_hbuffer, loi_features, 128 * 128 * 128 * sizeof(float));
  memcpy(loi_features_thin_hbuffer, loi_features_thin, 128 * 128 * 4 * sizeof(float));
  memcpy(loi_features_aux_hbuffer, loi_features_aux, 128 * 128 * 4 * sizeof(float));

  int index = 0;
  for (auto &i : idx_lines_for_junctions_unique_) {
    idx_lines_for_junctions_hbuffer[index] = (float)i.first;
    idx_lines_for_junctions_hbuffer[index + 1] = (float)i.second;
    index = index + 2;
  }

  for (int i = 0; i < is_keep_index_.size(); ++i) {
    iskeep_index_hbuffer[i] = (float)is_keep_index_[i];
  }

  for (int i = 0; i < inverse_.size(); ++i) {
    inverse_hbuffer[i] = (float)inverse_[i];
  }

  buffers1.copyInputToDevice();
  bool status = context1_->executeV2(buffers1.getDeviceBindings().data());
  if (!status) {
    return false;
  }
  buffers1.copyOutputToHost();

  auto *line_ajusted_hbuffer = static_cast<float *>(buffers1.getHostBuffer("lines_adjusted"));
  auto *scores_line_hbuffer = static_cast<float *>(buffers1.getHostBuffer("scores_line"));

  std::vector<std::vector<bool>> junction_map(resized_height, std::vector<bool>(resized_width, false));
  const float length_square_threshold = plnet_config_.line_length_threshold * plnet_config_.line_length_threshold;
  for (int i = 0; i < idx_lines_for_junctions_unique_.size(); ++i) {
    if (scores_line_hbuffer[i] < 0.5) continue;

    float x1 = line_ajusted_hbuffer[i * 4] * 4;
    float y1 = line_ajusted_hbuffer[i * 4 + 1] * 4;
    float x2 = line_ajusted_hbuffer[i * 4 + 2] * 4;
    float y2 = line_ajusted_hbuffer[i * 4 + 3] * 4;

    int xi1 = (int)(x1+0.1); // x1, x2, y1, and y2 are converted from int in fact.
    int yi1 = (int)(y1+0.1);
    int xi2 = (int)(x2+0.1);
    int yi2 = (int)(y2+0.1);
    int border = std::max(plnet_config_.remove_borders, 0);
    bool p1_valid = (xi1 > border) && (xi1 < resized_width - border) && (yi1 > border) && (yi1 < resized_height - border);
    bool p2_valid = (xi2 > border) && (xi2 < resized_width - border) && (yi2 > border) && (yi2 < resized_height - border);
    junction_map[yi1][xi1] = p1_valid;
    junction_map[yi2][xi2] = p2_valid;

    if(scores_line_hbuffer[i] < plnet_config_.line_threshold) continue;

    float length_square = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1);
    if(length_square < length_square_threshold) continue;

    lines.emplace_back(x1, y1, x2, y2);

    // add some junctions to keypoints by add scores
    // if(scores_line_hbuffer[i] > 0.8 && junction_detection){
    //   if(p1_valid){
    //     int idx1 = xi1 + yi1 * resized_width;
    //     *(scores + idx1) += plnet_config_.keypoint_threshold;
    //   }

    //   if(p2_valid){
    //     int idx2 = xi2 + yi2 * resized_width;
    //     *(scores + idx2) += plnet_config_.keypoint_threshold;
    //   }
    // }
  }

  if(!keypoints_decoder(scores, descriptors, features)){
    return false;
  }


  if(junction_detection){
    if(!junction_detector(scores, descriptors, junction_map, junctions)){
      return false;
    }
    junctions.block(1, 0, 1, junctions.cols()) = junctions.block(1, 0, 1, junctions.cols()) * w_scale;
    junctions.block(2, 0, 1, junctions.cols()) = junctions.block(2, 0, 1, junctions.cols()) * h_scale;
  }

  // re-scale
  features.block(1, 0, 1, features.cols()) = features.block(1, 0, 1, features.cols()) * w_scale;
  features.block(2, 0, 1, features.cols()) = features.block(2, 0, 1, features.cols()) * h_scale;

  for(int i = 0; i < lines.size(); i++){
    lines[i][0] *= w_scale;
    lines[i][1] *= h_scale;
    lines[i][2] *= w_scale;
    lines[i][3] *= h_scale;
  }

  return true;
}

void PLNet::save_engine() {
  if (engine0_ != nullptr) {
    nvinfer1::IHostMemory *data = engine0_->serialize();
    std::ofstream file(plnet_config_.plnet_s0_engine, std::ios::binary);
    if (!file) return;
    file.write(reinterpret_cast<const char *>(data->data()), data->size());
  }
  if (engine1_ != nullptr) {
    nvinfer1::IHostMemory *data = engine1_->serialize();
    std::ofstream file(plnet_config_.plnet_s1_engine, std::ios::binary);
    if (!file) return;
    file.write(reinterpret_cast<const char *>(data->data()), data->size());
  }
}

bool PLNet::deserialize_engine() {
  std::ifstream file0(plnet_config_.plnet_s0_engine, std::ios::binary);
  std::ifstream file1(plnet_config_.plnet_s1_engine, std::ios::binary);
  bool deserialize0 = false;
  bool deserialize1 = false;
  if (file0.is_open()) {
    file0.seekg(0, std::ifstream::end);
    size_t size = file0.tellg();
    file0.seekg(0, std::ifstream::beg);
    char *model_stream = new char[size];
    file0.read(model_stream, size);
    file0.close();
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
    if (runtime == nullptr) {
      delete[] model_stream;
    }
    engine0_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_stream, size));
    if (engine0_ == nullptr) {
      delete[] model_stream;
    }
    delete[] model_stream;
    deserialize0 = true;
  }
  if (file1.is_open()) {
    file1.seekg(0, std::ifstream::end);
    size_t size = file1.tellg();
    file1.seekg(0, std::ifstream::beg);
    char *model_stream = new char[size];
    file1.read(model_stream, size);
    file1.close();
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
    if (runtime == nullptr) {
      delete[] model_stream;
    }
    engine1_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_stream, size));
    if (engine1_ == nullptr) {
      delete[] model_stream;
    }
    delete[] model_stream;
    deserialize1 = true;
  }
  return deserialize0 && deserialize1;
}