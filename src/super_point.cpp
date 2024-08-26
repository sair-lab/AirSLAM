//
// Created by haoyuefan on 2021/9/22.
//
#include "super_point.h"
#include <utility>
#include <unordered_map>
#include <opencv2/opencv.hpp>

using namespace tensorrt_log;
using namespace tensorrt_buffer;

SuperPoint::SuperPoint(const SuperPointConfig &super_point_config): resized_width(512), 
        resized_height(512), super_point_config_(super_point_config), engine_(nullptr) {
    setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
    // setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
}

bool SuperPoint::build() {
    // cudaSetDevice(2);
    if(deserialize_engine()){
        return true;
    }
    auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder) {
        return false;
    }
    const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
    if (!network) {
        return false;
    }
    auto config = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }
    auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(
            nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser) {
        return false;
    }
    
    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        return false;
    }
    profile->setDimensions(super_point_config_.input_tensor_names[0].c_str(),
                           nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 1, 100, 100));
    profile->setDimensions(super_point_config_.input_tensor_names[0].c_str(),
                           nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 1, 500, 500));
    profile->setDimensions(super_point_config_.input_tensor_names[0].c_str(),
                           nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 1, 1500, 1500));
    config->addOptimizationProfile(profile);
    
    auto constructed = construct_network(builder, network, config, parser);
    if (!constructed) {
        return false;
    }
    auto profile_stream = makeCudaStream();
    if (!profile_stream) {
        return false;
    }
    config->setProfileStream(*profile_stream);
    TensorRTUniquePtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }
    TensorRTUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger.getTRTLogger())};
    if (!runtime) {
        return false;
    }
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine_) {
        return false;
    }
    save_engine();
    ASSERT(network->getNbInputs() == 1);
    input_dims_ = network->getInput(0)->getDimensions();
    ASSERT(input_dims_.nbDims == 4);
    ASSERT(network->getNbOutputs() == 2);
    semi_dims_ = network->getOutput(0)->getDimensions();
    ASSERT(semi_dims_.nbDims == 3);
    desc_dims_ = network->getOutput(1)->getDimensions();
    ASSERT(desc_dims_.nbDims == 4);
    return true;
}

bool SuperPoint::construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder,
                                   TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                                   TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                                   TensorRTUniquePtr<nvonnxparser::IParser> &parser) const {
    auto parsed = parser->parseFromFile(super_point_config_.onnx_file.c_str(),
                                        static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }
    // config->setMaxWorkspaceSize(512_MiB);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    enableDLA(builder.get(), config.get(), super_point_config_.dla_core);
    return true;
}


bool SuperPoint::infer(const cv::Mat &image_, Eigen::Matrix<float, 259, Eigen::Dynamic> &features) {
    if (!context_) {
        context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_) {
            return false;
        }
    }

    input_height = image_.rows;
    input_width = image_.cols;
    h_scale = (float)input_height / resized_height;
    w_scale = (float)input_width / resized_width;
    cv::Mat image;
    cv::resize(image_, image, cv::Size(resized_width, resized_height));

    assert(engine_->getNbBindings() == 3);

    const int input_index = engine_->getBindingIndex(super_point_config_.input_tensor_names[0].c_str());

    context_->setBindingDimensions(input_index, nvinfer1::Dims4(1, 1, image.rows, image.cols));

    BufferManager buffers(engine_, 0, context_.get());
    
    ASSERT(super_point_config_.input_tensor_names.size() == 1);
    if (!process_input(buffers, image)) {
        return false;
    }

    buffers.copyInputToDevice();

    bool status = context_->executeV2(buffers.getDeviceBindings().data());
    if (!status) {
        return false;
    }

    buffers.copyOutputToHost();
    if (!process_output(buffers, features)) {
        return false;
    }

    return true;
}

bool SuperPoint::process_input(const BufferManager &buffers, const cv::Mat &image) {
    input_dims_.d[2] = image.rows;
    input_dims_.d[3] = image.cols;
    semi_dims_.d[1] = image.rows;
    semi_dims_.d[2] = image.cols;
    desc_dims_.d[1] = 256;
    desc_dims_.d[2] = image.rows / 8;
    desc_dims_.d[3] = image.cols / 8;

    auto *host_data_buffer = static_cast<float *>(buffers.getHostBuffer(super_point_config_.input_tensor_names[0]));
    for(int row = 0; row < image.rows; ++row){
        const uchar *ptr = image.ptr(row);
        int row_shift = row * image.cols;
        for (int col = 0; col < image.cols; ++col) {
            host_data_buffer[row_shift + col] = float(ptr[0]) / 255.0;
            ptr++;
        }
    }
    return true;
}

std::vector<size_t> SuperPoint::sort_indexes(std::vector<float> &data) {
  std::vector<size_t> indexes(data.size());
  iota(indexes.begin(), indexes.end(), 0);
  sort(indexes.begin(), indexes.end(), [&data](size_t i1, size_t i2) { return data[i1] > data[i2]; });
  return indexes;
}

void SuperPoint::detect_point(const float* heat_map, Eigen::Matrix<float, 259, Eigen::Dynamic>& features, 
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
    if(*(heat_map+i) < threshold) continue;

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
    for (int i = 0; i < top_k; ++i) {
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

int SuperPoint::clip(int val, int max) {
  if (val < 0) return 0;
  return std::min(val, max - 1);
}

void SuperPoint::extract_descriptors(const float *descriptors, Eigen::Matrix<float, 259, Eigen::Dynamic> &features, int h, int w, int s){
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

bool SuperPoint::keypoints_decoder(const float* scores, const float* descriptors, Eigen::Matrix<float, 259, Eigen::Dynamic> &features){
  detect_point(scores, features, resized_height, resized_width, super_point_config_.keypoint_threshold, 
      super_point_config_.remove_borders, super_point_config_.max_keypoints);
  extract_descriptors(descriptors, features, resized_height / 8, resized_width / 8, 8);

  features.block(1, 0, 1, features.cols()) = features.block(1, 0, 1, features.cols()) * w_scale;
  features.block(2, 0, 1, features.cols()) = features.block(2, 0, 1, features.cols()) * h_scale;
  return true;
}


bool SuperPoint::process_output(const BufferManager &buffers, Eigen::Matrix<float, 259, Eigen::Dynamic> &features) {
    keypoints_.clear();
    descriptors_.clear();
    auto *output_score = static_cast<float *>(buffers.getHostBuffer(super_point_config_.output_tensor_names[0]));
    auto *output_desc = static_cast<float *>(buffers.getHostBuffer(super_point_config_.output_tensor_names[1]));

    keypoints_decoder(output_score, output_desc, features);
    return true;
}

void SuperPoint::save_engine() {
    if (super_point_config_.engine_file.empty()) return;
    if (engine_ != nullptr) {
        nvinfer1::IHostMemory *data = engine_->serialize();
        std::ofstream file(super_point_config_.engine_file, std::ios::binary);
        if (!file) return;
        file.write(reinterpret_cast<const char *>(data->data()), data->size());
    }
}

bool SuperPoint::deserialize_engine() {
    std::ifstream file(super_point_config_.engine_file.c_str(), std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, std::ifstream::end);
        size_t size = file.tellg();
        file.seekg(0, std::ifstream::beg);
        char *model_stream = new char[size];
        file.read(model_stream, size);
        file.close();
	nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
        // if (runtime == nullptr) return false;
        if (runtime == nullptr) {
            delete[] model_stream;
            return false;
        }
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_stream, size));
        delete[] model_stream;
        if (engine_ == nullptr) return false;
        return true;
    }
    return false;
}