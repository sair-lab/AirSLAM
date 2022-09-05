//
// Created by haoyuefan on 2021/9/22.
//
#include "super_point.h"
#include <utility>
#include <unordered_map>
#include <opencv2/opencv.hpp>

using namespace tensorrt_common;
using namespace tensorrt_log;
using namespace tensorrt_buffer;

SuperPoint::SuperPoint(const SuperPointConfig &super_point_config)
        : super_point_config_(super_point_config), engine_(
        nullptr) {
    setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
}

bool SuperPoint::build() {
    if(deserialize_engine()){
        return true;
    }
    auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder) {
        return false;
    }
    const auto explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
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
                           OptProfileSelector::kMIN, Dims4(1, 1, 100, 100));
    profile->setDimensions(super_point_config_.input_tensor_names[0].c_str(),
                           OptProfileSelector::kOPT, Dims4(1, 1, 500, 500));
    profile->setDimensions(super_point_config_.input_tensor_names[0].c_str(),
                           OptProfileSelector::kMAX, Dims4(1, 1, 1500, 1500));
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
    TensorRTUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }
    TensorRTUniquePtr<IRuntime> runtime{createInferRuntime(gLogger.getTRTLogger())};
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
    config->setMaxWorkspaceSize(512_MiB);
    config->setFlag(BuilderFlag::kFP16);
    enableDLA(builder.get(), config.get(), super_point_config_.dla_core);
    return true;
}


bool SuperPoint::infer(const cv::Mat &image, Eigen::Matrix<double, 259, Eigen::Dynamic> &features) {
    if (!context_) {
        context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_) {
            return false;
        }
    }
    
    assert(engine_->getNbBindings() == 3);

    const int input_index = engine_->getBindingIndex(super_point_config_.input_tensor_names[0].c_str());

    context_->setBindingDimensions(input_index, Dims4(1, 1, image.rows, image.cols));

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
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            host_data_buffer[row * image.cols + col] = float(image.at<unsigned char>(row, col)) / 255.0;
        }
    }
    return true;
}

void SuperPoint::find_high_score_index(std::vector<float> &scores, std::vector<std::vector<int>> &keypoints,
                                       int h, int w, double threshold) {
    std::vector<float> new_scores;
    for (int i = 0; i < scores.size(); ++i) {
        if (scores[i] > threshold) {
            std::vector<int> location = {int(i / w), i % w};
            keypoints.emplace_back(location);
            new_scores.push_back(scores[i]);
        }
    }
    scores.swap(new_scores);
}


void SuperPoint::remove_borders(std::vector<std::vector<int>> &keypoints, std::vector<float> &scores, int border,
                                int height,
                                int width) {
    std::vector<std::vector<int>> keypoints_selected;
    std::vector<float> scores_selected;
    for (int i = 0; i < keypoints.size(); ++i) {
        bool flag_h = (keypoints[i][0] >= border) && (keypoints[i][0] < (height - border));
        bool flag_w = (keypoints[i][1] >= border) && (keypoints[i][1] < (width - border));
        if (flag_h && flag_w) {
            keypoints_selected.push_back(std::vector<int>{keypoints[i][1], keypoints[i][0]});
            scores_selected.push_back(scores[i]);
        }
    }
    keypoints.swap(keypoints_selected);
    scores.swap(scores_selected);
}

std::vector<size_t> SuperPoint::sort_indexes(std::vector<float> &data) {
    std::vector<size_t> indexes(data.size());
    iota(indexes.begin(), indexes.end(), 0);
    sort(indexes.begin(), indexes.end(), [&data](size_t i1, size_t i2) { return data[i1] > data[i2]; });
    return indexes;
}

void SuperPoint::top_k_keypoints(std::vector<std::vector<int>> &keypoints, std::vector<float> &scores, int k) {
    if (k < keypoints.size() && k != -1) {
        std::vector<std::vector<int>> keypoints_top_k;
        std::vector<float> scores_top_k;
        std::vector<size_t> indexes = sort_indexes(scores);
        for (int i = 0; i < k; ++i) {
            keypoints_top_k.push_back(keypoints[indexes[i]]);
            scores_top_k.push_back(scores[indexes[i]]);
        }
        keypoints.swap(keypoints_top_k);
        scores.swap(scores_top_k);
    }
}

void
normalize_keypoints(const std::vector<std::vector<int>> &keypoints, std::vector<std::vector<double>> &keypoints_norm,
                    int h, int w, int s) {
    for (auto &keypoint : keypoints) {
        std::vector<double> kp = {keypoint[0] - s / 2 + 0.5, keypoint[1] - s / 2 + 0.5};
        kp[0] = kp[0] / (w * s - s / 2 - 0.5);
        kp[1] = kp[1] / (h * s - s / 2 - 0.5);
        kp[0] = kp[0] * 2 - 1;
        kp[1] = kp[1] * 2 - 1;
        keypoints_norm.push_back(kp);
    }
}

int clip(int val, int max) {
    if (val < 0) return 0;
    return std::min(val, max - 1);
}

void grid_sample(const float *input, std::vector<std::vector<double>> &grid,
                 std::vector<std::vector<double>> &output, int dim, int h, int w) {
    // descriptors 1x256x60x106
    // keypoints 1x1xnumberx2
    // out 1x256x1xnumber
    for (auto &g : grid) {
        double ix = ((g[0] + 1) / 2) * (w - 1);
        double iy = ((g[1] + 1) / 2) * (h - 1);

        int ix_nw = clip(std::floor(ix), w);
        int iy_nw = clip(std::floor(iy), h);

        int ix_ne = clip(ix_nw + 1, w);
        int iy_ne = clip(iy_nw, h);

        int ix_sw = clip(ix_nw, w);
        int iy_sw = clip(iy_nw + 1, h);

        int ix_se = clip(ix_nw + 1, w);
        int iy_se = clip(iy_nw + 1, h);

        double nw = (ix_se - ix) * (iy_se - iy);
        double ne = (ix - ix_sw) * (iy_sw - iy);
        double sw = (ix_ne - ix) * (iy - iy_ne);
        double se = (ix - ix_nw) * (iy - iy_nw);

        std::vector<double> descriptor;
        for (int i = 0; i < dim; ++i) {
            // 256x60x106 whd
            // x * Height * Depth + y * Depth + z
            float nw_val = input[i * h * w + iy_nw * w + ix_nw];
            float ne_val = input[i * h * w + iy_ne * w + ix_ne];
            float sw_val = input[i * h * w + iy_sw * w + ix_sw];
            float se_val = input[i * h * w + iy_se * w + ix_se];
            descriptor.push_back(nw_val * nw + ne_val * ne + sw_val * sw + se_val * se);
        }
        output.push_back(descriptor);
    }
}

template<typename Iter_T>
double vector_normalize(Iter_T first, Iter_T last) {
    return sqrt(inner_product(first, last, first, 0.0));
}

void normalize_descriptors(std::vector<std::vector<double>> &dest_descriptors) {
    for (auto &descriptor : dest_descriptors) {
        double norm_inv = 1.0 / vector_normalize(descriptor.begin(), descriptor.end());
        std::transform(descriptor.begin(), descriptor.end(), descriptor.begin(),
                       std::bind1st(std::multiplies<double>(), norm_inv));
    }
}

void SuperPoint::sample_descriptors(std::vector<std::vector<int>> &keypoints, float *descriptors,
                                    std::vector<std::vector<double>> &dest_descriptors, int dim, int h, int w, int s) {
    std::vector<std::vector<double>> keypoints_norm;
    normalize_keypoints(keypoints, keypoints_norm, h, w, s);
    grid_sample(descriptors, keypoints_norm, dest_descriptors, dim, h, w);
    normalize_descriptors(dest_descriptors);
}

bool SuperPoint::process_output(const BufferManager &buffers, Eigen::Matrix<double, 259, Eigen::Dynamic> &features) {
    keypoints_.clear();
    descriptors_.clear();
    auto *output_score = static_cast<float *>(buffers.getHostBuffer(super_point_config_.output_tensor_names[0]));
    auto *output_desc = static_cast<float *>(buffers.getHostBuffer(super_point_config_.output_tensor_names[1]));
    int semi_feature_map_h = semi_dims_.d[1];
    int semi_feature_map_w = semi_dims_.d[2];
    std::vector<float> scores_vec(output_score, output_score + semi_feature_map_h * semi_feature_map_w);
    find_high_score_index(scores_vec, keypoints_, semi_feature_map_h, semi_feature_map_w,
                          super_point_config_.keypoint_threshold);
    remove_borders(keypoints_, scores_vec, super_point_config_.remove_borders, semi_feature_map_h, semi_feature_map_w);
    top_k_keypoints(keypoints_, scores_vec, super_point_config_.max_keypoints);
    // std::cout << "super point number is " << std::to_string(scores_vec.size()) << std::endl;
    features.resize(259, scores_vec.size());
    int desc_feature_dim = desc_dims_.d[1];
    int desc_feature_map_h = desc_dims_.d[2];
    int desc_feature_map_w = desc_dims_.d[3];
    sample_descriptors(keypoints_, output_desc, descriptors_, desc_feature_dim, desc_feature_map_h, desc_feature_map_w);
    
    for (int i = 0; i < scores_vec.size(); i++){
        features(0, i) = scores_vec[i];
    }

    for (int i = 1; i < 3; ++i) {
        for (int j = 0; j < keypoints_.size(); ++j) {
            features(i, j) = keypoints_[j][i-1];
        }
    }
    for (int m = 3; m < 259; ++m) {
        for (int n = 0; n < descriptors_.size(); ++n) {
            features(m, n) = descriptors_[n][m-3];
        }
    }
    return true;
}

void SuperPoint::visualization(const std::string &image_name, const cv::Mat &image) {
    cv::Mat image_display;
    if(image.channels() == 1)
        cv::cvtColor(image, image_display, cv::COLOR_GRAY2BGR);
    else
        image_display = image.clone();
    for (auto &keypoint : keypoints_) {
        cv::circle(image_display, cv::Point(int(keypoint[0]), int(keypoint[1])), 1, cv::Scalar(255, 0, 0), -1, 16);
    }
    cv::imwrite(image_name + ".jpg", image_display);
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
        IRuntime *runtime = createInferRuntime(gLogger);
        if (runtime == nullptr) return false;
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_stream, size));
        if (engine_ == nullptr) return false;
        return true;
    }
    return false;
}

