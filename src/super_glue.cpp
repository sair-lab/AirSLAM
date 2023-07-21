//
// Created by haoyuefan on 2021/9/22.
//

#include "super_glue.h"
#include <cfloat>
#include <utility>
#include <unordered_map>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace tensorrt_common;
using namespace tensorrt_log;
using namespace tensorrt_buffer;


SuperGlue::SuperGlue(const SuperGlueConfig &superglue_config) : superglue_config_(superglue_config), engine_(nullptr) {
    setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
}

bool SuperGlue::build() {
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
    profile->setDimensions(superglue_config_.input_tensor_names[0].c_str(), OptProfileSelector::kMIN, Dims3(1, 1, 2));
    profile->setDimensions(superglue_config_.input_tensor_names[0].c_str(), OptProfileSelector::kOPT, Dims3(1, 512, 2));
    profile->setDimensions(superglue_config_.input_tensor_names[0].c_str(), OptProfileSelector::kMAX,
                           Dims3(1, 1024, 2));
    profile->setDimensions(superglue_config_.input_tensor_names[1].c_str(), OptProfileSelector::kMIN, Dims2(1, 1));
    profile->setDimensions(superglue_config_.input_tensor_names[1].c_str(), OptProfileSelector::kOPT, Dims2(1, 512));
    profile->setDimensions(superglue_config_.input_tensor_names[1].c_str(), OptProfileSelector::kMAX, Dims2(1, 1024));
    profile->setDimensions(superglue_config_.input_tensor_names[2].c_str(), OptProfileSelector::kMIN, Dims3(1, 256, 1));
    profile->setDimensions(superglue_config_.input_tensor_names[2].c_str(), OptProfileSelector::kOPT,
                           Dims3(1, 256, 512));
    profile->setDimensions(superglue_config_.input_tensor_names[2].c_str(), OptProfileSelector::kMAX,
                           Dims3(1, 256, 1024));
    profile->setDimensions(superglue_config_.input_tensor_names[3].c_str(), OptProfileSelector::kMIN, Dims3(1, 1, 2));
    profile->setDimensions(superglue_config_.input_tensor_names[3].c_str(), OptProfileSelector::kOPT, Dims3(1, 512, 2));
    profile->setDimensions(superglue_config_.input_tensor_names[3].c_str(), OptProfileSelector::kMAX,
                           Dims3(1, 1024, 2));
    profile->setDimensions(superglue_config_.input_tensor_names[4].c_str(), OptProfileSelector::kMIN, Dims2(1, 1));
    profile->setDimensions(superglue_config_.input_tensor_names[4].c_str(), OptProfileSelector::kOPT, Dims2(1, 512));
    profile->setDimensions(superglue_config_.input_tensor_names[4].c_str(), OptProfileSelector::kMAX, Dims2(1, 1024));
    profile->setDimensions(superglue_config_.input_tensor_names[5].c_str(), OptProfileSelector::kMIN, Dims3(1, 256, 1));
    profile->setDimensions(superglue_config_.input_tensor_names[5].c_str(), OptProfileSelector::kOPT,
                           Dims3(1, 256, 512));
    profile->setDimensions(superglue_config_.input_tensor_names[5].c_str(), OptProfileSelector::kMAX,
                           Dims3(1, 256, 1024));
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

    ASSERT(network->getNbInputs() == 6);
    keypoints_0_dims_ = network->getInput(0)->getDimensions();
    scores_0_dims_ = network->getInput(1)->getDimensions();
    descriptors_0_dims_ = network->getInput(2)->getDimensions();
    keypoints_1_dims_ = network->getInput(3)->getDimensions();
    scores_1_dims_ = network->getInput(4)->getDimensions();
    descriptors_1_dims_ = network->getInput(5)->getDimensions();
    assert(keypoints_0_dims_.d[1] == -1);
    assert(scores_0_dims_.d[1] == -1);
    assert(descriptors_0_dims_.d[2] == -1);
    assert(keypoints_1_dims_.d[1] == -1);
    assert(scores_1_dims_.d[1] == -1);
    assert(descriptors_1_dims_.d[2] == -1);
    return true;
}

bool SuperGlue::construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder,
                                  TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                                  TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                                  TensorRTUniquePtr<nvonnxparser::IParser> &parser) const {
    auto parsed = parser->parseFromFile(superglue_config_.onnx_file.c_str(),
                                        static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }
    config->setMaxWorkspaceSize(512_MiB);
    config->setFlag(BuilderFlag::kFP16);
    enableDLA(builder.get(), config.get(), superglue_config_.dla_core);
    return true;
}

bool SuperGlue::infer(const Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
                      const Eigen::Matrix<double, 259, Eigen::Dynamic> &features1,
                      Eigen::VectorXi &indices0,
                      Eigen::VectorXi &indices1,
                      Eigen::VectorXd &mscores0,
                      Eigen::VectorXd &mscores1) {
    // Create RAII buffer manager object
    if (!context_) {
        context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_) {
            return false;
    }
    }

    assert(engine_->getNbBindings() == 7);

    const int keypoints_0_index = engine_->getBindingIndex(superglue_config_.input_tensor_names[0].c_str());
    const int scores_0_index = engine_->getBindingIndex(superglue_config_.input_tensor_names[1].c_str());
    const int descriptors_0_index = engine_->getBindingIndex(superglue_config_.input_tensor_names[2].c_str());
    const int keypoints_1_index = engine_->getBindingIndex(superglue_config_.input_tensor_names[3].c_str());
    const int scores_1_index = engine_->getBindingIndex(superglue_config_.input_tensor_names[4].c_str());
    const int descriptors_1_index = engine_->getBindingIndex(superglue_config_.input_tensor_names[5].c_str());
    const int output_score_index = engine_->getBindingIndex(superglue_config_.output_tensor_names[0].c_str());

    context_->setBindingDimensions(keypoints_0_index, Dims3(1, features0.cols(), 2));
    context_->setBindingDimensions(scores_0_index, Dims2(1, features0.cols()));
    context_->setBindingDimensions(descriptors_0_index, Dims3(1, 256, features0.cols()));
    context_->setBindingDimensions(keypoints_1_index, Dims3(1, features1.cols(), 2));
    context_->setBindingDimensions(scores_1_index, Dims2(1, features1.cols()));
    context_->setBindingDimensions(descriptors_1_index, Dims3(1, 256, features1.cols()));

    keypoints_0_dims_ = context_->getBindingDimensions(keypoints_0_index);
    scores_0_dims_ = context_->getBindingDimensions(scores_0_index);
    descriptors_0_dims_ = context_->getBindingDimensions(descriptors_0_index);
    keypoints_1_dims_ = context_->getBindingDimensions(keypoints_1_index);
    scores_1_dims_ = context_->getBindingDimensions(scores_1_index);
    descriptors_1_dims_ = context_->getBindingDimensions(descriptors_1_index);
    output_scores_dims_ = context_->getBindingDimensions(output_score_index);

    BufferManager buffers(engine_, 0, context_.get());

    ASSERT(superglue_config_.input_tensor_names.size() == 6);
    if (!process_input(buffers, features0, features1)) {
        return false;
    }

    buffers.copyInputToDevice();

    bool status = context_->executeV2(buffers.getDeviceBindings().data());
    if (!status) {
        return false;
    }
    buffers.copyOutputToHost();

    // Verify results
    if (!process_output(buffers, indices0, indices1, mscores0, mscores1)) {
        return false;
    }

    return true;
}

bool SuperGlue::process_input(const BufferManager &buffers,
                              const Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
                              const Eigen::Matrix<double, 259, Eigen::Dynamic> &features1) {
    auto *keypoints_0_buffer = static_cast<float *>(buffers.getHostBuffer(superglue_config_.input_tensor_names[0]));
    auto *scores_0_buffer = static_cast<float *>(buffers.getHostBuffer(superglue_config_.input_tensor_names[1]));
    auto *descriptors_0_buffer = static_cast<float *>(buffers.getHostBuffer(superglue_config_.input_tensor_names[2]));
    auto *keypoints_1_buffer = static_cast<float *>(buffers.getHostBuffer(superglue_config_.input_tensor_names[3]));
    auto *scores_1_buffer = static_cast<float *>(buffers.getHostBuffer(superglue_config_.input_tensor_names[4]));
    auto *descriptors_1_buffer = static_cast<float *>(buffers.getHostBuffer(superglue_config_.input_tensor_names[5]));

    for (int rows0 = 0; rows0 < 1; ++rows0) {
        for (int cols0 = 0; cols0 < features0.cols(); ++cols0) {
            scores_0_buffer[rows0 * features0.cols() + cols0] = features0(rows0, cols0);
        }
    }

    for (int colk0 = 0; colk0 < features0.cols(); ++colk0) {
        for (int rowk0 = 1; rowk0 < 3; ++rowk0) {
            keypoints_0_buffer[colk0 * 2 + (rowk0 - 1)] = features0(rowk0, colk0);
        }
    }

    for (int rowd0 = 3; rowd0 < features0.rows(); ++rowd0) {
        for (int cold0 = 0; cold0 < features0.cols(); ++cold0) {
            descriptors_0_buffer[(rowd0 - 3) * features0.cols() + cold0] = features0(rowd0, cold0);
        }
    }

    for (int rows1 = 0; rows1 < 1; ++rows1) {
        for (int cols1 = 0; cols1 < features1.cols(); ++cols1) {
            scores_1_buffer[rows1 * features1.cols() + cols1] = features1(rows1, cols1);
        }
    }

    for (int colk1 = 0; colk1 < features1.cols(); ++colk1) {
        for (int rowk1 = 1; rowk1 < 3; ++rowk1) {
            keypoints_1_buffer[colk1 * 2 + (rowk1 - 1)] = features1(rowk1, colk1);
        }
    }

    for (int rowd1 = 3; rowd1 < features1.rows(); ++rowd1) {
        for (int cold1 = 0; cold1 < features1.cols(); ++cold1) {
            descriptors_1_buffer[(rowd1 - 3) * features1.cols() + cold1] = features1(rowd1, cold1);
        }
    }

    return true;
}

void where_negative_one(const int *flag_data, const int *data, int size, std::vector<int> &indices) {
    for (int i = 0; i < size; ++i) {
        if (flag_data[i] == 1) {
            indices.push_back(data[i]);
        } else {
            indices.push_back(-1);
        }
    }
}

void max_matrix(const float *data, int *indices, float *values, int h, int w, int dim) {
    if (dim == 2) {
        for (int i = 0; i < h - 1; ++i) {
            float max_value = -FLT_MAX;
            int max_indices = 0;
            for (int j = 0; j < w - 1; ++j) {
                if (max_value < data[i * w + j]) {
                    max_value = data[i * w + j];
                    max_indices = j;
                }
            }
            values[i] = max_value;
            indices[i] = max_indices;
        }
    } else if (dim == 1) {
        for (int i = 0; i < w - 1; ++i) {
            float max_value = -FLT_MAX;
            int max_indices = 0;
            for (int j = 0; j < h - 1; ++j) {
                if (max_value < data[j * w + i]) {
                    max_value = data[j * w + i];
                    max_indices = j;
                }
            }
            values[i] = max_value;
            indices[i] = max_indices;
        }
    }
}

void equal_gather(const int *indices0, const int *indices1, int *mutual, int size) {
    for (int i = 0; i < size; ++i) {
        if (indices0[indices1[i]] == i) {
            mutual[i] = 1;
        } else {
            mutual[i] = 0;
        }
    }
}

void where_exp(const int *flag_data, float *data, std::vector<double> &mscores0, int size) {
    for (int i = 0; i < size; ++i) {
        if (flag_data[i] == 1) {
            mscores0.push_back(std::exp(data[i]));
        } else {
            mscores0.push_back(0);
        }
    }
}

void where_gather(const int *flag_data, int *indices, std::vector<double> &mscores0, std::vector<double> &mscores1,
                  int size) {
    for (int i = 0; i < size; ++i) {
        if (flag_data[i] == 1) {
            mscores1.push_back(mscores0[indices[i]]);
        } else {
            mscores1.push_back(0);
        }
    }
}

void and_threshold(const int *mutual0, int *valid0, const std::vector<double> &mscores0, double threhold) {
    for (int i = 0; i < mscores0.size(); ++i) {
        if (mutual0[i] == 1 && mscores0[i] > threhold) {
            valid0[i] = 1;
        } else {
            valid0[i] = 0;
        }
    }
}

void and_gather(const int *mutual1, const int *valid0, const int *indices1, int *valid1, int size) {
    for (int i = 0; i < size; ++i) {
        if (mutual1[i] == 1 && valid0[indices1[i]] == 1) {
            valid1[i] = 1;
        } else {
            valid1[i] = 0;
        }
    }
}

void decode(float *scores, int h, int w, std::vector<int> &indices0, std::vector<int> &indices1,
            std::vector<double> &mscores0, std::vector<double> &mscores1) {
    auto *max_indices0 = new int[h - 1];
    auto *max_indices1 = new int[w - 1];
    auto *max_values0 = new float[h - 1];
    auto *max_values1 = new float[w - 1];
    max_matrix(scores, max_indices0, max_values0, h, w, 2);
    max_matrix(scores, max_indices1, max_values1, h, w, 1);
    auto *mutual0 = new int[h - 1];
    auto *mutual1 = new int[w - 1];
    equal_gather(max_indices1, max_indices0, mutual0, h - 1);
    equal_gather(max_indices0, max_indices1, mutual1, w - 1);
    where_exp(mutual0, max_values0, mscores0, h - 1);
    where_gather(mutual1, max_indices1, mscores0, mscores1, w - 1);
    auto *valid0 = new int[h - 1];
    auto *valid1 = new int[w - 1];
    and_threshold(mutual0, valid0, mscores0, 0.2);
    and_gather(mutual1, valid0, max_indices1, valid1, w - 1);
    where_negative_one(valid0, max_indices0, h - 1, indices0);
    where_negative_one(valid1, max_indices1, w - 1, indices1);
    delete[] max_indices0;
    delete[] max_indices1;
    delete[] max_values0;
    delete[] max_values1;
    delete[] mutual0;
    delete[] mutual1;
    delete[] valid0;
    delete[] valid1;
}

void log_sinkhorn_iterations(float *couplings, float *Z, int m, int n,
                             float *log_mu, float *log_nu, int iters) {
  auto *u = new float[m]();
  auto *v = new float[n]();
  for (int k = 0; k < iters; ++k) {
    for (int ki = 0; ki < m; ++ki) {
      float nu_expsum = 0.0;
      for (int kn = 0; kn < n; ++kn) {
        nu_expsum += std::exp(couplings[ki * n + kn] + v[kn]);
      }
      u[ki] = log_mu[ki] - std::log(nu_expsum);
    }
    for (int kj = 0; kj < n; ++kj) {
      float nu_expsum = 0.0;
      for (int km = 0; km < m; ++km) {
        nu_expsum += std::exp(couplings[km * n + kj] + u[km]);
      }
      v[kj] = log_nu[kj] - std::log(nu_expsum);
    }
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      Z[i * n + j] = couplings[i * n + j] + u[i] + v[j];
    }
  }
  delete[] u;
  delete[] v;
}

void log_optimal_transport(float *scores, float *Z, int m, int n,
                           float alpha = 2.3457, int iters = 100) {
  auto *couplings = new float[(m + 1) * (n + 1)];
  for (int i = 0; i < m + 1; ++i) {
    for (int j = 0; j < n + 1; ++j) {
      if (i == m || j == n) {
        couplings[i * (n + 1) + j] = alpha;
      } else {
        couplings[i * (n + 1) + j] = scores[i * n + j];
      }
    }
  }

  float norm = -std::log(m + n);

  auto *log_mu = new float[m + 1];
  auto *log_nu = new float[n + 1];
  for (int ii = 0; ii < m; ++ii) {
    log_mu[ii] = norm;
  }
  log_mu[m] = std::log(n) + norm;

  for (int jj = 0; jj < n; ++jj) {
    log_nu[jj] = norm;
  }
  log_nu[n] = std::log(m) + norm;

  log_sinkhorn_iterations(couplings, Z, m + 1, n + 1, log_mu, log_nu, iters);
  for (int ii = 0; ii < m + 1; ++ii) {
    for (int jj = 0; jj < n + 1; ++jj) {
      Z[ii * (n + 1) + jj] = Z[ii * (n + 1) + jj] - norm;
    }
  }
  delete[] couplings;
  delete[] log_mu;
  delete[] log_nu;
}

bool SuperGlue::process_output(const BufferManager &buffers,
                               Eigen::VectorXi &indices0,
                               Eigen::VectorXi &indices1,
                               Eigen::VectorXd &mscores0,
                               Eigen::VectorXd &mscores1) {
    indices0_.clear();
    indices1_.clear();
    mscores0_.clear();
    mscores1_.clear();
    auto *output_score = static_cast<float *>(buffers.getHostBuffer(superglue_config_.output_tensor_names[0]));
    int scores_map_h = output_scores_dims_.d[1];
    int scores_map_w = output_scores_dims_.d[2];
    //auto *scores = new float[(scores_map_h + 1) * (scores_map_w + 1)];
    //log_optimal_transport(output_score, scores, scores_map_h, scores_map_w);
    //delete []scores;
    //scores_map_h = scores_map_h + 1;
    //scores_map_w = scores_map_w + 1;
    decode(output_score, scores_map_h, scores_map_w, indices0_, indices1_, mscores0_, mscores1_);
    indices0.resize(indices0_.size());
    indices1.resize(indices1_.size());
    mscores0.resize(mscores0_.size());
    mscores1.resize(mscores1_.size());
    for (int i0 = 0; i0 < indices0_.size(); ++i0) {
        indices0(i0) = indices0_[i0];
    }
    for (int i1 = 0; i1 < indices1_.size(); ++i1) {
        indices1(i1) = indices1_[i1];
    }
    for (int j0 = 0; j0 < mscores0_.size(); ++j0) {
        mscores0(j0) = mscores0_[j0];
    }
    for (int j1 = 0; j1 < mscores1_.size(); ++j1) {
        mscores1(j1) = mscores1_[j1];
    }
    return true;
}

void SuperGlue::save_engine() {
    if (superglue_config_.engine_file.empty()) return;
    if (engine_ != nullptr) {
        nvinfer1::IHostMemory *data = engine_->serialize();
        std::ofstream file(superglue_config_.engine_file, std::ios::binary);;
        if (!file) return;
        file.write(reinterpret_cast<const char *>(data->data()), data->size());
    }
}

bool SuperGlue::deserialize_engine() {
    std::ifstream file(superglue_config_.engine_file, std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, std::ifstream::end);
        size_t size = file.tellg();
        file.seekg(0, std::ifstream::beg);
        char *model_stream = new char[size];
        file.read(model_stream, size);
        file.close();
        IRuntime *runtime = createInferRuntime(gLogger);
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


