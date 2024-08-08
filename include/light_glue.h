//
// Created by haoyuefan on 2021/9/22.
//

#ifndef LIGHT_GLUE_H_
#define LIGHT_GLUE_H_

#include <string>
#include <memory>
#include <NvInfer.h>
#include <Eigen/Core>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>


#include "3rdparty/tensorrtbuffer/include/buffers.h"
#include "read_configs.h"

using tensorrt_buffer::TensorRTUniquePtr;

class SuperPointLightGlue {
public:
    SuperPointLightGlue() {};
    explicit SuperPointLightGlue(const PointMatcherConfig &lightglue_config);

    bool build();

    bool infer(const Eigen::Matrix<float, 258, Eigen::Dynamic> &features0,
               const Eigen::Matrix<float, 258, Eigen::Dynamic> &features1,
               Eigen::Matrix<int, Eigen::Dynamic, 2> &matches_index,
               Eigen::Matrix<float, Eigen::Dynamic, 1> &matches_score);

    void save_engine();

    bool deserialize_engine();

private:
    PointMatcherConfig lightglue_config_;
    std::vector<int> indices0_;
    std::vector<int> indices1_;
    std::vector<float> mscores0_;
    std::vector<float> mscores1_;

    nvinfer1::Dims keypoints_0_dims_{};
    nvinfer1::Dims descriptors_0_dims_{};
    nvinfer1::Dims keypoints_1_dims_{};
    nvinfer1::Dims descriptors_1_dims_{};

    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;

    bool construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder,
                           TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                           TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                           TensorRTUniquePtr<nvonnxparser::IParser> &parser) const;

    bool process_input(const tensorrt_buffer::BufferManager &buffers,
                       const Eigen::Matrix<float, 258, Eigen::Dynamic> &features0,
                       const Eigen::Matrix<float, 258, Eigen::Dynamic> &features1);

    bool process_output(const tensorrt_buffer::BufferManager &buffers, Eigen::Matrix<int, Eigen::Dynamic, 2> &matches_index, Eigen::Matrix<float, Eigen::Dynamic, 1> &matches_score);

};

typedef std::shared_ptr<SuperPointLightGlue> SuperPointLightGluePtr;

#endif //LIGHT_GLUE_H_
