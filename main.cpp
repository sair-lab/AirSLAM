#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <ros/ros.h>

#include "read_configs.h"
#include "dataset.h"
#include "map_builder.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "air_slam");
  std::string config_file = argv[1];
  Configs configs(config_file);
  MapBuilder map_builder(configs);
  Dataset dataset(configs.dataroot);
  size_t dataset_length = dataset.GetDatasetLength();
  for(size_t i = 0; i < dataset_length && ros::ok(); ++i){
    std::cout << "i ===== " << i << std::endl;
    auto before_infer = std::chrono::steady_clock::now();
    cv::Mat left_image, right_image;
    double timestamp;
    if(!dataset.GetData(i, left_image, right_image, timestamp)) continue;
    map_builder.AddInput(i, left_image, right_image, timestamp);

    auto after_infer = std::chrono::steady_clock::now();
    auto cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(after_infer - before_infer).count();
    std::cout << "One Frame Processinh Time: " << cost_time << " ms." << std::endl;
  }
  map_builder.SaveTrajectory();
  ros::shutdown();

  return 0;
}
