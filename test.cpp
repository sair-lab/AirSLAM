#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <ros/ros.h>

#include "read_configs.h"
#include "dataset.h"
#include "map_builder.h"
#include "line_processor.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "air_slam");
  std::string config_file = argv[1];
  Configs configs(config_file);
  MapBuilder map_builder(configs);
  Dataset dataset(configs.dataroot);
  size_t dataset_length = dataset.GetDatasetLength();

  CameraPtr camera = std::shared_ptr<Camera>(new Camera(configs.camera_config_path));
  SuperPointPtr superpoint = std::shared_ptr<SuperPoint>(new SuperPoint(configs.superpoint_config));
  if (!_superpoint->build()){
    std::cout << "Error in SuperPoint building" << std::endl;
    exit(0);
  }
  PointMatchingPtr point_matching = std::shared_ptr<PointMatching>(new PointMatching(configs.superglue_config));
  LineDetectorPtr line_detector = std::shared_ptr<LineDetector>(new LineDetector(configs.line_detector_config));
  RosPublisherPtr ros_publisher = std::shared_ptr<RosPublisher>(new RosPublisher(configs.ros_publisher_config));
  MapPtr map = std::shared_ptr<Map>(new Map(camera, ros_publisher));


  for(size_t i = 0; i < dataset_length && ros::ok(); ++i){
    std::cout << "i ===== " << i << std::endl;
    auto before_infer = std::chrono::steady_clock::now();
    cv::Mat left_image, right_image;
    double timestamp;
    if(!dataset.GetData(i, left_image, right_image, timestamp)) continue;

    std::vector<Eigen::Vector4d> lines;
    line_detector->LineExtractor(left_image, lines);
    std::cout << "line num = " << lines.size() << std::endl;
  }
  ros::shutdown();

  return 0;
}