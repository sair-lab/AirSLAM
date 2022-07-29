#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <ros/ros.h>

#include "super_point.h"
#include "super_glue.h"
#include "read_configs.h"
#include "dataset.h"
#include "camera.h"
#include "frame.h"
#include "point_matching.h"
#include "line_processor.h"
#include "map.h"
#include "ros_publisher.h"
#include "g2o_optimization/types.h"
#include "debug.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "air_slam");
  std::string config_file = argv[1];
  Configs configs(config_file);
  Dataset dataset(configs.dataroot);
  size_t dataset_length = dataset.GetDatasetLength();

  CameraPtr camera = std::shared_ptr<Camera>(new Camera(configs.camera_config_path));
  SuperPointPtr superpoint = std::shared_ptr<SuperPoint>(new SuperPoint(configs.superpoint_config));
  if (!superpoint->build()){
    std::cout << "Error in SuperPoint building" << std::endl;
    exit(0);
  }
  PointMatchingPtr point_matching = std::shared_ptr<PointMatching>(new PointMatching(configs.superglue_config));
  LineDetectorPtr line_detector = std::shared_ptr<LineDetector>(new LineDetector(configs.line_detector_config));
  RosPublisherPtr ros_publisher = std::shared_ptr<RosPublisher>(new RosPublisher(configs.ros_publisher_config));
  MapPtr map = std::shared_ptr<Map>(new Map(camera, ros_publisher));

  for(size_t i = 0; i < dataset_length && ros::ok(); ++i){
    auto before_infer = std::chrono::steady_clock::now();
    // 1. get data
    cv::Mat left_image, right_image;
    double timestamp;
    if(!dataset.GetData(i, left_image, right_image, timestamp)) continue;

    // 2. undistort image
    cv::Mat image_left_rect, image_right_rect;
    camera->UndistortImage(left_image, right_image, image_left_rect, image_right_rect);

    // // 3. extract point feature
    // Eigen::Matrix<double, 259, Eigen::Dynamic> features_left, features_right;
    // if(!superpoint->infer(image_left_rect, features_left)){
    //   std::cout << "Failed when extracting features of left image !" << std::endl;
    //   exit(0);
    // }
    // if(!superpoint->infer(image_right_rect, features_right)){
    //   std::cout << "Failed when extracting features of right image !" << std::endl;
    //   exit(0);
    // }

    // 4. extract line
    std::vector<Eigen::Vector4d> lines;
    line_detector->LineExtractor(image_left_rect, lines);
    std::cout << "line num = " << lines.size() << std::endl;

    SaveLineDetectionResult(image_left_rect, lines, configs.saving_dir, std::to_string(i));

    // // 5. assign points to lines
    // std::vector<std::set<int>> points_on_lines;
    // Eigen::Matrix2Xd points = features_left.middleRows(1, 2);
    // AssignPointsToLines(lines, points, points_on_lines);

    // auto after_infer = std::chrono::steady_clock::now();
    // auto cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(after_infer - before_infer).count();
    // std::cout << "One Frame Processinh Time: " << cost_time << " ms." << std::endl;

    // SavePointLineRelation(image_left_rect, lines, points, points_on_lines, configs.saving_dir, std::to_string(i));
  }
  // ros::shutdown();

  return 0;
}
