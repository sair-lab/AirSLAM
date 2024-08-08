#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <ros/ros.h>
#include <thread>

#include "read_configs.h"
#include "dataset.h"
#include "map_user.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "air_slam");
  ros::NodeHandle nh;

  std::string config_path, model_dir, map_root, voc_path, traj_path;
  ros::param::get("~config_path", config_path);
  ros::param::get("~model_dir", model_dir);
  ros::param::get("~map_root", map_root);
  ros::param::get("~voc_path", voc_path);
  ros::param::get("~traj_path", traj_path);

  RelocalizationConfigs configs(config_path, model_dir);
  ros::param::get("~dataroot", configs.dataroot);
  ros::param::get("~camera_config_path", configs.camera_config_path);

  MapUser map_user(configs, nh);
  map_user.LoadMap(map_root);
  map_user.LoadVocabulary(voc_path);

  std::vector<std::string> image_names;
  GetFileNames(configs.dataroot, image_names);
  std::sort(image_names.begin(), image_names.end());
  size_t dataset_length = image_names.size();

  std::vector<std::pair<std::string, Eigen::Matrix4d>> trajectory;
  Eigen::Matrix4d base_frame_pose = map_user.GetBaseFramePose();
  double base_frame_time = map_user.GetBaseFrameTimestamp();
  trajectory.emplace_back(std::make_pair(("base "+DoubleTimeToString(base_frame_time)), base_frame_pose));

  int success_num = 0;
  for(size_t i = 0; i < dataset_length && ros::ok(); ++i){
    std::cout << "i ====== " << i << std::endl;

    cv::Mat image = cv::imread(ConcatenateFolderAndFileName(configs.dataroot, image_names[i]), 0);
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    size_t pos = image_names[i].find_last_of('.');
    std::string image_idx = "fail " + image_names[i].substr(0, pos);

    auto before_infer = std::chrono::high_resolution_clock::now();
    if(map_user.Relocalization(image, pose)){
      image_idx = "success " + image_names[i].substr(0, pos);
      success_num++;
    }
    auto after_infer = std::chrono::high_resolution_clock::now();
    auto cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(after_infer - before_infer).count();
    std::cout << "One Frame Processinh Time: " << cost_time << " ms." << std::endl;

    trajectory.emplace_back(std::make_pair(image_idx, pose));
  }

  SaveTumTrajectoryToFile(traj_path, trajectory);
  std::cout << "sum_num = " << dataset_length << ", success_num = " << success_num << ", recall = " << (float)success_num / dataset_length << std::endl;

  map_user.StopVisualization();
  ros::shutdown();

  return 0;
}
