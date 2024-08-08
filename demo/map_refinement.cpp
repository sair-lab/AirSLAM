#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <ros/ros.h>

#include "utils.h"
#include "read_configs.h"
#include "map.h"
#include "map_refiner.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "air_slam");
  ros::NodeHandle nh;

  int breakpoint;
  ros::param::get("~breakpoint", breakpoint);

  std::string config_path, model_dir;
  ros::param::get("~config_path", config_path);
  ros::param::get("~model_dir", model_dir);
  MapRefinementConfigs configs(config_path, model_dir);
  MapRefiner map_refiner(configs, nh);

  std::string map_root;
  ros::param::get("~map_root", map_root);
  std::cout << "Loading map and vocabulary..." << std::endl;
  map_refiner.LoadMap(map_root);

  std::string voc_path;
  ros::param::get("~voc_path", voc_path);
  map_refiner.LoadVocabulary(voc_path);
  std::cout << "Done." << std::endl;

  map_refiner.Wait(breakpoint);

  std::cout << "Building covisibility graph..." << std::endl;
  map_refiner.UpdateCovisibilityGraph();
  std::cout << "Done." << std::endl;

  std::cout << "Loop detection..." << std::endl;
  int loop_num = map_refiner.LoopDetection();
  std::cout << "Done, " << loop_num << " loop pairs are found." << std::endl;

  std::cout << "Optimizing pose graph..." << std::endl;
  map_refiner.PoseGraphRefinement();
  std::cout << "Done." << std::endl;

  map_refiner.Wait(breakpoint);

  std::cout << "Merging mappoints..." << std::endl;
  map_refiner.MergeMap();
  std::cout << "Done." << std::endl;

  map_refiner.Wait(breakpoint);

  std::cout << "Optimizing global map..." << std::endl;
  map_refiner.GlobalMapOptimization();
  map_refiner.UpdateCovisibilityGraph();
  std::cout << "Done." << std::endl;

  std::cout << "Build junction database..." << std::endl;
  map_refiner.BuildJunctionDatabase();
  std::cout << "Done." << std::endl;

  std::string trajectory_global_ba_path = ConcatenateFolderAndFileName(map_root, "trajectory_v1.txt");
  map_refiner.SaveTrajectory(trajectory_global_ba_path);

  map_refiner.Wait(breakpoint);

  std::cout << "Saving final map..." << std::endl;
  map_refiner.SaveFinalMap(map_root);
  std::cout << "Done." << std::endl;

  exit(0);
  map_refiner.StopVisualization();
  ros::shutdown();

  return 0;
}
