#ifndef DATASET_H_
#define DATASET_H_

#include <vector>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "imu.h"
#include "utils.h"

class Dataset{
public:
  Dataset(const std::string& dataroot, bool use_imu);
  void ReadImuData(const std::string& imu_file_path, ImuDataList& all_imu_data);
  size_t GetDatasetLength();
  bool GetData(size_t idx, cv::Mat& left_image, cv::Mat& right_image, ImuDataList& batch_imu_data, double& timestamp);

private:
  bool _use_imu;
  std::vector<std::string> _left_images;
  std::vector<std::string> _right_images;
  std::vector<ImuDataList> _imu_data;
  std::vector<double> _timestamps;
};

#endif // DATASET_H_