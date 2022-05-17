#ifndef DATASET_H_
#define DATASET_H_

#include <vector>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "utils.h"

class Dataset{
public:
  Dataset(const std::string& dataroot);
  size_t GetDatasetLength();
  bool GetData(size_t idx, cv::Mat& left_image, cv::Mat& right_image, double& timestamp);

private:
  std::vector<std::string> _left_images;
  std::vector<std::string> _right_images;
  std::vector<double> _timestamps;
};

#endif // DATASET_H_