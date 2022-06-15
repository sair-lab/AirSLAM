#include <fstream>
#include <math.h>

#include "dataset.h"
#include "utils.h"

Dataset::Dataset(const std::string& dataroot){
  if(!PathExists(dataroot)){
    std::cout << "dataroot : " << dataroot << " doesn't exist" << std::endl;
    exit(0);
  }

  std::string left_image_dir = ConcatenateFolderAndFileName(dataroot, "cam0/data");
  std::string right_image_dir = ConcatenateFolderAndFileName(dataroot, "cam1/data");

  std::vector<std::string> image_names;
  std::cout << "left_image_dir = " << left_image_dir << std::endl;
  GetFileNames(left_image_dir, image_names);
  std::sort (image_names.begin(), image_names.end()); 
  for(std::string& image_name : image_names){
    _left_images.emplace_back(ConcatenateFolderAndFileName(left_image_dir, image_name));
    _right_images.emplace_back(ConcatenateFolderAndFileName(right_image_dir, image_name));
    double timestamp = (atof(image_name.substr(0, 10).c_str()) + atof(image_name.substr(10, 18).c_str())) / 1e9;
    _timestamps.emplace_back(timestamp);
  }
}

size_t Dataset::GetDatasetLength(){
  return _left_images.size();
}

bool Dataset::GetData(size_t idx, cv::Mat& left_image, cv::Mat& right_image, double& timestamp){
  if(idx >= _left_images.size()) return false;
  std::cout << "left_image = " << _left_images[idx] << std::endl;
  std::cout << "right_image = " << _right_images[idx] << std::endl;
  left_image = cv::imread(_left_images[idx], 0);
  right_image = cv::imread(_right_images[idx], 0);
  timestamp = _timestamps[idx];
  return true;
}