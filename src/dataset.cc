#include <fstream>
#include <math.h>

#include "dataset.h"
#include "ros_publisher.h"
#include "utils.h"

Dataset::Dataset(const std::string& dataroot){
  if(!PathExists(dataroot)){
    std::cout << "dataroot : " << dataroot << " doesn't exist" << std::endl;
    exit(0);
  }

  std::string left_image_dir = ConcatenateFolderAndFileName(dataroot, "cam0/data");
  std::string right_image_dir = ConcatenateFolderAndFileName(dataroot, "cam1/data");

  std::vector<std::string> image_names;
  GetFileNames(left_image_dir, image_names);
  if(image_names.size() < 1) return;
  std::sort(image_names.begin(), image_names.end()); 
  bool use_current_time = (image_names[0].size() < 18);
  for(std::string& image_name : image_names){
    _left_images.emplace_back(ConcatenateFolderAndFileName(left_image_dir, image_name));
    _right_images.emplace_back(ConcatenateFolderAndFileName(right_image_dir, image_name));
    if(!use_current_time){
      double timestamp = atof(image_name.substr(0, 10).c_str()) + atof(image_name.substr(10, 18).c_str()) / 1e9;
      _timestamps.emplace_back(timestamp);
    }
  }
}

size_t Dataset::GetDatasetLength(){
  return _left_images.size();
}

InputDataPtr Dataset::GetData(size_t idx){
  if(idx >= _left_images.size()) return nullptr;
  if(!FileExists(_left_images[idx]) || !FileExists(_right_images[idx])) return nullptr;

  InputDataPtr data = std::shared_ptr<InputData>(new InputData());
  data->index = idx;
  data->image_left = cv::imread(_left_images[idx], 0);
  data->image_right = cv::imread(_right_images[idx], 0);
  if(_timestamps.empty()){
    data->time = GetCurrentTime();
  }else{
    data->time = _timestamps[idx];
  }
  return data;
}