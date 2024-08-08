#include <fstream>
#include <math.h>

#include "dataset.h"
#include "ros_publisher.h"
#include "utils.h"
#include "imu.h"

Dataset::Dataset(const std::string& dataroot, const bool use_imu): _use_imu(use_imu){
  if(!PathExists(dataroot)){
    std::cout << "dataroot : " << dataroot << " doesn't exist" << std::endl;
    exit(0);
  }
  std::string imu_file = ConcatenateFolderAndFileName(dataroot, "imu0/data.csv");
  if(use_imu && !FileExists(imu_file)){
    std::cout << "use_imu is set to true, however the imu file : " << imu_file << " doesn't exist" << std::endl;
    exit(0);
  }

  std::string left_image_dir = ConcatenateFolderAndFileName(dataroot, "cam0/data");
  std::string right_image_dir = ConcatenateFolderAndFileName(dataroot, "cam1/data");
  std::vector<std::string> image_names;
  GetFileNames(left_image_dir, image_names);
  if(image_names.size() < 1) return;

  ImuDataList all_imu_data;
  if(use_imu){
    ReadImuData(imu_file, all_imu_data);
  }
  size_t num_imu_data = all_imu_data.size();

  std::sort(image_names.begin(), image_names.end()); 
  for(size_t i = 0; i < image_names.size(); ++i){
    // double image_time = atof(image_names[i].substr(0, 10).c_str()) + atof(image_names[i].substr(10, image_names[i].find_last_of('.')-10).c_str()) / 1e9;
    double image_time = ImageNameToTime(image_names[i]);
    if(num_imu_data > 0){
      // discard images without imu data 
      if(image_time < all_imu_data[0].timestamp) continue;
      if(image_time > all_imu_data[num_imu_data-1].timestamp) break;
    }

    _left_images.emplace_back(ConcatenateFolderAndFileName(left_image_dir, image_names[i]));
    _right_images.emplace_back(ConcatenateFolderAndFileName(right_image_dir, image_names[i]));
    _timestamps.emplace_back(image_time);  
  }

  if(num_imu_data > 0){
    size_t imu_idx = 0;
    double last_image_time = -1;
    for(double image_time : _timestamps){
      ImuDataList mini_batch_imu_data;
      for(; imu_idx < all_imu_data.size()-1; imu_idx++){
        if(all_imu_data[imu_idx+1].timestamp < last_image_time) continue;
        mini_batch_imu_data.emplace_back(all_imu_data[imu_idx]);
        if(all_imu_data[imu_idx].timestamp > image_time) break;
      }
      imu_idx--;
      
      last_image_time = image_time;
      _imu_data.emplace_back(mini_batch_imu_data);
    }
  }
}

void Dataset::ReadImuData(const std::string& imu_file_path, ImuDataList& all_imu_data){
  if(!FileExists(imu_file_path)){
    std::cout << "imu file : " << imu_file_path << " doesn't exist" << std::endl;
    exit(0);
  }

  std::vector<std::vector<std::string> > lines;
  ReadTxt(imu_file_path, lines, ",");
  all_imu_data.resize((lines.size()-1));
  for(size_t i = 1; i < lines.size(); ++i){
    all_imu_data[i-1].timestamp = StringTimeToDouble(lines[i][0]);
    all_imu_data[i-1].gyr << atof(lines[i][1].c_str()), atof(lines[i][2].c_str()), atof(lines[i][3].c_str()); 
    all_imu_data[i-1].acc << atof(lines[i][4].c_str()), atof(lines[i][5].c_str()), atof(lines[i][6].c_str()); 
  }
}

size_t Dataset::GetDatasetLength(){
  return _left_images.size();
}

bool Dataset::GetData(size_t idx, cv::Mat& left_image, cv::Mat& right_image, ImuDataList& batch_imu_data, double& timestamp){
  batch_imu_data.clear();
  if(idx >= _left_images.size()) return false;
  if(!FileExists(_left_images[idx]) || !FileExists(_right_images[idx])) return false;
  left_image = cv::imread(_left_images[idx], 0);
  right_image = cv::imread(_right_images[idx], 0);
  timestamp = _timestamps[idx];
  if(_imu_data.size() > idx){
    std::copy(_imu_data[idx].begin(), _imu_data[idx].end(), std::back_inserter(batch_imu_data));
  }
  return true;
}