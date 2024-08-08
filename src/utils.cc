
#include "utils.h"
#include <dirent.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <numeric> 

void ConvertVectorToRt(Eigen::Matrix<double, 7, 1>& m, Eigen::Matrix3d& R, Eigen::Vector3d& t){
  Eigen::Quaterniond q(m(0, 0), m(1, 0), m(2, 0), m(3, 0));
  R = q.matrix();
  t = m.block<3, 1>(4, 0);
}

// (f1 - f2) * (f1 - f2) = f1 * f1 + f2 * f2 - 2 * f1 *f2 = 2 - 2 * f1 * f2 -> [0, 4]
float DescriptorDistance(const Eigen::Matrix<float, 256, 1>& f1, const Eigen::Matrix<float, 256, 1>& f2){
  return 2 * (1.0 - f1.transpose() * f2);
}

std::string DoubleTimeToString(double timestamp_seconds){
  // auto timestamp_nanoseconds = static_cast<long long>(timestamp_seconds * 1e9);
  // std::ostringstream oss;
  // oss << timestamp_nanoseconds;
  // return oss.str();

  std::ostringstream stream;
  stream << std::setprecision(9) << std::fixed << timestamp_seconds;
  return stream.str();
}

cv::Scalar GenerateColor(int id){
  id++;
  int red = (id * 23) % 255;
  int green = (id * 53) % 255;
  int blue = (id * 79) % 255;
  return cv::Scalar(blue, green, red);
}

void GenerateColor(int id, Eigen::Vector3d color){
  id++;
  int red = (id * 23) % 255;
  int green = (id * 53) % 255;
  int blue = (id * 79) % 255;
  color << red, green, blue;
  color *= (1.0 / 255.0);
}

double StringTimeToDouble(std::string time_str){
  time_str.erase(std::remove_if(time_str.begin(), time_str.end(), [](char c) { return c == '.'; }), time_str.end());

  double time_double1 = atof(time_str.substr(0, 10).c_str());
  double time_double2 = atof(("0." + time_str.substr(10, time_str.length()+1)).c_str());
  return (time_double1 + time_double2);
}

double ImageNameToTime(const std::string& image_name){
  std::string time_str;
  size_t pos = image_name.find_last_of('.');
  if (pos != std::string::npos) {
    time_str = image_name.substr(0, pos);
  } 
  return StringTimeToDouble(time_str);
}

double CalculateStdDev(const std::vector<double>& data) {
  if (data.empty()) return 0.0;
  double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();

  double variance = 0.0;
  for (const auto& value : data) {
    variance += (value - mean) * (value - mean);
  }
  variance /= data.size();
  return std::sqrt(variance);
}

cv::Mat DrawFeatures(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, 
    const std::vector<Eigen::Vector4d>& lines, bool draw_on_one){
  cv::Mat img_color;
  cv::cvtColor(image, img_color, cv::COLOR_GRAY2RGB);

  // draw points
  for(size_t j = 0; j < keypoints.size(); j++){
    cv::circle(img_color, keypoints[j].pt, 2, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
  }

  // draw lines
  cv::Mat drawed_image;
  cv::Mat line_image = img_color.clone();
  for(size_t i = 0; i < lines.size(); i++){
    Eigen::Vector4d line = lines[i];
    cv::line(line_image, cv::Point2i((int)(line(0)+0.5), (int)(line(1)+0.5)), 
        cv::Point2i((int)(line(2)+0.5), (int)(line(3)+0.5)), cv::Scalar(0, 0, 255), 3);
  }

  if(draw_on_one){
    drawed_image = line_image;
  }else{
    cv::hconcat(line_image, img_color, drawed_image);
  }

  return drawed_image;
}


cv::Mat DrawFeatures(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, 
    const std::vector<bool>& inliers, const std::vector<Eigen::Vector4d>& lines, 
    const std::vector<int>& line_track_ids, const std::vector<std::map<int, double>>& points_on_lines){
  cv::Mat img_color;
  cv::cvtColor(image, img_color, cv::COLOR_GRAY2RGB);

  size_t point_num = keypoints.size();
  std::vector<cv::Scalar> colors(point_num, cv::Scalar(0, 255, 0));
  std::vector<int> radii(point_num, 2);

  // draw lines
  for(size_t i = 0; i < lines.size(); i++){
    if(line_track_ids[i] < 0) continue;
    cv::Scalar color = GenerateColor(line_track_ids[i]);
    Eigen::Vector4d line = lines[i];
    cv::line(img_color, cv::Point2i((int)(line(0)+0.5), (int)(line(1)+0.5)), 
        cv::Point2i((int)(line(2)+0.5), (int)(line(3)+0.5)), color, 2);

    cv::putText(img_color, std::to_string(line_track_ids[i]), cv::Point((int)((line(0)+line(2))/2), 
        (int)((line(1)+line(3))/2)), cv::FONT_HERSHEY_DUPLEX, 1.0, color, 2);

    for(auto& kv : points_on_lines[i]){
      colors[kv.first] = color;
      // radii[kv.first] *= 2;
      radii[kv.first] = 3;
    }
  }

  // draw points
  for(size_t j = 0; j < point_num; j++){
    cv::Scalar colar = inliers[j] ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    cv::circle(img_color, keypoints[j].pt, radii[j], colar, 2, cv::LINE_AA);
    // cv::circle(img_color, keypoints[j].pt, radii[j], colors[j], 1, cv::LINE_AA);
  }

  return img_color;
}

cv::Mat DrawMatches(const cv::Mat& ref_image, const cv::Mat& image, const std::vector<cv::KeyPoint>& ref_kpts, 
    const std::vector<cv::KeyPoint>& kpts, const std::vector<cv::DMatch>& matches){

  cv::Mat merged_image;
  cv::hconcat(ref_image, image, merged_image);
  cv::Mat rgba_image;
  cv::cvtColor(merged_image, rgba_image, cv::COLOR_BGR2BGRA);

  for (size_t i = 0; i < matches.size(); i++) {
    cv::line(rgba_image, cv::Point2f(ref_kpts[matches[i].queryIdx].pt.x + image.cols, ref_kpts[matches[i].queryIdx].pt.y),  
              cv::Point2f(kpts[matches[i].trainIdx].pt.x + ref_image.cols, kpts[matches[i].trainIdx].pt.y), 
              cv::Scalar(0,255,0, 10), 2);    
  }
  cv::Mat result;
  cv::cvtColor(rgba_image, result, cv::COLOR_BGRA2BGR);

  return result;
}

void GetFileNames(std::string path, std::vector<std::string>& filenames){
  DIR *pDir;
  struct dirent* ptr;
  if(!(pDir = opendir(path.c_str()))){
    std::cout << "Folder doesn't Exist!" << std::endl;
    return;
  }
  while((ptr = readdir(pDir))!= 0) {
    if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
      filenames.push_back(ptr->d_name);
    }
  }
  closedir(pDir);
}

bool FileExists(const std::string& file) {
  struct stat file_status;
  if (stat(file.c_str(), &file_status) == 0 &&
      (file_status.st_mode & S_IFREG)) {
    return true;
  }
  return false;
}


bool PathExists(const std::string& path) {
  struct stat file_status;
  if (stat(path.c_str(), &file_status) == 0 &&
      (file_status.st_mode & S_IFDIR)) {
    return true;
  }
  return false;
}

void ConcatenateFolderAndFileName(
    const std::string& folder, const std::string& file_name,
    std::string* path) {
  *path = folder;
  if (path->back() != '/') {
    *path += '/';
  }
  *path = *path + file_name;
}

std::string ConcatenateFolderAndFileName(
    const std::string& folder, const std::string& file_name) {
  std::string path;
  ConcatenateFolderAndFileName(folder, file_name, &path);
  return path;
}

void MakeDir(const std::string& path){
  if(!PathExists(path)){
    mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  }
}

void ReadTxt(const std::string& file_path, 
    std::vector<std::vector<std::string> >& lines, std::string seq){
  if(!FileExists(file_path)){
    std::cout << "file: " << file_path << " dosen't exist" << std::endl;
    exit(0);
  }
  
  std::ifstream infile(file_path, std::ifstream::in);   
  if(!infile.is_open()){
    std::cout << "open file: " << file_path << " failure" << std::endl;
    exit(0);
  }

  std::string line;
  while (getline(infile, line)){ 
    std::string whitespaces(" \t\f\v\n\r");
    std::size_t found = line.find_last_not_of(whitespaces);
    if (found!=std::string::npos){
      line.erase(found+1);

      std::vector<std::string> line_data;
      while (true){
        int index = line.find(seq);
        std::string sub_str = line.substr(0, index);
        if (!sub_str.empty()){
          line_data.push_back(sub_str);
        }

        line.erase(0, index + seq.size());
        if (index == -1){
          break;
        }
      }
      lines.emplace_back(line_data);
    }
    else{
      line.clear();            // str is all whitespace
    }
  }
}

void WriteTxt(const std::string file_path, 
    std::vector<std::vector<std::string> >& lines, std::string seq){
  std::fstream file;
  file.open(file_path.c_str(), std::ios::out);
  if(!file.good()){
    std::cout << "Error: cannot open file " << file_path << std::endl;
    exit(0);
  }
  for(std::vector<std::string>& line : lines){
    size_t num_in_line = line.size();
    if(num_in_line < 1) continue;
    std::string line_txt = line[0];
    for(size_t i = 1; i < num_in_line; ++i){
      line_txt = line_txt + seq + line[i];
    }
    line_txt += "\n";
    file << line_txt;
  }
  file.close();
}

void SaveTumTrajectoryToFile(const std::string file_path, 
    const std::vector<std::pair<double, Eigen::Matrix4d>>& trajectory){
  std::cout << "Save file to " << file_path << std::endl;
  std::ofstream f;
  f.open(file_path.c_str());
  f << std::fixed;
  std::cout << "trajectory.size = " << trajectory.size() << std::endl;
  for(const auto& pose_pair : trajectory){
    Eigen::Vector3d t = pose_pair.second.block<3, 1>(0, 3);
    Eigen::Quaterniond q(pose_pair.second.block<3, 3>(0, 0));

    f << std::setprecision(9) << pose_pair.first << " " 
      << std::setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " "
      << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
  f.close();
}

void SaveTumTrajectoryToFile(const std::string file_path, 
    const std::vector<std::pair<std::string, Eigen::Matrix4d>>& trajectory){
  std::cout << "Save file to " << file_path << std::endl;
  std::ofstream f;
  f.open(file_path.c_str());
  f << std::fixed;
  std::cout << "trajectory.size = " << trajectory.size() << std::endl;
  for(const auto& pose_pair : trajectory){
    Eigen::Vector3d t = pose_pair.second.block<3, 1>(0, 3);
    Eigen::Quaterniond q(pose_pair.second.block<3, 3>(0, 0));

    f << pose_pair.first << " " 
      << std::setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " "
      << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
  f.close();
}
