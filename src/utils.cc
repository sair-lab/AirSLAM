
#include "utils.h"
#include <dirent.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

void ConvertVectorToRt(Eigen::Matrix<double, 7, 1>& m, Eigen::Matrix3d& R, Eigen::Vector3d& t){
  Eigen::Quaterniond q(m(0, 0), m(1, 0), m(2, 0), m(3, 0));
  R = q.matrix();
  t = m.block<3, 1>(4, 0);
}

// (f1 - f2) * (f1 - f2) = f1 * f1 + f2 * f2 - 2 * f1 *f2 = 2 - 2 * f1 * f2 -> [0, 4]
double DescriptorDistance(const Eigen::Matrix<double, 256, 1>& f1, const Eigen::Matrix<double, 256, 1>& f2){
  return 2 * (1.0 - f1.transpose() * f2);
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

void GetFileNames(std::string path, std::vector<std::string>& filenames){
  DIR *pDir;
  struct dirent* ptr;
  std::cout << "path = " << path << std::endl;
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
  file.open(file_path.c_str(), std::ios::out|std::ios::app);
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