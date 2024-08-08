#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sys/types.h>    
#include <sys/stat.h>
#include <functional>
#include <map>
#include <limits.h>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>


#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/slam3d_addons/types_slam3d_addons.h>

typedef std::shared_ptr<g2o::Line3D> Line3DPtr;
typedef std::shared_ptr<const g2o::Line3D> ConstLine3DPtr;

// Eigen type
typedef Eigen::Matrix<double, 5, 1> Vector5d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 8, 1> Vector8d;
typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 8, 8> Matrix8d;
typedef Eigen::Matrix<double, 9, 9> Matrix9d;
typedef Eigen::Matrix<double, 15, 15> Matrix15d;

template <template <typename, typename> class Container, typename Type>
using Aligned = Container<Type, Eigen::aligned_allocator<Type>>;

template <typename KeyType, typename ValueType>
using AlignedMap =
    std::map<KeyType, ValueType, std::less<KeyType>,
             Eigen::aligned_allocator<std::pair<const KeyType, ValueType>>>;



void ConvertVectorToRt(Eigen::Matrix<double, 7, 1>& m, Eigen::Matrix3d& R, Eigen::Vector3d& t);
float DescriptorDistance(const Eigen::Matrix<float, 256, 1>& f1, const Eigen::Matrix<float, 256, 1>& f2);
std::string DoubleTimeToString(double timestamp_seconds);
double StringTimeToDouble(std::string time_str);
double ImageNameToTime(const std::string& image_name);
double CalculateStdDev(const std::vector<double>& data);

// visualization
cv::Scalar GenerateColor(int id);
void GenerateColor(int id, Eigen::Vector3d color);
cv::Mat DrawFeatures(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, 
    const std::vector<Eigen::Vector4d>& lines, bool draw_on_one);
cv::Mat DrawFeatures(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, 
    const std::vector<bool>& inliers, const std::vector<Eigen::Vector4d>& lines, 
    const std::vector<int>& line_track_ids, const std::vector<std::map<int, double>>& points_on_lines);
cv::Mat DrawMatches(const cv::Mat& ref_image, const cv::Mat& image, const std::vector<cv::KeyPoint>& ref_kpts, 
    const std::vector<cv::KeyPoint>& kpts, const std::vector<cv::DMatch>& matches);


// files
void GetFileNames(std::string path, std::vector<std::string>& filenames);
bool FileExists(const std::string& file);
bool PathExists(const std::string& path);
void ConcatenateFolderAndFileName(
    const std::string& folder, const std::string& file_name,
    std::string* path);

std::string ConcatenateFolderAndFileName(
    const std::string& folder, const std::string& file_name);

void MakeDir(const std::string& path);

void ReadTxt(const std::string& file_path, 
    std::vector<std::vector<std::string> >& lines, std::string seq);

void WriteTxt(const std::string file_path, 
    std::vector<std::vector<std::string> >& lines, std::string seq);

void SaveTumTrajectoryToFile(const std::string file_path, 
    const std::vector<std::pair<double, Eigen::Matrix4d>>& trajectory);

void SaveTumTrajectoryToFile(const std::string file_path, 
    const std::vector<std::pair<std::string, Eigen::Matrix4d>>& trajectory);

// boost serialization
template <class Archive>
void SerializeLine3D(Archive &ar, Line3DPtr &line, const unsigned int version){
  g2o::Vector6 v;
  if (Archive::is_saving::value){
    v = line->toCartesian();
  }

  ar & boost::serialization::make_array(v.data(), v.size());

  if (Archive::is_loading::value){
    g2o::Line3D line_3d = g2o::Line3D::fromCartesian(v);
    line = std::make_shared<g2o::Line3D>(line_3d);
  }
}

template<class Archive>
void SerializeCVMat(Archive& ar, cv::Mat& mat, const unsigned int version){
  int cols, rows, type;
  bool continuous;

  if (Archive::is_saving::value) {
    cols = mat.cols; rows = mat.rows; type = mat.type();
    continuous = mat.isContinuous();
  }

  ar & cols & rows & type & continuous;

  if (Archive::is_loading::value)
    mat.create(rows, cols, type);

  if (continuous) {
    const unsigned int data_size = rows * cols * mat.elemSize();
    ar & boost::serialization::make_array(mat.ptr(), data_size);
  } else {
    const unsigned int row_size = cols*mat.elemSize();
    for (int i = 0; i < rows; i++) {
        ar & boost::serialization::make_array(mat.ptr(i), row_size);
    }
  }
}

template<class Archive>
void SerializeKeypoints(Archive& ar, std::vector<cv::KeyPoint>& kps, const unsigned int version){
  int kp_num;

  if (Archive::is_saving::value){
    kp_num = kps.size();
  }

  ar & kp_num;

  if (Archive::is_loading::value){
    kps.resize(kp_num);
  }

  for(int i=0; i < kp_num; ++i){
    ar & kps[i].angle;
    ar & kps[i].response;
    ar & kps[i].size;
    ar & kps[i].pt.x;
    ar & kps[i].pt.y;
    ar & kps[i].class_id;
    ar & kps[i].octave;
  }
}

template <class Archive>
void SerializeEigenVector3dList(Archive &ar, std::vector<Eigen::Vector3d> &v, const unsigned int version){
  int l;
  if (Archive::is_saving::value){
    l = v.size();
  }

  ar & l;

  if (Archive::is_loading::value){
    v.resize(l);
  }

  for(int i=0; i < l; ++i){
    ar & boost::serialization::make_array(v[i].derived().data(), v[i].size());;
  }
}

template <class Archive>
void SerializeEigenVector4dList(Archive &ar, std::vector<Eigen::Vector4d> &v, const unsigned int version){
  int l;
  if (Archive::is_saving::value){
    l = v.size();
  }

  ar & l;

  if (Archive::is_loading::value){
    v.resize(l);
  }

  for(int i=0; i < l; ++i){
    ar & boost::serialization::make_array(v[i].derived().data(), v[i].size());;
  }
}

template<class Archive>
void SerializeFeatures(Archive& ar, Eigen::Matrix<float, 259, Eigen::Dynamic>& features, const unsigned int version){
  int cols, rows;

  if (Archive::is_saving::value) {
    cols = features.cols(); 
    rows = features.rows();
  }

  ar & cols;
  ar & rows;

  if(Archive::is_loading::value){
    features.resize(rows, cols);
  }

  ar & boost::serialization::make_array(features.data(), features.size());
}

template<class Archive>
void SerializeDiagonalMatrix6d(Archive& ar, Eigen::DiagonalMatrix<double, 6>& data, const unsigned int version){
  Vector6d v;

  if (Archive::is_saving::value) {
    v = data.diagonal();
  }

  ar & boost::serialization::make_array(v.data(), v.size());

  if(Archive::is_loading::value){
    data.diagonal() = v;
  }
}


// read and write binary file
template<class Matrix>
void WriteMatrixToBinary(const char* filename, const Matrix& matrix){
  std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
  typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
  out.write((char*) (&rows), sizeof(typename Matrix::Index));
  out.write((char*) (&cols), sizeof(typename Matrix::Index));
  out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
  out.close();
}

template<class Matrix>
void ReadMatrixFromBinary(const char* filename, Matrix& matrix){
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  typename Matrix::Index rows=0, cols=0;
  in.read((char*) (&rows),sizeof(typename Matrix::Index));
  in.read((char*) (&cols),sizeof(typename Matrix::Index));
  matrix.resize(rows, cols);
  in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
  in.close();
}


#endif  // UTILS_H_