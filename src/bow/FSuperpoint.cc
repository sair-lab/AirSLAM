/**
 * File: FSuperpoint.cpp
 * Date: November 2023
 * Author: Kuan Xu
 * Description: functions for Superpoint descriptors
 * License: see the LICENSE.txt file
 *
 */
 
#include <vector>
#include <string>
#include <sstream>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "3rdparty/DBoW2/include/DBoW2/FClass.h"
#include "include/bow/FSuperpoint.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------

void FSuperpoint::meanValue(const std::vector<FSuperpoint::pDescriptor> &descriptors, 
  FSuperpoint::TDescriptor &mean)
{
  mean = Eigen::Matrix<float, L, 1>::Zero();
  float s = descriptors.size();
  vector<FSuperpoint::pDescriptor>::const_iterator it;
  for(it = descriptors.begin(); it != descriptors.end(); ++it)
  {
    const FSuperpoint::TDescriptor &desc = **it;
    mean += desc / s;
  }
}

// --------------------------------------------------------------------------
  
double FSuperpoint::distance(const FSuperpoint::TDescriptor &a, const FSuperpoint::TDescriptor &b)
{
  FSuperpoint::TDescriptor diff = a - b;
  return diff.transpose() * diff;
}

// --------------------------------------------------------------------------

std::string FSuperpoint::toString(const FSuperpoint::TDescriptor &a)
{
  stringstream ss;
  for (int i = 0; i < L; i++) {
    ss << a(i, 0) << " ";
  }
  return ss.str();
}

// --------------------------------------------------------------------------
  
void FSuperpoint::fromString(FSuperpoint::TDescriptor &a, const std::string &s)
{
  stringstream ss(s);
  for (int i = 0; i < L; i++) {
    std::string tmp;
    ss >> tmp;
    a(i, 0) = std::stof(tmp);
  }
}

// --------------------------------------------------------------------------

void FSuperpoint::toMat32F(const std::vector<TDescriptor> &descriptors, 
    cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }
  
  const int N = descriptors.size();
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1>> eigen_map(descriptors[0].data(), L, N);
  Eigen::MatrixXf eigen_matrix = eigen_map;
  cv::eigen2cv(eigen_matrix, mat);
}

// --------------------------------------------------------------------------

} // namespace DBoW2