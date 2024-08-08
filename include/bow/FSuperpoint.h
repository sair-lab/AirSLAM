/**
 * File: FSuperpoint.cpp
 * Date: November 2023
 * Author: Kuan Xu
 * Description: functions for Superpoint descriptors
 * License: see the LICENSE.txt file
 *
 */
 
#ifndef __D_T_F_SUPERPOINT__
#define __D_T_F_SUPERPOINT__

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Core> 
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <vector>
#include <string>

#include "3rdparty/DBoW2/include/DBoW2/FClass.h"

namespace DBoW2 {


/// Functions to manipulate Superpoint descriptors
class FSuperpoint: protected FClass
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Descriptor length
  static const int L = 256; 
  /// Descriptor type
  typedef Eigen::Matrix<float, L, 1> TDescriptor;
  /// Pointer to a single descriptor
  typedef const TDescriptor *pDescriptor;


  /**
   * Returns the number of dimensions of the descriptor space
   * @return dimensions
   */
  inline static int dimensions()
  {
    return L;
  }

  /**
   * Calculates the mean value of a set of descriptors
   * @param descriptors vector of pointers to descriptors
   * @param mean mean descriptor
   */
  static void meanValue(const std::vector<pDescriptor> &descriptors, 
    TDescriptor &mean);
  
  /**
   * Calculates the (squared) distance between two descriptors
   * @param a
   * @param b
   * @return (squared) distance
   */
  static double distance(const TDescriptor &a, const TDescriptor &b);
  
  /**
   * Returns a string version of the descriptor
   * @param a descriptor
   * @return string version
   */
  static std::string toString(const TDescriptor &a);
  
  /**
   * Returns a descriptor from a string
   * @param a descriptor
   * @param s string version
   */
  static void fromString(TDescriptor &a, const std::string &s);

  /**
   * Returns a mat with the descriptors in float format
   * @param descriptors
   * @param mat (out) NxL 32F matrix
   */
  static void toMat32F(const std::vector<TDescriptor> &descriptors, 
    cv::Mat &mat);

};

} // namespace DBoW2

#endif