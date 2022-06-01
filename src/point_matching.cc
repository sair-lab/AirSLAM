#include "point_matching.h"

#include <opencv2/opencv.hpp>

PointMatching::PointMatching(SuperGlueConfig& superglue_cofig) :superglue(superglue_cofig){
  if (!superglue.build()){
    std::cout << "Erron in superglue building" << std::endl;
  }
}

int PointMatching::MatchingPoints(Eigen::Matrix<double, 259, Eigen::Dynamic>& features0, 
    Eigen::Matrix<double, 259, Eigen::Dynamic>& features1, std::vector<cv::DMatch>& matches){
  matches.clear();

  Eigen::VectorXi indices0, indices1;
  Eigen::VectorXd mscores0, mscores1;
  superglue.infer(features0, features1, indices0, indices1, mscores0, mscores1);

  int num_match = 0;
  std::vector<cv::Point2f> points0, points1;
  std::vector<int> point_indexes;
  for(size_t i = 0; i < indices0.size(); i++){
    if(indices0(i) < indices1.size() && indices0(i) >= 0 && indices1(indices0(i)) == i){
      double d = 1.0 - (mscores0[i] + mscores1[indices0[i]]) / 2.0;
      matches.emplace_back(i, indices0[i], d);
      points0.emplace_back(features0(1, i), features0(2, i));
      points1.emplace_back(features1(1, indices0(i)), features1(2, indices0(i)));
      num_match++;
    }
  }
  

  // reject outliers
  std::vector<uchar> inliers;
  cv::findFundamentalMat(points0, points1, cv::FM_RANSAC, 3, 0.99, inliers);
  int j = 0;
  for(int i = 0; i < matches.size(); i++){
    if(inliers[i]){
      matches[j++] = matches[i];
    }
  }
  matches.resize(j);

  return j;
}