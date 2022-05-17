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
  for(size_t i = 0; i < indices0.size(); i++){
    if(indices0(i) < indices1.size() && indices0(i) >= 0 && indices1(indices0(i)) == i){
      double d = 1.0 - (mscores0[i] + mscores1[indices0[i]]) / 2.0;
      matches.emplace_back(i, indices0[i], d);
      num_match++;
    }
  }

  return num_match;
}