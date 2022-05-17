#ifndef POINT_MATCHING_H_
#define POINT_MATCHING_H_

#include "super_glue.h"
#include "read_configs.h"

class PointMatching{
public:
  PointMatching(SuperGlueConfig& superglue_cofig);
  int MatchingPoints(Eigen::Matrix<double, 259, Eigen::Dynamic>& features0, 
      Eigen::Matrix<double, 259, Eigen::Dynamic>& features1, std::vector<cv::DMatch>& matches);

private:
  SuperGlue superglue;
};

typedef std::shared_ptr<PointMatching> PointMatchingPtr;

#endif  // POINT_MATCHING_H_ 