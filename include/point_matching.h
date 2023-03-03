#ifndef POINT_MATCHING_H_
#define POINT_MATCHING_H_

#include "super_glue.h"
#include "read_configs.h"

class PointMatching{
public:
  PointMatching(SuperGlueConfig& superglue_config);
  int MatchingPoints(const Eigen::Matrix<double, 259, Eigen::Dynamic>& features0, 
      const Eigen::Matrix<double, 259, Eigen::Dynamic>& features1, std::vector<cv::DMatch>& matches,  bool outlier_rejection=false);
  Eigen::Matrix<double, 259, Eigen::Dynamic> NormalizeKeypoints(
      const Eigen::Matrix<double, 259, Eigen::Dynamic> &features, int width, int height);

private:
  SuperGlue superglue;
  SuperGlueConfig _superglue_config;
};

typedef std::shared_ptr<PointMatching> PointMatchingPtr;

#endif  // POINT_MATCHING_H_ 
