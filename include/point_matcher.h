#ifndef POINT_MATCHING_H_
#define POINT_MATCHING_H_

#include "super_glue.h"
#include "light_glue.h"
#include "read_configs.h"

class PointMatcher{
public:
  PointMatcher(const PointMatcherConfig& _config);

  void NormalizeKeypoints(const Eigen::Matrix<float, 259, Eigen::Dynamic> &features, 
      Eigen::Matrix<float, 259, Eigen::Dynamic>& normalized_features, 
      int width, int height, float scale);

  int MatchingPoints(const Eigen::Matrix<float, 259, Eigen::Dynamic>& features0, 
      const Eigen::Matrix<float, 259, Eigen::Dynamic>& features1, 
      std::vector<cv::DMatch>& matches,  bool outlier_rejection=false);

private:
  PointMatcherConfig _config;
  SuperPointLightGluePtr _lightglue;
  SuperGluePtr _superglue;
};


typedef std::shared_ptr<PointMatcher> PointMatcherPtr;

#endif  // POINT_MATCHING_H_ 
