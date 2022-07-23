#ifndef LINE_PROCESSOR_H_
#define LINE_PROCESSOR_H_

#include <string>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "read_configs.h"

void FilterShortLines(std::vector<Eigen::Vector4f>& lines, float length_thr);
void FilterShortLines(std::vector<Eigen::Vector4d>& lines, float length_thr);
void AssignPointsToLines(std::vector<Eigen::Vector4d>& lines, Eigen::Matrix2Xd& points, std::vector<std::set<int>>& relation);

class LineDetector{
public:
	LineDetector(const LineDetectorConfig &line_detector_config);
	void LineExtractor(cv::Mat& image, std::vector<Eigen::Vector4d>& lines);
	void MergeLines(std::vector<Eigen::Vector4f>& source_lines, std::vector<Eigen::Vector4f>& dst_lines, 
      float angle_threshold, float distance_threshold, float endpoint_threshold, float theta_threshold);

private:
  void MinXY(float& min_x, float& min_y, float& x, float& y, bool to_sort_x);
  void MaxXY(float& max_x, float& max_y, float& x, float& y, bool to_sort_x);
  float PointLineDistance(Eigen::Vector4f line, Eigen::Vector2f point);

private:
	LineDetectorConfig _line_detector_config;
	std::shared_ptr<cv::ximgproc::FastLineDetector> fld;
};
typedef std::shared_ptr<LineDetector> LineDetectorPtr;



#endif  // LINE_PROCESSOR_H_