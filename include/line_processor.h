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

#include "mappoint.h"
#include "frame.h"
#include "read_configs.h"

class LineDetector{
public:
	LineDetector(const LineDetectorConfig &line_detector_config);
	void LineExtractor(cv::Mat& image, std::vector<Eigen::Vector4d>& lines);
	void MergeLines(std::vector<Eigen::Vector4f>& source_lines, std::vector<Eigen::Vector4f>& dst_lines);

private:
	LineDetectorConfig _line_detector_config;
	std::shared_ptr<cv::ximgproc::FastLineDetector> fld;
};
typedef std::shared_ptr<LineDetector> LineDetectorPtr;



#endif  // LINE_PROCESSOR_H_