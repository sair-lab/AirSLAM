#ifndef DEBUG_H_
#define DEBUG_H_

#include <iostream>  
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <math.h>
#include <float.h>
#include <iostream>
#include <numeric>

#include "utils.h"
#include "line_processor.h"
#include "frame.h"

void SaveDetectorResult(
    cv::Mat& image, Eigen::Matrix<float, 259, Eigen::Dynamic>& features, std::string save_root, std::string idx);

void SaveMatchingResult(
    cv::Mat& image1, const std::vector<cv::KeyPoint>& keypoints1, 
    cv::Mat& image2, const std::vector<cv::KeyPoint>& keypoints2, 
    std::vector<cv::DMatch>& matches, std::string save_path);

void SaveStereoMatchResult(cv::Mat& image_left, cv::Mat& image_right, Eigen::Matrix<float, 259, Eigen::Dynamic>& features_left, 
    Eigen::Matrix<float, 259, Eigen::Dynamic>&features_right, std::vector<cv::DMatch>& stereo_matches, std::string stereo_save_root, int frame_id);


void SaveTrackingResult(cv::Mat& last_image, cv::Mat& image, FramePtr last_frame, FramePtr frame, 
    std::vector<cv::DMatch>& matches, std::string save_root);

void SaveLineDetectionResult(cv::Mat& image, std::vector<Eigen::Vector4d>& lines, std::string save_root, std::string idx);
void SaveLineDetectionResult(cv::Mat& image, std::vector<Eigen::Vector4d>& lines, std::vector<Eigen::Vector2i>& junctions, std::string save_root, std::string idx);

void SavePointLineRelation(cv::Mat& image, std::vector<Eigen::Vector4d>& lines, Eigen::Matrix2Xd& points, 
    std::vector<std::map<int, double>>& relation,  std::string save_root, std::string idx);

cv::Mat DrawLinePointRelation(cv::Mat& image, const Eigen::Matrix<float, 259, Eigen::Dynamic>& features,
    const std::vector<Eigen::Vector4d>& lines, const std::vector<std::map<int, double>>& points_on_line, std::vector<int>& line_ids);

void SaveStereoLineMatch(cv::Mat& image_left, cv::Mat& image_right, 
    Eigen::Matrix<float, 259, Eigen::Dynamic>& feature_left,
    Eigen::Matrix<float, 259, Eigen::Dynamic>& feature_right,
    std::vector<Eigen::Vector4d>& lines_left, std::vector<Eigen::Vector4d>& lines_right,
    std::vector<std::map<int, double>>& points_on_line_left, 
    std::vector<std::map<int, double>>& points_on_line_right,
    std::vector<int>& right_to_left_line_matches, std::string save_root, std::string idx);
  

cv::Mat DrawLineWithText(cv::Mat& image, std::vector<Eigen::Vector4d>& lines, std::vector<int>& track_ids);

void DrawStereoLinePair(cv::Mat& image_left, cv::Mat& image_right, FramePtr frame,
    std::string save_root, std::string idx);

void DrawDbowMatchingResults(FramePtr query_frame, cv::Mat& query_image, std::vector<FramePtr>& database_frames, 
    DatabasePtr database, std::string image_root, std::string save_root);

void DrawDbowJunctionMatchingResults(FramePtr query_frame, cv::Mat& query_image, FramePtr database_frame, 
    std::vector<std::vector<bool>>& match_matrix, std::string image_root, std::string save_root);


#endif // DEBUG_H_