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
    cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, std::string save_path){
}

void SaveMatchingResult(
    cv::Mat& image1, std::vector<cv::KeyPoint>& keypoints1, 
    cv::Mat& image2, std::vector<cv::KeyPoint>& keypoints2, 
    std::vector<cv::DMatch>& matches, std::string save_path){

  cv::Mat save_image;
  // if(matches.size() < 1) return;

  cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, save_image);
  cv::imwrite(save_path, save_image);
}

void SaveStereoMatchResult(cv::Mat& image_left, cv::Mat& image_right, Eigen::Matrix<double, 259, Eigen::Dynamic>& features_left, 
    Eigen::Matrix<double, 259, Eigen::Dynamic>&features_right, std::vector<cv::DMatch>& stereo_matches, std::string stereo_save_root, int frame_id){
   std::vector<cv::KeyPoint> left_keypoints, right_keypoints;
  for(size_t i = 0; i < features_left.cols(); ++i){
    double score = features_left(0, i);
    double x = features_left(1, i);
    double y = features_left(2, i);
    left_keypoints.emplace_back(x, y, 8, -1, score);
  }
  for(size_t i = 0; i < features_right.cols(); ++i){
    double score = features_right(0, i);
    double x = features_right(1, i);
    double y = features_right(2, i);
    right_keypoints.emplace_back(x, y, 8, -1, score);
  }
  std::string stereo_debug_save_dir = ConcatenateFolderAndFileName(stereo_save_root, "stereo_debug");
  MakeDir(stereo_debug_save_dir);  
  cv::Mat stereo_debug_left = image_left.clone();
  cv::Mat stereo_debug_right = image_right.clone();
  std::string stereo_save_image_name = "stereo_matching_" + std::to_string(frame_id) + ".jpg";
  std::string stereo_save_image_path = ConcatenateFolderAndFileName(stereo_debug_save_dir, stereo_save_image_name);
  SaveMatchingResult(stereo_debug_left, left_keypoints, stereo_debug_right, right_keypoints, stereo_matches, stereo_save_image_path);
}

void SaveTrackingResult(cv::Mat& last_image, cv::Mat& image, FramePtr last_frame, FramePtr frame, 
    std::vector<cv::DMatch>& matches, std::string save_root){
  std::string debug_save_dir = ConcatenateFolderAndFileName(save_root, "debug/tracking");
  MakeDir(debug_save_dir);  
  cv::Mat debug_image = image.clone();
  cv::Mat debug_last_image = last_image.clone();
  int frame_id = frame->GetFrameId();
  int last_frame_id = last_frame->GetFrameId();
  std::vector<cv::KeyPoint>& kpts = frame->GetAllKeypoints();
  std::vector<cv::KeyPoint>& last_kpts = last_frame->GetAllKeypoints();
  std::string save_image_name = "matching_" + std::to_string(last_frame_id) + "_" + std::to_string(frame_id) + ".jpg";
  std::string save_image_path = ConcatenateFolderAndFileName(debug_save_dir, save_image_name);
    SaveMatchingResult(debug_last_image, last_kpts, debug_image, kpts, matches, save_image_path);
}

void SaveLineDetectionResult(cv::Mat& image, std::vector<Eigen::Vector4d>& lines, std::string save_root, std::string idx){
  cv::Mat img_color;
  cv::cvtColor(image, img_color, cv::COLOR_GRAY2RGB);
  std::string line_save_dir = ConcatenateFolderAndFileName(save_root, "line_detection");
  MakeDir(line_save_dir); 
  std::string line_save_image_name = "line_detection_" + idx + ".jpg";
  std::string save_image_path = ConcatenateFolderAndFileName(line_save_dir, line_save_image_name);

  for(auto& line : lines){
    cv::line(img_color, cv::Point2i((int)(line(0)+0.5), (int)(line(1)+0.5)), 
        cv::Point2i((int)(line(2)+0.5), (int)(line(3)+0.5)), cv::Scalar(0, 250, 0), 1);
  }
  cv::imwrite(save_image_path, img_color);
}

void SavePointLineRelation(cv::Mat& image, std::vector<Eigen::Vector4d>& lines, Eigen::Matrix2Xd& points, 
    std::vector<std::map<int, double>>& relation,  std::string save_root, std::string idx){
  cv::Mat img_color;
  cv::cvtColor(image, img_color, cv::COLOR_GRAY2RGB);
  std::string line_save_dir = ConcatenateFolderAndFileName(save_root, "point_line_relation");
  MakeDir(line_save_dir); 
  std::string line_save_image_name = "point_line_relation" + idx + ".jpg";
  std::string save_image_path = ConcatenateFolderAndFileName(line_save_dir, line_save_image_name);

  size_t point_num = points.cols();
  std::vector<cv::Scalar> colors(point_num, cv::Scalar(0, 255, 0));
  std::vector<int> radii(point_num, 1);

  // draw lines
  for(size_t i = 0; i < lines.size(); i++){
    cv::Scalar color = GenerateColor(i);
    Eigen::Vector4d line = lines[i];
    cv::line(img_color, cv::Point2i((int)(line(0)+0.5), (int)(line(1)+0.5)), 
        cv::Point2i((int)(line(2)+0.5), (int)(line(3)+0.5)), color, 1);

    for(auto& kv : relation[i]){
      colors[kv.first] = color;
      radii[kv.first] *= 2;
    }
  }

  for(size_t j = 0; j < point_num; j++){
    double x = points(0, j);
    double y = points(1, j);
    cv::circle(img_color, cv::Point(x, y), radii[j], colors[j], 1, cv::LINE_AA);
  }


  cv::imwrite(save_image_path, img_color);
}

cv::Mat DrawLinePointRelation(cv::Mat& image, Eigen::Matrix<double, 259, Eigen::Dynamic>& features,
    std::vector<Eigen::Vector4d>& lines, std::vector<std::map<int, double>>& points_on_line, std::vector<int>& line_ids){
  cv::Mat img_color;
  cv::cvtColor(image, img_color, cv::COLOR_GRAY2RGB);

  Eigen::Matrix2Xd points = features.middleRows(1, 2);
  size_t point_num = points.cols();
  std::vector<cv::Scalar> colors(point_num, cv::Scalar(0, 255, 0));
  std::vector<int> radii(point_num, 2);

  // draw lines
  for(size_t i = 0; i < lines.size(); i++){
    cv::Scalar color = GenerateColor(line_ids[i]);
    Eigen::Vector4d line = lines[i];
    cv::line(img_color, cv::Point2i((int)(line(0)+0.5), (int)(line(1)+0.5)), 
        cv::Point2i((int)(line(2)+0.5), (int)(line(3)+0.5)), color, 2);

    for(auto& kv : points_on_line[i]){
      colors[kv.first] = color;
      radii[kv.first] *= 2;
    }
  }

  // draw points
  for(size_t j = 0; j < point_num; j++){
    double x = points(0, j);
    double y = points(1, j);
    cv::circle(img_color, cv::Point(x, y), radii[j], colors[j], 1, cv::LINE_AA);
  }

  return img_color;
}

void SaveStereoLineMatch(cv::Mat& image_left, cv::Mat& image_right, 
    Eigen::Matrix<double, 259, Eigen::Dynamic>& feature_left,
    Eigen::Matrix<double, 259, Eigen::Dynamic>& feature_right,
    std::vector<Eigen::Vector4d>& lines_left, std::vector<Eigen::Vector4d>& lines_right,
    std::vector<std::map<int, double>>& points_on_line_left, 
    std::vector<std::map<int, double>>& points_on_line_right,
    std::vector<int>& right_to_left_line_matches, std::string save_root, std::string idx){
  
  std::vector<int> line_ids_left(lines_left.size());
  std::vector<int> line_ids_right(lines_right.size(), -1);
  size_t line_id = 1;
  for(size_t i = 0; i < lines_left.size(); i++){
    line_ids_left[i] = line_id;
    int matched_right = right_to_left_line_matches[i];
    if(matched_right > 0){
      line_ids_right[matched_right] = line_id;
    }
    line_id++;
  }

  for(size_t i = 0; i < lines_right.size(); i++){
    if(line_ids_right[i] < 0){
      line_ids_right[i] = line_id++;
    }
  }

  cv::Mat img_left_color = DrawLinePointRelation(image_left, feature_left, lines_left, points_on_line_left, line_ids_left);
  cv::Mat img_right_color = DrawLinePointRelation(image_right, feature_right, lines_right, points_on_line_right, line_ids_right);

  // save image
  std::string stereo_line_matching_save_dir = ConcatenateFolderAndFileName(save_root, "stereo_line_matching");
  MakeDir(stereo_line_matching_save_dir); 
  std::string line_save_image_name = "stereo_line_matching_" + idx + ".jpg";
  std::string save_image_path = ConcatenateFolderAndFileName(stereo_line_matching_save_dir, line_save_image_name);

  int save_rows = img_left_color.rows;
  int save_cols = img_left_color.cols + 10 + img_right_color.cols;
  cv::Mat save_image = cv::Mat::zeros(save_rows, save_cols, img_left_color.type());
  cv::Rect rect1(0, 0, img_left_color.cols, img_left_color.rows);
  cv::Rect rect2( img_left_color.cols + 10, 0, img_right_color.cols, img_left_color.rows);
  img_left_color.copyTo(save_image(rect1));
  img_right_color.copyTo(save_image(rect2));
  cv::imwrite(save_image_path, save_image);
}

cv::Mat DrawLineWithText(cv::Mat& image, std::vector<Eigen::Vector4d>& lines, std::vector<int>& track_ids){
  cv::Mat img_color;
  cv::cvtColor(image, img_color, cv::COLOR_GRAY2RGB);

  // draw lines
  for(size_t i = 0; i < lines.size(); i++){
    cv::Scalar color = GenerateColor(track_ids[i]);
    Eigen::Vector4d line = lines[i];
    // std::cout << "line_id = " << track_ids[i] << " line = " << line.transpose() << std::endl;
    cv::line(img_color, cv::Point2i((int)(line(0)+0.5), (int)(line(1)+0.5)), 
        cv::Point2i((int)(line(2)+0.5), (int)(line(3)+0.5)), color, 2);

    cv::putText(img_color, std::to_string(track_ids[i]), cv::Point((int)((line(0)+line(2))/2), (int)((line(1)+line(3))/2)), 
        cv::FONT_HERSHEY_DUPLEX, 1.0, color, 2);
  }
  return img_color;
}

void DrawStereoLinePair(cv::Mat& image_left, cv::Mat& image_right, FramePtr frame,
    std::string save_root, std::string idx){
  const std::vector<Eigen::Vector4d>& lines_left = frame->GatAllLines();
  const std::vector<Eigen::Vector4d>& lines_right = frame->GatAllRightLines();
  const std::vector<int>& line_track_ids = frame->GetAllLineTrackId();
  const std::vector<bool>& lines_right_valid = frame->GetAllRightLineStatus();

  std::vector<int> good_track_ids;  
  std::vector<Eigen::Vector4d> good_lines_left, good_lines_right;
  for(size_t i = 0; i < lines_left.size(); i++){
    if(!lines_right_valid[i]) continue;
    good_lines_left.push_back(lines_left[i]);
    good_lines_right.push_back(lines_right[i]);
    good_track_ids.push_back(line_track_ids[i]);
  }

  // std::cout << "left DrawLineWithText-----------------------" << std::endl;
  cv::Mat img_left_color = DrawLineWithText(image_left, good_lines_left, good_track_ids);
  // std::cout << "right DrawLineWithText-----------------------" << std::endl;
  cv::Mat img_right_color = DrawLineWithText(image_right, good_lines_right, good_track_ids);

  // save image
  std::string stereo_line_matching_save_dir = ConcatenateFolderAndFileName(save_root, "stereo_line_pair");
  MakeDir(stereo_line_matching_save_dir); 
  std::string line_save_image_name = "stereo_line_pair" + idx + ".jpg";
  std::string save_image_path = ConcatenateFolderAndFileName(stereo_line_matching_save_dir, line_save_image_name);

  int save_rows = img_left_color.rows;
  int save_cols = img_left_color.cols + 10 + img_right_color.cols;
  cv::Mat save_image = cv::Mat::zeros(save_rows, save_cols, img_left_color.type());
  cv::Rect rect1(0, 0, img_left_color.cols, img_left_color.rows);
  cv::Rect rect2( img_left_color.cols + 10, 0, img_right_color.cols, img_left_color.rows);
  img_left_color.copyTo(save_image(rect1));
  img_right_color.copyTo(save_image(rect2));
  cv::imwrite(save_image_path, save_image);

}