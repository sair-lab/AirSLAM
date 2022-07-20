#include <iostream>  
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

void SaveDetectorResult(
    cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, std::string save_path){
}

void SaveMatchingResult(
    cv::Mat& image1, std::vector<cv::KeyPoint>& keypoints1, 
    cv::Mat& image2, std::vector<cv::KeyPoint>& keypoints2, 
    std::vector<cv::DMatch>& matches, std::string save_path){

  cv::Mat save_image;
  if(matches.size() < 1) return;

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

void SaveLineDetectionResult(cv::Mat image, std::vector<Eigen::Vector4d>& lines, std::string save_root, std::string idx){
  std::string line_save_dir = ConcatenateFolderAndFileName(save_root, "debug/line_detection");
  MakeDir(line_save_dir); 
  std::string line_save_image_name = "line_detection_" + idx + ".jpg";
  std::string save_image_path = ConcatenateFolderAndFileName(line_save_dir, line_save_image_name);

  for(auto& line : lines){
    cv::line(image, cv::Point2i((int)(line(0)+0.5), (int)(line(1)+0.5)), 
        cv::Point2i((int)(line(2)+0.5), (int)(line(3)+0.5)), cv::Scalar(0, 250, 0), 2)
  }
  cv::imwrite(save_path, save_image);
}