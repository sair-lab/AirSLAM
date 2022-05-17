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

  drawMatches(image1, keypoints1, image2, keypoints2, matches, save_image);
  imwrite(save_path, save_image);
}