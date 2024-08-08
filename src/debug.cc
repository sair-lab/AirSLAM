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
#include "bow/database.h"

void SaveDetectorResult(
    cv::Mat& image, Eigen::Matrix<float, 259, Eigen::Dynamic>& features, std::string save_root, std::string idx){
  cv::Mat img_color;
  cv::cvtColor(image, img_color, cv::COLOR_GRAY2RGB);
  for(size_t j = 0; j < features.cols(); j++){
    double x = features(1, j);
    double y = features(2, j);
    cv::circle(img_color, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
  }

  std::string save_image_name = "point_detection_" + idx + ".jpg";
  std::string save_image_path = ConcatenateFolderAndFileName(save_root, save_image_name);
  cv::imwrite(save_image_path, img_color);
}

void SaveMatchingResult(
    cv::Mat& image1, const std::vector<cv::KeyPoint>& keypoints1, 
    cv::Mat& image2, const std::vector<cv::KeyPoint>& keypoints2, 
    std::vector<cv::DMatch>& matches, std::string save_path){

  cv::Mat save_image;
  // if(matches.size() < 1) return;

  cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, save_image);
  cv::imwrite(save_path, save_image);
}

void SaveStereoMatchResult(cv::Mat& image_left, cv::Mat& image_right, Eigen::Matrix<float, 259, Eigen::Dynamic>& features_left, 
    Eigen::Matrix<float, 259, Eigen::Dynamic>&features_right, std::vector<cv::DMatch>& stereo_matches, std::string stereo_save_root, int frame_id){
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
  std::string debug_save_dir = ConcatenateFolderAndFileName(save_root, "tracking");
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

void SaveLineDetectionResult(cv::Mat& image, std::vector<Eigen::Vector4d>& lines, 
    std::vector<Eigen::Vector2i>& junctions, std::string save_root, std::string idx){
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

  for (Eigen::Vector2i& junction : junctions) {
    cv::circle(img_color, cv::Point2i(junction(0), junction(1)), 3, cv::Scalar(0, 0, 255), -1);
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

cv::Mat DrawLinePointRelation(cv::Mat& image, const Eigen::Matrix<float, 259, Eigen::Dynamic>& features,
    const std::vector<Eigen::Vector4d>& lines, const std::vector<std::map<int, double>>& points_on_line, std::vector<int>& line_ids){
  cv::Mat img_color;
  cv::cvtColor(image, img_color, cv::COLOR_GRAY2RGB);

  Eigen::Matrix2Xf points = features.middleRows(1, 2);
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
    Eigen::Matrix<float, 259, Eigen::Dynamic>& feature_left,
    Eigen::Matrix<float, 259, Eigen::Dynamic>& feature_right,
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


void DrawDbowMatchingResults(FramePtr query_frame, cv::Mat& query_image, std::vector<FramePtr>& database_frames, 
    DatabasePtr database, std::string image_root, std::string save_root){
  std::vector<std::string> image_names;
  GetFileNames(image_root, image_names);
  std::sort(image_names.begin(), image_names.end());
  std::function<cv::Mat(FramePtr)> read_image = [&](FramePtr frame_){
    double timestamp = frame_->GetTimestamp();
    std::string frame_image_name = "";
    double dt = 100;
    for(std::string image_name : image_names){
      double image_time = ImageNameToTime(image_name);
      double time_diff = std::abs(image_time-timestamp);
      if(time_diff < dt){
        frame_image_name = image_name;
        dt = time_diff;
      }
    }

    if(dt > 0.1){
      return cv::Mat();
    }

    std::string image_path = ConcatenateFolderAndFileName(image_root, frame_image_name);
    cv::Mat image = cv::imread(image_path, 0);

    cv::Mat image_rect;
    CameraPtr camera = frame_->GetCamera();
    camera->UndistortImage(image, image_rect);

    return image_rect;
  };

  const Eigen::Matrix<float, 259, Eigen::Dynamic>& query_features = query_frame->GetAllFeatures();
  std::vector<Eigen::Vector4d> query_lines = query_frame->GatAllLines();
  std::vector<std::map<int, double>> query_points_on_line = query_frame->GetPointsOnLines();
  std::vector<int> query_line_ids(query_lines.size());
  std::iota(query_line_ids.begin(), query_line_ids.end(), 1);
  cv::Mat query_drawed_image = DrawLinePointRelation(query_image, query_features, query_lines, query_points_on_line, query_line_ids);

  const std::vector<cv::KeyPoint>& query_keypoints = query_frame->GetAllKeypoints();

  DBoW2::WordIdToFeatures query_word_features; 
  DBoW2::BowVector query_bow_vector;
  std::vector<DBoW2::WordId> query_word_of_features;
  database->FrameToBow(query_frame, query_word_features, query_bow_vector, query_word_of_features);

  for(FramePtr database_frame : database_frames){
    const Eigen::Matrix<float, 259, Eigen::Dynamic>& base_features = database_frame->GetAllFeatures();
    DBoW2::WordIdToFeatures base_word_features; 
    DBoW2::BowVector base_bow_vector;
    std::vector<DBoW2::WordId> base_word_of_features;
    database->FrameToBow(database_frame, base_word_features, base_bow_vector, base_word_of_features);

    double score = database->Score(query_bow_vector, base_bow_vector);

    std::vector<cv::DMatch> matches;
    for(int query_idx = 0; query_idx < query_word_of_features.size(); query_idx++){
      DBoW2::WordId word_id = query_word_of_features[query_idx];
      if(word_id<UINT_MAX && base_word_features.count(word_id) > 0){
        for(int base_idx : base_word_features[word_id]){
          matches.emplace_back(query_idx, base_idx, 0);
        }
      }
    }

    const std::vector<cv::KeyPoint>& base_keypoints = database_frame->GetAllKeypoints();


    {
      std::vector<cv::Point> points1, points2;
      for(const cv::DMatch& match : matches){
        points1.push_back(query_keypoints[match.queryIdx].pt);
        points2.push_back(base_keypoints[match.trainIdx].pt);
      }

      std::vector<uchar> inliers;
      cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 20, 0.99, inliers);
      int j = 0;
      for(int i = 0; i < matches.size(); i++){
        if(inliers[i]){
          matches[j++] = matches[i];
        }
      }
      matches.resize(j);
    }


    cv::Mat base_image = read_image(database_frame);
    std::vector<Eigen::Vector4d> base_lines = database_frame->GatAllLines();
    std::vector<std::map<int, double>> base_points_on_line = database_frame->GetPointsOnLines();
    std::vector<int> base_line_ids(base_lines.size());
    std::iota(base_line_ids.begin(), base_line_ids.end(), 1);
    cv::Mat base_drawed_image = DrawLinePointRelation(base_image, base_features, base_lines, base_points_on_line, base_line_ids);


    std::string image_save_name = DoubleTimeToString(database_frame->GetTimestamp()) + ".png";
    std::string image_save_path = ConcatenateFolderAndFileName(save_root, image_save_name);

    cv::Mat save_image;
    cv::drawMatches(query_drawed_image, query_keypoints, base_drawed_image, base_keypoints, matches, save_image);
    std::string text = "Matching : " + std::to_string(matches.size()) + ", Score : " + std::to_string(score);

    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 1.0;
    int thickness = 2;
    cv::putText(save_image, text, cv::Point(10, 30), fontFace, fontScale, cv::Scalar(0, 0, 255), thickness);
    cv::imwrite(image_save_path, save_image); 
  }
}


void DrawDbowJunctionMatchingResults(FramePtr query_frame, cv::Mat& query_image, FramePtr database_frame, 
    std::vector<std::vector<bool>>& match_matrix, std::string image_root, std::string save_root){
  std::vector<std::string> image_names;
  GetFileNames(image_root, image_names);
  std::sort(image_names.begin(), image_names.end());
  std::function<cv::Mat(FramePtr)> read_image = [&](FramePtr frame_){
    double timestamp = frame_->GetTimestamp();
    std::string frame_image_name = "";
    double dt = 100;
    for(std::string image_name : image_names){
      double image_time = ImageNameToTime(image_name);
      double time_diff = std::abs(image_time-timestamp);
      if(time_diff < dt){
        frame_image_name = image_name;
        dt = time_diff;
      }
    }

    if(dt > 0.1){
      return cv::Mat();
    }

    std::string image_path = ConcatenateFolderAndFileName(image_root, frame_image_name);
    cv::Mat image = cv::imread(image_path, 0);

    cv::Mat image_rect;
    CameraPtr camera = frame_->GetCamera();
    camera->UndistortImage(image, image_rect);

    return image_rect;
  };

  std::function<cv::Mat(cv::Mat&, std::vector<Eigen::Vector4d>&)> draw_lines = [&](cv::Mat& image, std::vector<Eigen::Vector4d>& lines){
    cv::Mat img_color;
    cv::cvtColor(image, img_color, cv::COLOR_GRAY2RGB);
    for(size_t i = 0; i < lines.size(); i++){
      Eigen::Vector4d line = lines[i];
      cv::line(img_color, cv::Point2i((int)(line(0)+0.5), (int)(line(1)+0.5)), 
          cv::Point2i((int)(line(2)+0.5), (int)(line(3)+0.5)), cv::Scalar(0, 255, 0), 2);
    }
    return img_color;
  };

  const Eigen::Matrix<float, 259, Eigen::Dynamic>& query_features = query_frame->GetJunctions();
  std::vector<cv::KeyPoint> query_points;
  for(int i = 0; i < query_features.cols(); i++){
    float x = query_features(1, i);
    float y = query_features(2, i);
    query_points.emplace_back(x, y, 8, -1, 1);
  }
  std::vector<Eigen::Vector4d> query_lines = query_frame->GatAllLines();
  cv::Mat query_drawed_image = draw_lines(query_image, query_lines); 


  const Eigen::Matrix<float, 259, Eigen::Dynamic>& base_features = database_frame->GetJunctions();
  std::vector<cv::KeyPoint> base_points;
  for(int i = 0; i < base_features.cols(); i++){
    float x = base_features(1, i);
    float y = base_features(2, i);
    base_points.emplace_back(x, y, 8, -1, 1);
  }
  std::vector<Eigen::Vector4d> base_lines = database_frame->GatAllLines();
  cv::Mat base_image = read_image(database_frame);
  cv::Mat base_drawed_image = draw_lines(base_image, base_lines); 



  std::vector<cv::DMatch> matches;
  for(int i = 0; i < match_matrix.size(); i++){ // query
    for(int j = 0; j < match_matrix[i].size(); j++){ // base
      if(match_matrix[i][j]){
        matches.emplace_back(i, j, 0);
      }
    }
  }

  std::string image_save_name = DoubleTimeToString(database_frame->GetTimestamp()) + ".png";
  std::string image_save_path = ConcatenateFolderAndFileName(save_root, image_save_name);

  int max_query = 0;
  int max_base = 0;
  for(const auto& match : matches){
    max_query = std::max(max_query, match.queryIdx);
    max_base = std::max(max_base, match.trainIdx);
  }


  cv::Mat save_image;
  cv::drawMatches(query_drawed_image, query_points, base_drawed_image, base_points, matches, save_image);
  std::string text = "Matching : " + std::to_string(matches.size());

  int fontFace = cv::FONT_HERSHEY_SIMPLEX;
  double fontScale = 1.0;
  int thickness = 2;
  cv::putText(save_image, text, cv::Point(10, 30), fontFace, fontScale, cv::Scalar(0, 0, 255), thickness);
  cv::imwrite(image_save_path, save_image); 
}