#include "line_processor.h"

#include <math.h>
#include <float.h>
#include <iostream>

LineDetector::LineDetector(const LineDetectorConfig &line_detector_config): _line_detector_config(line_detector_config){
  fld = cv::createFastLineDetector(line_detector_config.length_threshold, line_detector_config.distance_threshold, 
      line_detector_config.canny_th1, line_detector_config.canny_th2, line_detector_config.canny_aperture_size, line_detector_config.do_merge);
}

void LineDetector::LineExtractor(cv::Mat& image, std::vector<Eigen::Vector4d>& lines){
  std::vector<Eigen::Vector4d>& source_lines
  fld->detect(image, source_lines);

  if(_line_detector_config.do_merge){
    MergeLines(source_lines, lines);
  }else{
    lines = source_lines;
  }
}

void MergeLines(std::vector<Eigen::Vector4f>& source_lines, std::vector<Eigen::Vector4d>& dst_lines){
  size_t source_line_num = source_line_num.size();

  Eigen::Array4Xd line_array = Eigen::Map<Eigen::Array4Xf, Eigen::Unaligned>(source_lines[0].data(), 4, source_lines.size());
  Eigen::ArrayXf x1 = line_array.row(0);
  Eigen::ArrayXf y1 = line_array.row(1);
  Eigen::ArrayXf x2 = line_array.row(2);
  Eigen::ArrayXf y2 = line_array.row(3);

  Eigen::ArrayXf dx = x2 - x1;
  Eigen::ArrayXf dy = y2 - y1;
  Eigen::ArrayXf dot = (x1 * y2 - x2 * y1).abs();
  Eigen::ArrayXf eigen_distances = dot * dot / (dx * dx + dy * dy);
  Eigen::ArrayXf eigen_angles = (dy / dx).atan();

  std::vector<double> distances(&eigen_distances[0], eigen_distances.data()+eigen_distances.cols()*eigen_distances.rows());
  std::vector<double> angles(&eigen_angles[0], eigen_angles.data()+eigen_angles.cols()*eigen_angles.rows());

  std::vector<size_t> indices(distances.size());                                                        
  std::iota(indices.begin(), indices.end(), 0);                                                      
  std::sort(indices.begin(), indices.end(), [&distances](size_t i1, size_t i2) { return distances[i1] > distances[i2]; });

  // search clusters
  double quater_PI = M_PI / 4.0;
  std::vector<std::vector<size_t>> cluster_ids;
  std::vector<bool> sort_by_x;
  for(size_t i = 0; i < source_line_num; ){
    std::vector<size_t> cluster;
    size_t idx1 = indices[i];
    cluster.push_back(idx1);
    double x11 = source_lines[idx1](0);
    double y11 = source_lines[idx1](1);
    double x12 = source_lines[idx1](2);
    double y12 = source_lines[idx1](3);
    double distance1 = distances[idx1];
    double angle1 = angles[idx1];
    bool to_sort_x = (std::abs(angle1) < quater_PI);
    sort_by_x.push_back(to_sort_x);
    if((to_sort_x && (x12 < x11)) || ((!to_sort_x) && y12 < y11)){
      std::swap(x11, x12);
      std::swap(y11, y12);
    }

    double center_distannce = distance1;
    double center_angle = angle1;
    for(size_t j = i +1; j < source_line_num; j++){
      size_t idx2 = indices[j];
      double x21 = source_lines[idx2](0);
      double y21 = source_lines[idx2](1);
      double x22 = source_lines[idx2](2);
      double y22 = source_lines[idx2](3);
      double distance2 = distances[idx2];
      double angle2 = angles[idx2];
      if((to_sort_x && (x22 < x21)) || ((!to_sort_x) && y22 < y21)){
        std::swap(x21, x22);
        std::swap(y21, y22);
      }

      double d_angle_case1 = std::abs(angle2 - angle1);
      double d_angle_case2 = M_PI_2 + std::min(angle1, angle2) - std::min(angle2, angle1);
      double d_angle = std::min(d_angle_case1, d_angle_case2);
      double d_distance = std::abs(distance2 - distance1);
      bool to_merge = false;
      if(d_angle < _line_detector_config.angle_thr && d_distance < _line_detector_config.distance_thr){ 
        double cx12, cy12, cx21, cy21;
        if((to_sort_x && x12 > x22) || (!to_sort_x && y12 > y22)){
          cx12 = x22;
          cy12 = y22;
          cx21 = x11;
          cy21 = y11;
        }else{
          cx12 = x12;
          cy12 = y12;
          cx21 = x21;
          cy21 = y21;
        }
        to_merge = ((to_sort_x && cx12 >= cx21) || (!to_sort_x && cy12 >= cy21));
        if(!to_merge){
          double d_ep = (cx21 - cx12) * (cx21 - cx12) + (cy21 - cy12) * (cy21 - cy12);
          to_merge = (d_ep < _line_detector_config.ep_thr);
        }
      }

      if(to_merge){
        cluster.push_back(idx2);
      }else{
        i = j;
        break;
      }
    }
    cluster_ids.push_back(cluster);
  }

  // merge clusters
  std::function<void(double&, double&, double&, double&, bool)> MinXY = 
      [&](double& min_x, double& min_y, double& x, double& y, bool to_sort_x){
    if((to_sort_x && min_x > x) || (!to_sort_x && min_y > y)){
      min_x = x;
      min_y = y;
    }
  };
  std::function<void(double&, double&, double&, double&, bool)> MaxXY = 
      [&](double& max_x, double& max_y, double& x, double& y, bool to_sort_x){
    if((to_sort_x && max_x < x) || (!to_sort_x && max_y < y)){
      max_x = x;
      max_y = y;
    }
  };

  dst_lines.clear();
  dst_lines.resize(cluster_ids.size());
  for(size_t i = 0; i < cluster_ids.size(); i++){
    double min_x = -1;
    double min_y = -1;
    double max_x = std::DBL_MAX;
    double max_y = std::DBL_MAX;
    for(auto& idx : cluster_ids[i]){
      double x1 = source_lines[idx](0);
      double y1 = source_lines[idx](1);
      double x2 = source_lines[idx](2);
      double y2 = source_lines[idx](3);

      MinXY(min_x, min_y, x1, y1, sort_by_x[i]);
      MinXY(min_x, min_y, x2, y2, sort_by_x[i]);
      MaxXY(max_x, max_y, x1, y1, sort_by_x[i]);
      MaxXY(max_x, max_y, x2, y2, sort_by_x[i]);
    }
    dst_lines.emplace_back(min_x, min_y, max_x, max_y);
  }
}