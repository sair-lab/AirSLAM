#include "line_processor.h"

#include <math.h>
#include <float.h>
#include <iostream>
#include <numeric>

#include "camera.h"
#include "timer.h"

float PointLineDistance(Eigen::Vector4f line, Eigen::Vector2f point){
  float x0 = point(0);
  float y0 = point(1);
  float x1 = line(0);
  float y1 = line(1);
  float x2 = line(2);
  float y2 = line(3);
  float d = (std::fabs((y2 - y1) * x0 +(x1 - x2) * y0 + ((x2 * y1) -(x1 * y2)))) / (std::sqrt(std::pow(y2 - y1, 2) + std::pow(x1 - x2, 2)));
  return d;
}

double CVPointLineDistance3D(const std::vector<cv::Point3f> points, const cv::Vec6f& line, std::vector<float>& dist){
  float px = line[3], py = line[4], pz = line[5];
  float vx = line[0], vy = line[1], vz = line[2];
  dist.resize(points.size());
  double sum_dist = 0.;
  float x, y, z;
  Eigen::Vector3f p;

  for(int j = 0; j < points.size(); j++){
    x = points[j].x - px;
    y = points[j].y - py;
    z = points[j].z - pz;

   // cross 
    p(0) = vy * z - vz * y;
    p(1) = vz * x - vx * z;
    p(2) = vx * y - vy * x;

    dist[j] = p.norm();
    sum_dist += dist[j];
  }

  return sum_dist;
}

void EigenPointLineDistance3D(
    const std::vector<Eigen::Vector3d>& points, const Vector6d& line, std::vector<double>& dist){
  Eigen::Vector3d pl = line.head(3);
  Eigen::Vector3d v = line.tail(3);
  Eigen::Vector3d dp;
  Eigen::Vector3d p;
  size_t point_num = points.size();
  dist.resize(point_num);
  for(size_t i = 0; i < point_num; i++){
    dp = points[i] - pl;
    p = v.cross(dp);
    dist[i] = p.norm();
  }
}

float AngleDiff(float& angle1, float& angle2){
  float d_angle_case1 = std::abs(angle2 - angle1);
  float d_angle_case2 = M_PI + std::min(angle1, angle2) - std::max(angle1, angle2);
  return std::min(d_angle_case1, d_angle_case2);
}

void AssignPointsToLines(std::vector<Eigen::Vector4d>& lines, Eigen::Matrix<float, 259, Eigen::Dynamic>& points, 
    std::vector<std::map<int, double>>& relation){
  Eigen::Array2Xd point_array = points.middleRows(1, 2).array().cast<double>();
  Eigen::Array4Xd line_array = Eigen::Map<Eigen::Array4Xd, Eigen::Unaligned>(lines[0].data(), 4, lines.size());

  Eigen::ArrayXd x = point_array.row(0);
  Eigen::ArrayXd y = point_array.row(1); 

  Eigen::ArrayXd x1 = line_array.row(0);
  Eigen::ArrayXd y1 = line_array.row(1);
  Eigen::ArrayXd x2 = line_array.row(2);
  Eigen::ArrayXd y2 = line_array.row(3);

  Eigen::ArrayXd A = y2 - y1;
  Eigen::ArrayXd B = x1 - x2;
  Eigen::ArrayXd C = x2 * y1 - x1 * y2;
  Eigen::ArrayXd D = (A.square() + B.square()).sqrt();

  relation.clear();
  relation.reserve(lines.size());
  for(int i = 0, line_num = lines.size(); i < line_num; i++){
    std::map<int, double> points_on_line;
    for(int j = 0, point_num = points.cols(); j < point_num; j++){
      // filter by x, y
      double lx1 = x1(i);
      double ly1 = y1(i);
      double lx2 = x2(i);
      double ly2 = y2(i);
      double px = x(j);
      double py = y(j);
      
      double min_lx = lx1;
      double max_lx = lx2;
      double min_ly = ly1;
      double max_ly = ly2;
      if(lx1 > lx2) std::swap(min_lx, max_lx);
      if(ly1 > ly2) std::swap(min_ly, max_ly);
      if(px < min_lx - 3 || px > max_lx + 3 || py < min_ly - 3 || py > max_ly + 3) continue;

      // check distance
      float pl_distance = std::abs((A(i) * px + B(i) * py + C(i))) / D(i);
      if(pl_distance > 3) continue;

      double side1 = std::pow((lx1 - px), 2) + std::pow((ly1 - py), 2);
      double side2 = std::pow((lx2 - px), 2) + std::pow((ly2 - py), 2);
      double line_side = std::pow(D(i), 2);
      if(side1 <= 9 || side2 <= 9 || ((side1 < line_side + side2) && (side2 < line_side + side1))){
        points_on_line[j] = pl_distance;
      }
    }
    relation.push_back(points_on_line);
  }
}

void MatchLines(const std::vector<std::map<int, double>>& points_on_line0, 
    const std::vector<std::map<int, double>>& points_on_line1, const std::vector<cv::DMatch>& point_matches, 
    size_t point_num0, size_t point_num1, std::vector<int>& line_matches){
  size_t line_num0 = points_on_line0.size();
  size_t line_num1 = points_on_line1.size();
  line_matches.clear();
  line_matches.resize(line_num0);
  for(size_t i = 0; i < line_num0; i++){
    line_matches[i] = -1;
  }
  if(point_num0 == 0 || point_num1 == 0 || line_num0 == 0 || line_num1 == 0) return;

  std::vector<std::vector<int>> assigned_lines0, assigned_lines1;
  assigned_lines0.resize(point_num0);
  assigned_lines1.resize(point_num1);
  for(size_t i = 0; i < points_on_line0.size(); i++){
    for(auto& kv : points_on_line0[i]){
      assigned_lines0[kv.first].push_back(i);
    }
  }
  
  for(size_t i = 0; i < points_on_line1.size(); i++){
    for(auto& kv : points_on_line1[i]){
      assigned_lines1[kv.first].push_back(i);
    }
  }

  // fill in matching matrix
  Eigen::MatrixXi matching_matrix = Eigen::MatrixXi::Zero(line_num0, line_num1);
  for(auto& point_match : point_matches){
    int idx0 = point_match.queryIdx;
    int idx1 = point_match.trainIdx;

    for(auto& l0 : assigned_lines0[idx0]){
      for(auto& l1 : assigned_lines1[idx1]){
        matching_matrix(l0, l1) += 1;
      }
    }
  }

  // find good matches
  int line_match_num = 0;
  std::vector<int> row_max_value(line_num0), col_max_value(line_num1);
  std::vector<Eigen::VectorXi::Index> row_max_location(line_num0), col_max_location(line_num1);
  for(size_t i = 0; i < line_num0; i++){
    row_max_value[i] = matching_matrix.row(i).maxCoeff(&row_max_location[i]);
  }
  for(size_t j = 0; j < line_num1; j++){
    Eigen::VectorXi::Index col_max_location;
    int col_max_val = matching_matrix.col(j).maxCoeff(&col_max_location);
    if(col_max_val < 2 || row_max_location[col_max_location] != j) continue;

    float score = (float)(col_max_val * col_max_val) / std::min(points_on_line0[col_max_location].size(), points_on_line1[j].size());
    if(score < 0.8) continue;

    line_matches[col_max_location] = j;
    line_match_num++;
  }
}

void SortPointsOnLine(std::vector<Eigen::Vector2d>& points, std::vector<size_t>& order, bool sort_by_x){
  size_t num_points = points.size();
  if(num_points < 1) return;

  order.clear();
  order.resize(num_points);
  std::iota(order.begin(), order.end(), 0);       
  if(sort_by_x){
    std::sort(order.begin(), order.end(), [&points](size_t i1, size_t i2) { return points[i1](0) < points[i2](0); });
  }else{
    std::sort(order.begin(), order.end(), [&points](size_t i1, size_t i2) { return points[i1](1) < points[i2](1); });
  }                                
}

bool TriangulateByStereo(const Eigen::Vector4d& line_left, const Eigen::Vector4d& line_right, 
    const Eigen::Matrix4d& Twc, const CameraPtr& camera, Vector6d& line_3d){
  double x11 = line_left(0);
  double y11 = line_left(1);
  double x12 = line_left(2);
  double y12 = line_left(3);

  double x21 = line_right(0);
  double y21 = line_right(1);
  double x22 = line_right(2);
  double y22 = line_right(3);

  double dx_left = x12 - x11;
  double dy_left = y12 - y11;
  double angle_left = std::atan(dy_left / dx_left); 
  if(std::abs(dy_left) <= 3 || std::abs(angle_left) < 0.175) return false;    // horizontal line

  double dx_right = x22 - x21;
  double dy_right = y22 - y21;
  double angle_right = std::atan(dy_right / dx_right); 
  if(std::abs(dy_right) <= 3 || std::abs(angle_right) < 0.175) return false;    // horizontal line

  double k_inv_right = dx_right / dy_right;
  double x11_right = x21 + k_inv_right * (y11 - y21);
  double x12_right = x21 + k_inv_right * (y12 - y21);

  Eigen::Vector3d point_2d1, point_2d2;
  Eigen::Vector3d point_3d1, point_3d2;

  point_2d1 << x11, y11, x11_right;
  point_2d2 << x12, y12, x12_right;

  double dx1 = point_2d1(0) - point_2d1(2);
  double dx2 = point_2d2(0) - point_2d2(2);
  double min_x_diff = camera->MinXDiff();
  double max_x_diff = camera->MaxXDiff();
  if(dx1 < min_x_diff || dx1 > max_x_diff || dx2 < min_x_diff || dx2 > max_x_diff) return false;

  camera->BackProjectStereo(point_2d1, point_3d1);
  camera->BackProjectStereo(point_2d2, point_3d2);

  // form camera to world
  Eigen::Matrix3d Rwc = Twc.block<3, 3>(0, 0);
  Eigen::Vector3d twc = Twc.block<3, 1>(0, 3);
  point_3d1 = Rwc * point_3d1 + twc;
  point_3d2 = Rwc * point_3d2 + twc;
  line_3d << point_3d1, point_3d2;

  return true;
}

bool CompoutePlaneFromPoints(const Eigen::Vector3d& point1, const Eigen::Vector3d& point2, 
    const Eigen::Vector3d& point3, Eigen::Vector4d& plane){
  Eigen::Vector3d line12 = point2 - point1;
  Eigen::Vector3d line13 = point3 - point1;
  Eigen::Vector3d n = line12.cross(line13);
  plane.head(3) = n.normalized();
  plane(3) = - n.transpose() * point1;
  return true;
}

bool ComputeLineFramePlanes(const Eigen::Vector4d& plane1, const Eigen::Vector4d& plane2, Line3DPtr line_3d){
  Eigen::Vector3d n1 = plane1.head(3);
  Eigen::Vector3d n2 = plane2.head(3);

  double cos_theta = n1.transpose() * n2;
  cos_theta /= (n1.norm() * n2.norm());

  // cos10 = cos170 = 0.9848
  // if(std::abs(cos_theta) > 0.9848) return false;

  Eigen::Vector3d d = n1.cross(n2);
  Eigen::Vector3d w = plane2(3) * n1 - plane1(3) * n2;
  line_3d->setD(d);
  line_3d->setW(w);
  line_3d->normalize();
  return true;
}

bool TriangulateByTwoFrames(const Eigen::Vector4d& line_2d1, const Eigen::Matrix4d& pose1, 
    const Eigen::Vector4d& line_2d2, const Eigen::Matrix4d& pose2, const CameraPtr& camera, Line3DPtr line_3d){
  Eigen::Matrix3d Rw1 = pose1.block<3, 3>(0, 0);
  Eigen::Vector3d tw1 = pose1.block<3, 1>(0, 3);
  Eigen::Matrix3d Rw2 = pose2.block<3, 3>(0, 0);
  Eigen::Vector3d tw2 = pose2.block<3, 1>(0, 3);

  Eigen::Matrix3d R12 = Rw1.transpose() * Rw2;
  Eigen::Vector3d t12 = Rw1.transpose() * (tw2 - tw1);

  Eigen::Vector4d plane1, plane2;
  Eigen::Vector3d point11, point12, point13;
  camera->BackProjectMono(line_2d1.head(2), point11);
  camera->BackProjectMono(line_2d1.tail(2), point12);
  point13 << 0.0, 0.0, 0.0;
  if(!CompoutePlaneFromPoints(point11, point12, point13, plane1)) return false;

  Eigen::Vector3d point21, point22;
  camera->BackProjectMono(line_2d2.head(2), point21);
  camera->BackProjectMono(line_2d2.tail(2), point22);

  point21 = R12 * point21 + t12;
  point22 = R12 * point22 + t12;
  if(!CompoutePlaneFromPoints(point21, point22, t12, plane2)) return false;

  bool success = ComputeLineFramePlanes(plane1, plane2, line_3d);

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.rotate(Rw1);
  T.pretranslate(tw1);
  g2o::Line3D line_3d_w = T * (*line_3d);
  line_3d_w.normalize();
  line_3d->setW(line_3d_w.w());
  line_3d->setD(line_3d_w.d());
  return success;
}

bool ComputeLine3DFromEndpoints(const Vector6d& endpoints, Line3DPtr line_3d){
  Eigen::Vector3d point3d1 = endpoints.head(3);
  Eigen::Vector3d point3d2 = endpoints.tail(3);

  Eigen::Vector3d l = point3d2 - point3d1;
  if(l.norm() < 0.01) return false;

  Vector6d line_cart;
  line_cart << point3d1, l;
  g2o::Line3D line = g2o::Line3D::fromCartesian(line_cart);

  line_3d->setW(line.w());
  line_3d->setD(line.d());
  return true;
}

bool Point2DTo3D(const Eigen::Vector3d& anchor_point_3d1, const Eigen::Vector3d& anchor_point_3d2, 
  	const Eigen::Vector2d& anchor_point_2d1, const Eigen::Vector2d& anchor_point_2d2, 
    const Eigen::Vector2d& p2D, Eigen::Vector3d& p3D){
  Eigen::Vector2d anchor_line2d = anchor_point_2d2 - anchor_point_2d1;
  anchor_line2d.normalize();
  size_t md = std::abs(anchor_line2d(0)) > std::abs(anchor_line2d(1)) ? 0 : 1;

  double rate = (p2D(md) - anchor_point_2d1(md)) / (anchor_point_2d2(md) - anchor_point_2d1(md));
  p3D = anchor_point_3d1 + rate * (anchor_point_3d2 - anchor_point_3d1);
  return true;
}