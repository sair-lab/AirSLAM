#include "optimization_3d/optimization_3d.h"

#include <fstream>
#include <iostream>
#include <string>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "camera.h"
#include "frame.h"
#include "mappoint.h"
#include "optimization_3d/types.h"
#include "optimization_3d/visual_error_term.h"
#include "timer.h"

INITIALIZE_TIMER;

int AddVisualErrorTerm(
    MapOfPoses& poses, MapOfPoints3d& points, std::vector<CameraPtr>& camera_list, 
    VectorOfPointConstraints& point_constraints, ceres::Problem* problem, 
    ceres::LocalParameterization* quaternion_local_parameterization){

  // double huber_loss_delta = 10.0;
  // double image_point_uncertainty = 0.8;
  // std::shared_ptr<ceres::LossFunction> loss_function(new ceres::LossFunctionWrapper(
  //     new ceres::HuberLoss(huber_loss_delta * image_point_uncertainty), ceres::TAKE_OWNERSHIP));
  ceres::LossFunction* loss_function = NULL;

  int num_visual_error_terms = 0;
  int mono_edge_num = 0;
  int stereo_edge_num = 0;
  for(PointConstraint& constraint : point_constraints){
    MapOfPoses::iterator pose_iter = poses.find(constraint.id_pose);
    CHECK(pose_iter != poses.end())
        << "Pose with ID: " << constraint.id_pose << " not found.";
    MapOfPoints3d::iterator point_iter = points.find(constraint.id_point);
    CHECK(point_iter != points.end())
        << "Pose with ID: " << constraint.id_point << " not found.";
    CHECK_LT(constraint.id_camera, camera_list.size());
    CameraPtr camera = camera_list[constraint.id_camera];

    ceres::CostFunction* cost_function;
    if(constraint.keypoint[2] > 0){
      cost_function = StereoPointReprojectionErrorTerm<Camera>::Create(
          constraint.keypoint, constraint.pixel_sigma, camera.get());
      stereo_edge_num++;
    }else{
      cost_function = MonoPointReprojectionErrorTerm<Camera>::Create(
          constraint.keypoint.head(2), constraint.pixel_sigma, camera.get());
      mono_edge_num++;
    }

    problem->AddResidualBlock(cost_function,
                              // loss_function.get(),
                              loss_function,
                              pose_iter->second.p.data(),
                              pose_iter->second.q.coeffs().data(),
                              point_iter->second.p.data());  

    problem->SetParameterization(pose_iter->second.q.coeffs().data(), quaternion_local_parameterization);
    num_visual_error_terms++;
  }
  std::cout << "mono_edge_num = " << mono_edge_num << std::endl;
  std::cout << "stereo_edge_num = " << stereo_edge_num << std::endl;
  return num_visual_error_terms;
}

void BuildOptimizationProblem(
    MapOfPoses& poses, MapOfPoints3d& points, std::vector<CameraPtr>& camera_list,
    VectorOfPointConstraints& point_constraints, std::vector<int>& fixed_poses, 
    std::vector<int>& fixed_points, ceres::Problem* problem){
  ceres::LocalParameterization* quaternion_local_parameterization =
      new ceres::EigenQuaternionParameterization;

  int num_visual_edge = AddVisualErrorTerm(poses, points, camera_list, point_constraints, 
      problem, quaternion_local_parameterization);

  std::cout << "Add " << num_visual_edge << " visual edges in total" << std::endl;
  
  for(int id : fixed_poses){
    MapOfPoses::iterator pose_iter = poses.find(id);
    if(pose_iter == poses.end()) continue;
    problem->SetParameterBlockConstant(pose_iter->second.p.data());
    problem->SetParameterBlockConstant(pose_iter->second.q.coeffs().data());
  }

  for(int id : fixed_points){
    MapOfPoints3d::iterator point_iter = points.find(id);
    if(point_iter == points.end()) continue;
    problem->SetParameterBlockConstant(point_iter->second.p.data());
  }
}

bool SolveOptimizationProblem(ceres::Problem* problem) {
  CHECK(problem != NULL);

  ceres::Solver::Options options;
  options.max_num_iterations = 3;
  options.max_solver_time_in_seconds = 0.05;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

  ceres::Solver::Summary summary;
  ceres::Solve(options, problem, &summary);

  std::cout << summary.FullReport() << '\n';
  return summary.IsSolutionUsable();
}

int Optimize(MapOfPoses& poses, MapOfPoints3d& points, 
    std::vector<CameraPtr> camera_list, VectorOfPointConstraints& point_constraints, 
    std::vector<int>& fixed_poses, std::vector<int>& fixed_points, std::vector<int>& inliers){
  ceres::Problem problem;
  START_TIMER;
  BuildOptimizationProblem(poses, points, camera_list, 
      point_constraints, fixed_poses, fixed_points, &problem);
  STOP_TIMER("BuildOptimizationProblem Time");
  START_TIMER;

  SolveOptimizationProblem(&problem);
  STOP_TIMER("SolveOptimizationProblem Time");
  return points.size();
}

int SolvePnPWithCV(FramePtr frame, std::vector<MappointPtr>& mappoints, 
    Eigen::Matrix4d& pose, std::vector<int>& inliers){
  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;
  std::vector<int> point_indexes;
  cv::Mat camera_matrix, dist_coeffs;
  CameraPtr camera = frame->GetCamera();
  camera->GetCamerMatrix(camera_matrix);
  camera->GetDistCoeffs(dist_coeffs);
  cv::Mat rotation_vector;
  cv::Mat translation_vector;
  cv::Mat cv_inliers;

  for(size_t i = 0; i < mappoints.size(); i++){
    MappointPtr mpt = mappoints[i];
    if(mpt == nullptr || !mpt->IsValid()) continue;
    Eigen::Vector3d keypoint; 
    if(!frame->GetKeypointPosition(i, keypoint)) continue;
    const Eigen::Vector3d& point_position = mpt->GetPosition();
    object_points.emplace_back(point_position(0), point_position(1), point_position(2));
    image_points.emplace_back(keypoint(0), keypoint(1));
    point_indexes.emplace_back(i);
  }

  cv::solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs, 
      rotation_vector, translation_vector, false, 100, 20.0, 0.99, cv_inliers);

  cv::Mat cv_Rcw;
  cv::Rodrigues(rotation_vector, cv_Rcw);
  Eigen::Matrix3d eigen_Rcw;
  Eigen::Vector3d eigen_tcw;
  cv::cv2eigen(cv_Rcw, eigen_Rcw);
  cv::cv2eigen(translation_vector, eigen_tcw);
  Eigen::Matrix3d eigen_Rwc = eigen_Rcw.transpose();
  pose.block<3, 3>(0, 0) = eigen_Rwc;
  pose.block<3, 1>(0, 3) = eigen_Rwc * (-eigen_tcw);

  inliers = std::vector<int>(mappoints.size(), -1);
  for(int i = 0; i < cv_inliers.rows; i++){
    int inlier_idx = cv_inliers.at<int>(i, 0);
    int point_idx = point_indexes[inlier_idx];
    inliers[point_idx] = mappoints[point_idx]->GetId();
  }
  return cv_inliers.rows;
}
