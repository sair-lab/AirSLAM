#include "g2o_optimization/g2o_optimization.h"

#include <fstream>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include "read_configs.h"
#include "g2o_optimization/vertex_imu.h"
#include "g2o_optimization/vertex_line3d.h"
#include "g2o_optimization/vertex_vi_pose.h"
#include "g2o_optimization/edge_imu.h"
#include "g2o_optimization/edge_project_point.h"
#include "g2o_optimization/edge_project_line.h"
#include "g2o_optimization/edge_relative_pose.h"

void AddFrameVertex(FramePtr frame, MapOfPoses& poses, int id_camera, bool fix_this_frame){
  int frame_id = frame->GetFrameId();
  Eigen::Matrix4d& frame_pose = frame->GetPose();
  Pose3d pose;
  pose.R = frame_pose.block<3, 3>(0, 0);
  pose.p = frame_pose.block<3, 1>(0, 3);
  pose.id_camera = id_camera;
  pose.fixed = fix_this_frame;
  poses.insert(std::pair<int, Pose3d>(frame_id, pose)); 
}

void AddFrameVertex(FramePtr frame, MapOfPoses& poses, int id_camera, MapOfVelocity& velocities, MapOfBias& biases, 
    VectorOfIMUConstraints& imu_constraints, bool fix_this_frame, bool add_imu_constraint, bool use_updated_bias){
  int frame_id = frame->GetFrameId();
  FramePtr last_frame = frame->PreviousFrame();
  PreinterationPtr preinteration = frame->GetIMUPreinteration();

  Eigen::Matrix4d& frame_pose = frame->GetPose();
  Pose3d pose;
  pose.R = frame_pose.block<3, 3>(0, 0);
  pose.p = frame_pose.block<3, 1>(0, 3);
  pose.id_camera = id_camera;
  pose.fixed = fix_this_frame;
  poses.insert(std::pair<int, Pose3d>(frame_id, pose)); 

  Bias bias;
  if(use_updated_bias){
    bias.gyr_bias = preinteration->bg + preinteration->dbg;
    bias.acc_bias = preinteration->ba + preinteration->dba;
  }else{
    bias.gyr_bias = preinteration->bg;
    bias.acc_bias = preinteration->ba;
  }
  bias.fixed = fix_this_frame;
  biases.insert(std::pair<int, Bias>(frame_id, bias)); 

  Velocity velocity;
  velocity.velocity = frame->GetVelocity();
  velocity.fixed = fix_this_frame;
  velocities.insert(std::pair<int, Velocity>(frame_id, velocity)); 

  if(last_frame != nullptr && add_imu_constraint){
    IMUConstraintPtr imu_constraint = std::shared_ptr<ImuConstraint>(new ImuConstraint()); 
    imu_constraint->id_pose1 = last_frame->GetFrameId();
    imu_constraint->id_pose2 = frame_id;
    imu_constraint->id_camera1 = 0;
    imu_constraint->id_camera2 = 0;
    imu_constraint->preinteration = preinteration;
    imu_constraints.emplace_back(imu_constraint);
  }
}

void LocalmapOptimization(MapOfPoses& poses, MapOfPoints3d& points, MapOfLine3d& lines, 
    MapOfVelocity& velocities, MapOfBias& biases, std::vector<CameraPtr>& camera_list, 
    VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints, 
    VectorOfMonoLineConstraints& mono_line_constraints, VectorOfStereoLineConstraints& stereo_line_constraints,
    VectorOfIMUConstraints& imu_constraints, const Eigen::Matrix3d& Rwg, const OptimizationConfig& cfg){

  // std::cout << "---------LocalmapOptimization----------" << std::endl;
  // std::cout << "poses.size = " << poses.size() << std::endl;
  // std::cout << "points.size = " << points.size() << std::endl;
  // std::cout << "lines.size = " << lines.size() << std::endl;
  // std::cout << "velocities.size = " << velocities.size() << std::endl;
  // std::cout << "biases.size = " << biases.size() << std::endl;
  // std::cout << "mono_point_constraints.size = " << mono_point_constraints.size() << std::endl;
  // std::cout << "stereo_point_constraints.size = " << stereo_point_constraints.size() << std::endl;
  // std::cout << "mono_line_constraints.size = " << mono_line_constraints.size() << std::endl;
  // std::cout << "stereo_line_constraints.size = " << stereo_line_constraints.size() << std::endl;
  // std::cout << "imu_constraints.size = " << imu_constraints.size() << std::endl;
  // std::cout << "------------------------------------" << std::endl;

  // 1. optimizer
  g2o::SparseOptimizer optimizer;
  auto linear_solver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
  g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));

  optimizer.setVerbose(false);
  optimizer.setAlgorithm(solver);

  // 2. frame vertex
  int max_frame_id = 0;
  for(auto& kv : poses){
    Eigen::Matrix4d Tcb = camera_list[kv.second.id_camera]->BodyToCamera();
    Eigen::Matrix3d Rcw = kv.second.R.transpose();
    Eigen::Vector3d tcw = -Rcw * kv.second.p;
    VIPose vi_pose(Rcw, tcw, Tcb.block<3, 3>(0, 0), Tcb.block<3, 1>(0, 3));
    VertexVIPose* frame_vertex = new VertexVIPose();
    frame_vertex->setEstimate(vi_pose);
    frame_vertex->setId(kv.first);
    frame_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(frame_vertex);
    max_frame_id = std::max(max_frame_id, kv.first);
  } 
  max_frame_id++;

  // 3. point vertex
  int max_point_id = max_frame_id;
  for(auto& kv : points){
    g2o::VertexPointXYZ* point_vertex = new g2o::VertexPointXYZ();
    point_vertex->setEstimate(kv.second.p);
    int point_id = kv.first+max_frame_id;
    point_vertex->setId((point_id));
    max_point_id = std::max(max_point_id, point_id);
    point_vertex->setMarginalized(true);
    point_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(point_vertex);
  }
  max_point_id++;

  // 4. line vertex
  int max_line_id = max_point_id;
  for(auto& kv : lines){
    g2o::VertexLine3D* line_vertex = new g2o::VertexLine3D();
    line_vertex->setEstimateData(kv.second.line_3d);
    int line_id = kv.first+max_point_id;
    max_line_id = std::max(max_line_id, line_id);
    line_vertex->setId(line_id);
    line_vertex->setMarginalized(true);
    line_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(line_vertex);
  }
  max_line_id++;

  // 5. velocity vertex
  int max_velocity_id = max_line_id;
  for(auto& kv : velocities){
    VertexVelocity* velocity_vertex = new VertexVelocity(kv.second.velocity);
    int velocity_id = kv.first + max_line_id;
    max_velocity_id = std::max(max_velocity_id, velocity_id);
    velocity_vertex->setId(velocity_id);
    velocity_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(velocity_vertex);
  }
  max_velocity_id++;

  // 6. bias vertex
  int max_bias_id = max_velocity_id;
  for(auto& kv : biases){
    VertexGyrBias* gyr_bias_vertex = new VertexGyrBias(kv.second.gyr_bias);
    int gyr_bias_id = max_velocity_id + kv.first * 2;
    gyr_bias_vertex->setId(gyr_bias_id);
    gyr_bias_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(gyr_bias_vertex);

    VertexAccBias* acc_bias_vertex = new VertexAccBias(kv.second.acc_bias);
    int acc_bias_id = max_velocity_id + kv.first * 2 + 1;
    max_bias_id = std::max(max_bias_id, acc_bias_id);
    acc_bias_vertex->setId(acc_bias_id);
    acc_bias_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(acc_bias_vertex);
  }
  max_bias_id++;

  // 7. gravity direction vertex, will be fixed in this function.
  VertexGDirection* gravity_direction_vertex = new VertexGDirection(Rwg);
  gravity_direction_vertex->setId(max_bias_id);
  gravity_direction_vertex->setFixed(true);
  optimizer.addVertex(gravity_direction_vertex);
  

  // 8. point edges
  std::vector<EdgeSE3ProjectPoint*> mono_edges; 
  mono_edges.reserve(mono_point_constraints.size());
  std::vector<EdgeSE3ProjectStereoPoint*> stereo_edges;
  stereo_edges.reserve(stereo_point_constraints.size());
  const double thHuberMonoPoint = sqrt(cfg.mono_point);
  const double thHuberStereoPoint = sqrt(cfg.stereo_point);

  // 8.1 mono point edges
  for(MonoPointConstraintPtr& mpc : mono_point_constraints){
    EdgeSE3ProjectPoint* e = new EdgeSE3ProjectPoint();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((mpc->id_point+max_frame_id))));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mpc->id_pose)));
    e->setMeasurement(mpc->keypoint);
    e->setInformation(Eigen::Matrix2d::Identity());
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(thHuberMonoPoint);
    e->fx = camera_list[mpc->id_camera]->Fx();
    e->fy = camera_list[mpc->id_camera]->Fy();
    e->cx = camera_list[mpc->id_camera]->Cx();
    e->cy = camera_list[mpc->id_camera]->Cy();

    optimizer.addEdge(e);
    mono_edges.push_back(e);
  }

  // 8.2 stereo point edges
  for(StereoPointConstraintPtr& spc : stereo_point_constraints){
    EdgeSE3ProjectStereoPoint* e = new EdgeSE3ProjectStereoPoint();

    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((spc->id_point+max_frame_id))));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(spc->id_pose)));
    e->setMeasurement(spc->keypoint);
    e->setInformation(Eigen::Matrix3d::Identity());
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(thHuberStereoPoint);
    e->fx = camera_list[spc->id_camera]->Fx();
    e->fy = camera_list[spc->id_camera]->Fy();
    e->cx = camera_list[spc->id_camera]->Cx();
    e->cy = camera_list[spc->id_camera]->Cy();
    e->bf = camera_list[spc->id_camera]->BF();

    optimizer.addEdge(e);
    stereo_edges.push_back(e);
  }

  // 9. line edges
  std::vector<EdgeSE3ProjectLine*> mono_line_edges; 
  mono_line_edges.reserve(mono_line_constraints.size());
  std::vector<EdgeStereoSE3ProjectLine*> stereo_line_edges;
  stereo_line_edges.reserve(stereo_line_constraints.size());
  const double thHuberMonoLine = sqrt(cfg.mono_line);
  const double thHuberStereoLine = sqrt(cfg.stereo_line);

  // 9.1 mono line edges
  for(MonoLineConstraintPtr& mlc : mono_line_constraints){
    EdgeSE3ProjectLine* e = new EdgeSE3ProjectLine();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((mlc->id_line+max_point_id))));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mlc->id_pose)));
    e->setMeasurement(mlc->line_2d);
    e->setInformation(Eigen::Matrix2d::Identity() * mlc->pixel_sigma);
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(thHuberMonoLine);
    double fx = camera_list[mlc->id_camera]->Fx();
    double fy = camera_list[mlc->id_camera]->Fy();
    double cx = camera_list[mlc->id_camera]->Cx();
    double cy = camera_list[mlc->id_camera]->Cy();
    e->fx = fx;
    e->fy = fy;
    e->Kv << -fy * cx, -fx * cy, fx * fy;
    optimizer.addEdge(e);
    mono_line_edges.push_back(e);
  }

  // 9.2 stereo line edges
  for(StereoLineConstraintPtr& slc : stereo_line_constraints){
    EdgeStereoSE3ProjectLine* e = new EdgeStereoSE3ProjectLine();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((slc->id_line+max_point_id))));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(slc->id_pose)));
    e->setMeasurement(slc->line_2d);
    e->setInformation(Eigen::Matrix4d::Identity() * slc->pixel_sigma);
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(thHuberStereoLine);
    double fx = camera_list[slc->id_camera]->Fx();
    double fy = camera_list[slc->id_camera]->Fy();
    double cx = camera_list[slc->id_camera]->Cx();
    double cy = camera_list[slc->id_camera]->Cy();
    double bf = camera_list[slc->id_camera]->BF();
    e->fx = fx;
    e->fy = fy;
    e->b = bf / fx;
    e->Kv << -fy * cx, -fx * cy, fx * fy;
    optimizer.addEdge(e);
    stereo_line_edges.push_back(e);
  }

  // 10 imu edges
  std::vector<EdgeIMU*> imu_edges; 
  imu_edges.reserve(imu_constraints.size());
  std::vector<EdgeGyr*> gyr_edges;
  std::vector<EdgeAcc*> acc_edges;

  for(IMUConstraintPtr& ipc : imu_constraints){
    // 10.1 pose and velocity edges
    EdgeIMU* e_imu = new EdgeIMU(ipc->preinteration);
    
    g2o::HyperGraph::Vertex *vp1 = optimizer.vertex(ipc->id_pose1);
    g2o::HyperGraph::Vertex *vv1 = optimizer.vertex(max_line_id + ipc->id_pose1);
    g2o::HyperGraph::Vertex *vg1 = optimizer.vertex(max_velocity_id + ipc->id_pose1 * 2);
    g2o::HyperGraph::Vertex *va1 = optimizer.vertex(max_velocity_id + ipc->id_pose1 * 2 + 1);
    g2o::HyperGraph::Vertex *vp2 = optimizer.vertex(ipc->id_pose2);
    g2o::HyperGraph::Vertex *vv2 = optimizer.vertex(max_line_id + ipc->id_pose2);
    g2o::HyperGraph::Vertex *vg2 = optimizer.vertex(max_velocity_id + ipc->id_pose2 * 2);
    g2o::HyperGraph::Vertex *va2 = optimizer.vertex(max_velocity_id + ipc->id_pose2 * 2 + 1);
    g2o::HyperGraph::Vertex *vG = optimizer.vertex(max_bias_id);
    if(!vp1 || !vv1 || !vg1 || !va1 || !vp2 || !vv2 || !vg2 || !va2  || !vG) continue;

    e_imu->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vp1));
    e_imu->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vv1));
    e_imu->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg2));
    e_imu->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va2));
    e_imu->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vp2));
    e_imu->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vv2));
    e_imu->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vG));

    if(poses[ipc->id_pose1].fixed || poses[ipc->id_pose2].fixed || false){
      g2o::RobustKernelHuber *rki = new g2o::RobustKernelHuber;
      e_imu->setRobustKernel(rki);
      e_imu->setInformation(e_imu->information() * 1e-2);
      rki->setDelta(sqrt(16.92));
    }
    optimizer.addEdge(e_imu);
    imu_edges.push_back(e_imu);

    // bias edges
    EdgeGyr* e_gyr = new EdgeGyr();
    EdgeAcc* e_acc = new EdgeAcc();

    e_gyr->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg1));
    e_acc->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va1));
    e_gyr->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg2));
    e_acc->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va2));

    Eigen::Matrix3d info_g = ipc->preinteration->Cov.block<3,3>(9,9).inverse();
    Eigen::Matrix3d info_a = ipc->preinteration->Cov.block<3,3>(12,12).inverse();
    e_gyr->setInformation(info_g);
    e_acc->setInformation(info_a);

    optimizer.addEdge(e_gyr);
    optimizer.addEdge(e_acc);
    gyr_edges.push_back(e_gyr);
    acc_edges.push_back(e_acc);
  }

  // solve 
  optimizer.initializeOptimization();
  optimizer.optimize(5);

  // check inlier observations
  for(size_t i=0; i < mono_edges.size(); i++){
    EdgeSE3ProjectPoint* e = mono_edges[i];
    if(e->chi2() > cfg.mono_point || !e->isDepthPositive()){
      e->setLevel(1);
    }
    e->setRobustKernel(0);
  }

  for(size_t i=0; i < stereo_edges.size(); i++){    
    EdgeSE3ProjectStereoPoint* e = stereo_edges[i];
    if(e->chi2() > cfg.stereo_point || !e->isDepthPositive()){
        e->setLevel(1);
    }
    e->setRobustKernel(0);
  }

  for(size_t i=0; i < mono_line_edges.size(); i++){
    EdgeSE3ProjectLine* e = mono_line_edges[i];
    if(e->chi2() > cfg.mono_line){
      e->setLevel(1);
    }
    e->setRobustKernel(0);
  }

  for(size_t i=0; i < stereo_line_edges.size(); i++){    
    EdgeStereoSE3ProjectLine* e = stereo_line_edges[i];
    if(e->chi2() > cfg.stereo_line){
        e->setLevel(1);
    }
    e->setRobustKernel(0);
  }

  // optimize again without the outliers
  optimizer.initializeOptimization(0);
  optimizer.optimize(15);


  // check inlier observations     
  for(size_t i = 0; i < mono_edges.size(); i++){
    EdgeSE3ProjectPoint* e = mono_edges[i];
    mono_point_constraints[i]->inlier = (e->chi2() <= cfg.mono_point && e->isDepthPositive());
  }

  for(size_t i = 0; i < stereo_edges.size(); i++){    
    EdgeSE3ProjectStereoPoint* e = stereo_edges[i];
    stereo_point_constraints[i]->inlier = (e->chi2() <= cfg.stereo_point && e->isDepthPositive());
  }

  for(size_t i = 0; i < mono_line_edges.size(); i++){
    EdgeSE3ProjectLine* e = mono_line_edges[i];
    mono_line_constraints[i]->inlier = (e->chi2() <= cfg.mono_line);
  }

  for(size_t i = 0; i < stereo_line_edges.size(); i++){    
    EdgeStereoSE3ProjectLine* e = stereo_line_edges[i];
    stereo_line_constraints[i]->inlier = (e->chi2() <= cfg.stereo_line);
  }

  // recover optimized data
  // keyframes
  for(MapOfPoses::iterator it = poses.begin(); it != poses.end(); ++it){
    VertexVIPose* frame_vertex = static_cast<VertexVIPose*>(optimizer.vertex(it->first));
    g2o::SE3Quat SE3quat = frame_vertex->estimate().Tcw.inverse();
    it->second.p = SE3quat.translation();
    it->second.R = SE3quat.rotation().toRotationMatrix();
  }

  // points
  for(MapOfPoints3d::iterator it = points.begin(); it != points.end(); ++it){
    g2o::VertexPointXYZ* point_vertex = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(it->first+max_frame_id));
    it->second.p = point_vertex->estimate();
  }

  // lines
  for(MapOfLine3d::iterator it = lines.begin(); it != lines.end(); ++it){
    g2o::VertexLine3D* line_vertex = static_cast<g2o::VertexLine3D*>(optimizer.vertex(it->first+max_point_id));
    it->second.line_3d = line_vertex->estimate();
  } 

  // velocities
  for(MapOfVelocity::iterator it = velocities.begin(); it != velocities.end(); it++){
    VertexVelocity* velocity_vertex = static_cast<VertexVelocity*>(optimizer.vertex(it->first+max_line_id));
    it->second.velocity = velocity_vertex->estimate();
  }

  // biases 
  for(MapOfBias::iterator it = biases.begin(); it != biases.end(); it++){
    VertexGyrBias* gyr_bias_vertex = static_cast<VertexGyrBias*>(optimizer.vertex(it->first*2+max_velocity_id));
    VertexAccBias* acc_bias_vertex = static_cast<VertexAccBias*>(optimizer.vertex(it->first*2+max_velocity_id+1));
    it->second.gyr_bias = gyr_bias_vertex->estimate();
    it->second.acc_bias = acc_bias_vertex->estimate();
  }

}

int FrameOptimization(MapOfPoses& poses, MapOfPoints3d& points, MapOfLine3d& lines,
    MapOfVelocity& velocities, MapOfBias& biases, std::vector<CameraPtr>& camera_list, 
    VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints, 
    VectorOfMonoLineConstraints& mono_line_constraints, VectorOfStereoLineConstraints& stereo_line_constraints,
    VectorOfIMUConstraints& imu_constraints, Eigen::Matrix3d& Rwg, const OptimizationConfig& cfg){

  // std::cout << "---------FrameOptimization----------" << std::endl;
  // std::cout << "poses.size = " << poses.size() << std::endl;
  // std::cout << "points.size = " << points.size() << std::endl;
  // std::cout << "lines.size = " << lines.size() << std::endl;
  // std::cout << "velocities.size = " << velocities.size() << std::endl;
  // std::cout << "biases.size = " << biases.size() << std::endl;
  // std::cout << "mono_point_constraints.size = " << mono_point_constraints.size() << std::endl;
  // std::cout << "stereo_point_constraints.size = " << stereo_point_constraints.size() << std::endl;
  // std::cout << "mono_line_constraints.size = " << mono_line_constraints.size() << std::endl;
  // std::cout << "stereo_line_constraints.size = " << stereo_line_constraints.size() << std::endl;
  // std::cout << "imu_constraints.size = " << imu_constraints.size() << std::endl;
  // std::cout << "------------------------------------" << std::endl;

  // 1. optimizer
  g2o::SparseOptimizer optimizer;
  auto linear_solver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>();
  g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));

  optimizer.setVerbose(false);
  optimizer.setAlgorithm(solver);

  // 2. frame vertex
  VIPose current_pose;
  VertexVIPose* current_frame;
  int max_frame_id = 0;
  for(auto& kv : poses){
    Eigen::Matrix4d Tcb = camera_list[kv.second.id_camera]->BodyToCamera();
    Eigen::Matrix3d Rcw = kv.second.R.transpose();
    Eigen::Vector3d tcw = -Rcw * kv.second.p;
    VIPose vi_pose(Rcw, tcw, Tcb.block<3, 3>(0, 0), Tcb.block<3, 1>(0, 3));
    VertexVIPose* frame_vertex = new VertexVIPose();
    frame_vertex->setEstimate(vi_pose);
    frame_vertex->updateCache();
    g2o::SE3Quat SE3quat = frame_vertex->estimate().Tcw.inverse();
    frame_vertex->setId(kv.first);
    if(!kv.second.fixed){
      current_frame = frame_vertex;
      current_pose = vi_pose;
    }
    frame_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(frame_vertex);
    max_frame_id = std::max(max_frame_id, kv.first);
  } 
  max_frame_id++;

  // 3. point vertex
  int max_point_id = max_frame_id;
  for(auto& kv : points){
    g2o::VertexPointXYZ* point_vertex = new g2o::VertexPointXYZ();
    point_vertex->setEstimate(kv.second.p);
    int point_id = kv.first+max_frame_id;
    point_vertex->setId((point_id));
    max_point_id = std::max(max_point_id, point_id);
    point_vertex->setMarginalized(true);
    point_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(point_vertex);
  }
  max_point_id++;

  // 4. line vertex
  int max_line_id = max_point_id;
  for(auto& kv : lines){
    g2o::VertexLine3D* line_vertex = new g2o::VertexLine3D();
    line_vertex->setEstimateData(kv.second.line_3d);
    int line_id = kv.first+max_point_id;
    max_line_id = std::max(max_line_id, line_id);
    line_vertex->setId(line_id);
    line_vertex->setMarginalized(true);
    line_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(line_vertex);
  }
  max_line_id++;

  // 5. velocity vertex
  VertexVelocity* velocity_vertex_debug;
  int max_velocity_id = max_line_id;
  for(auto& kv : velocities){
    VertexVelocity* velocity_vertex = new VertexVelocity(kv.second.velocity);
    int velocity_id = kv.first + max_line_id;
    max_velocity_id = std::max(max_velocity_id, velocity_id);
    velocity_vertex->setId(velocity_id);
    velocity_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(velocity_vertex);
    if(!kv.second.fixed) velocity_vertex_debug = velocity_vertex;
  }
  max_velocity_id++;

  // 6. bias vertex
  int max_bias_id = max_velocity_id;
  for(auto& kv : biases){
    VertexGyrBias* gyr_bias_vertex = new VertexGyrBias(kv.second.gyr_bias);
    int gyr_bias_id = max_velocity_id + kv.first * 2;
    gyr_bias_vertex->setId(gyr_bias_id);
    gyr_bias_vertex->setFixed(kv.second.fixed);
    // gyr_bias_vertex->setFixed(true);
    optimizer.addVertex(gyr_bias_vertex);
    // std::cout << "kv.second.gyr_bias = " << kv.second.gyr_bias.transpose() << std::endl;

    VertexAccBias* acc_bias_vertex = new VertexAccBias(kv.second.acc_bias);
    int acc_bias_id = max_velocity_id + kv.first * 2 + 1;
    max_bias_id = std::max(max_bias_id, acc_bias_id);
    acc_bias_vertex->setId(acc_bias_id);
    acc_bias_vertex->setFixed(kv.second.fixed);
    // acc_bias_vertex->setFixed(true);
    optimizer.addVertex(acc_bias_vertex);
  }
  max_bias_id++;

  // 7. gravity direction vertex, will be fixed in this function.
  VertexGDirection* gravity_direction_vertex = new VertexGDirection(Rwg);
  gravity_direction_vertex->setId(max_bias_id);
  gravity_direction_vertex->setFixed(true);
  optimizer.addVertex(gravity_direction_vertex);

  // 8. point edges
  std::vector<EdgeSE3ProjectPoint*> mono_edges; 
  mono_edges.reserve(mono_point_constraints.size());
  std::vector<EdgeSE3ProjectStereoPoint*> stereo_edges;
  stereo_edges.reserve(stereo_point_constraints.size());
  const double thHuberMonoPoint = sqrt(cfg.mono_point);
  const double thHuberStereoPoint = sqrt(cfg.stereo_point);

  // 8.1 mono point edges
  for(MonoPointConstraintPtr& mpc : mono_point_constraints){
    EdgeSE3ProjectPoint* e = new EdgeSE3ProjectPoint();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((mpc->id_point+max_frame_id))));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mpc->id_pose)));
    e->setMeasurement(mpc->keypoint);
    e->setInformation(Eigen::Matrix2d::Identity());
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(thHuberMonoPoint);
    e->fx = camera_list[mpc->id_camera]->Fx();
    e->fy = camera_list[mpc->id_camera]->Fy();
    e->cx = camera_list[mpc->id_camera]->Cx();
    e->cy = camera_list[mpc->id_camera]->Cy();

    optimizer.addEdge(e);
    mono_edges.push_back(e);
  }

  // 8.2 stereo point edges
  for(StereoPointConstraintPtr& spc : stereo_point_constraints){
    EdgeSE3ProjectStereoPoint* e = new EdgeSE3ProjectStereoPoint();

    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((spc->id_point+max_frame_id))));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(spc->id_pose)));
    e->setMeasurement(spc->keypoint);
    e->setInformation(Eigen::Matrix3d::Identity());
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(thHuberStereoPoint);
    e->fx = camera_list[spc->id_camera]->Fx();
    e->fy = camera_list[spc->id_camera]->Fy();
    e->cx = camera_list[spc->id_camera]->Cx();
    e->cy = camera_list[spc->id_camera]->Cy();
    e->bf = camera_list[spc->id_camera]->BF();

    optimizer.addEdge(e);
    stereo_edges.push_back(e);
  }

  // 9. line edges
  std::vector<EdgeSE3ProjectLine*> mono_line_edges; 
  mono_line_edges.reserve(mono_line_constraints.size());
  std::vector<EdgeStereoSE3ProjectLine*> stereo_line_edges;
  stereo_line_edges.reserve(stereo_line_constraints.size());
  const double thHuberMonoLine = sqrt(cfg.mono_line);
  const double thHuberStereoLine = sqrt(cfg.stereo_line);

  // 9.1 mono line edges
  for(MonoLineConstraintPtr& mlc : mono_line_constraints){
    EdgeSE3ProjectLine* e = new EdgeSE3ProjectLine();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((mlc->id_line+max_point_id))));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mlc->id_pose)));
    e->setMeasurement(mlc->line_2d);
    e->setInformation(Eigen::Matrix2d::Identity() * 0.1);
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(thHuberMonoLine);
    double fx = camera_list[mlc->id_camera]->Fx();
    double fy = camera_list[mlc->id_camera]->Fy();
    double cx = camera_list[mlc->id_camera]->Cx();
    double cy = camera_list[mlc->id_camera]->Cy();
    e->fx = fx;
    e->fy = fy;
    e->Kv << -fy * cx, -fx * cy, fx * fy;
    optimizer.addEdge(e);
    mono_line_edges.push_back(e);
  }

  // 9.2 stereo line edges
  for(StereoLineConstraintPtr& slc : stereo_line_constraints){
    EdgeStereoSE3ProjectLine* e = new EdgeStereoSE3ProjectLine();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex((slc->id_line+max_point_id))));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(slc->id_pose)));
    e->setMeasurement(slc->line_2d);
    e->setInformation(Eigen::Matrix4d::Identity() * 0.1);
    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    e->setRobustKernel(rk);
    rk->setDelta(thHuberStereoLine);
    double fx = camera_list[slc->id_camera]->Fx();
    double fy = camera_list[slc->id_camera]->Fy();
    double cx = camera_list[slc->id_camera]->Cx();
    double cy = camera_list[slc->id_camera]->Cy();
    double bf = camera_list[slc->id_camera]->BF();
    e->fx = fx;
    e->fy = fy;
    e->b = bf / fx;
    e->Kv << -fy * cx, -fx * cy, fx * fy;
    optimizer.addEdge(e);
    stereo_line_edges.push_back(e);
  }

  // 10 imu edges
  std::vector<EdgeIMU*> imu_edges; 
  imu_edges.reserve(imu_constraints.size());
  std::vector<EdgeGyr*> gyr_edges;
  std::vector<EdgeAcc*> acc_edges;

  for(IMUConstraintPtr& ipc : imu_constraints){
    // 10.1 pose and velocity edges
    EdgeIMU* e_imu = new EdgeIMU(ipc->preinteration);
    
    g2o::HyperGraph::Vertex *vp1 = optimizer.vertex(ipc->id_pose1);
    g2o::HyperGraph::Vertex *vv1 = optimizer.vertex(max_line_id + ipc->id_pose1);
    g2o::HyperGraph::Vertex *vg1 = optimizer.vertex(max_velocity_id + ipc->id_pose1 * 2);
    g2o::HyperGraph::Vertex *va1 = optimizer.vertex(max_velocity_id + ipc->id_pose1 * 2 + 1);
    g2o::HyperGraph::Vertex *vp2 = optimizer.vertex(ipc->id_pose2);
    g2o::HyperGraph::Vertex *vv2 = optimizer.vertex(max_line_id + ipc->id_pose2);
    g2o::HyperGraph::Vertex *vg2 = optimizer.vertex(max_velocity_id + ipc->id_pose2 * 2);
    g2o::HyperGraph::Vertex *va2 = optimizer.vertex(max_velocity_id + ipc->id_pose2 * 2 + 1);
    g2o::HyperGraph::Vertex *vG = optimizer.vertex(max_bias_id);
    if(!vp1 || !vv1 || !vg1 || !va1 || !vp2 || !vv2 || !vg2 || !va2  || !vG) continue;

    e_imu->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vp1));
    e_imu->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vv1));
    e_imu->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg2));
    e_imu->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va2));
    e_imu->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vp2));
    e_imu->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vv2));
    e_imu->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vG));

    if(poses[ipc->id_pose1].fixed || poses[ipc->id_pose2].fixed || false){
      g2o::RobustKernelHuber *rki = new g2o::RobustKernelHuber;
      e_imu->setRobustKernel(rki);
      e_imu->setInformation(e_imu->information() * 1e-2);
      rki->setDelta(sqrt(16.92));
    }
    optimizer.addEdge(e_imu);
    imu_edges.push_back(e_imu);

    // bias edges
    EdgeGyr* e_gyr = new EdgeGyr();
    EdgeAcc* e_acc = new EdgeAcc();

    e_gyr->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg1));
    e_acc->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va1));
    e_gyr->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg2));
    e_acc->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va2));

    Eigen::Matrix3d info_g = ipc->preinteration->Cov.block<3,3>(9,9).inverse();
    Eigen::Matrix3d info_a = ipc->preinteration->Cov.block<3,3>(12,12).inverse();
    e_gyr->setInformation(info_g);
    e_acc->setInformation(info_a);

    optimizer.addEdge(e_gyr);
    optimizer.addEdge(e_acc);
    gyr_edges.push_back(e_gyr);
    acc_edges.push_back(e_acc);
  }

  // solve
  const int its[4]={10, 10, 10, 10};  

  int num_outlier = 0;
  for(size_t iter = 0; iter < 3; iter++){
    current_frame->setEstimate(current_pose);
    optimizer.initializeOptimization(0);
    optimizer.optimize(its[iter]);
  
    num_outlier=0;
    for(size_t i = 0; i < mono_edges.size(); i++){
      EdgeSE3ProjectPoint* e = mono_edges[i];
      if(!mono_point_constraints[i]->inlier){
        e->computeError();
      }

      const float chi2 = e->chi2();
      if(chi2 > cfg.mono_point){                
        mono_point_constraints[i]->inlier = false;
        e->setLevel(1);
        num_outlier++;
      }
      else{
        mono_point_constraints[i]->inlier = true;
        e->setLevel(0);
      }

      if(iter == 2) e->setRobustKernel(0);
    }

    
    for(size_t i = 0; i < stereo_edges.size(); i++){
      EdgeSE3ProjectStereoPoint* e = stereo_edges[i];
      if(!stereo_point_constraints[i]->inlier){
         e->computeError();
      }

      const float chi2 = e->chi2();
      if(chi2 > cfg.stereo_point){                
        stereo_point_constraints[i]->inlier = false;
        e->setLevel(1);
        num_outlier++;
      }
      else{
        stereo_point_constraints[i]->inlier = true;
        e->setLevel(0);
      }
      if(iter == 2) e->setRobustKernel(0);
    }

    for(size_t i = 0; i < mono_line_constraints.size(); i++){
      EdgeSE3ProjectLine* e = mono_line_edges[i];
      if(!mono_line_constraints[i]->inlier){
        e->computeError();
      }

      const float chi2 = e->chi2();
      if(chi2 > cfg.mono_line){                
        mono_line_constraints[i]->inlier = false;
        e->setLevel(1);
        num_outlier++;
      }
      else{
        mono_line_constraints[i]->inlier = true;
        e->setLevel(0);
      }
      if(iter == 2) e->setRobustKernel(0);      
    }

    for(size_t i = 0; i < stereo_line_constraints.size(); i++){
      EdgeStereoSE3ProjectLine* e = stereo_line_edges[i];
      if(!stereo_line_constraints[i]->inlier){
        e->computeError();
      }

      const float chi2 = e->chi2();
      if(chi2 > cfg.stereo_line){                
        stereo_line_constraints[i]->inlier = false;
        e->setLevel(1);
        num_outlier++;
      }
      else{
        stereo_line_constraints[i]->inlier = true;
        e->setLevel(0);
      }
      if(iter == 2) e->setRobustKernel(0);      
    }

    if(optimizer.edges().size()<10) break;
  }


  // // check edge
  // {
  //   std::cout << "------------------------after optimization ----------------------" << std::endl;
  //   std::cout << "imu_edges.size = " << imu_edges.size() << std::endl;
  //   for(size_t i = 0; i < imu_edges.size(); i++){
  //     EdgeIMU* e = imu_edges[i];
  //     e->computeError();
  //     e->PrintError();
  //     const float chi2 = e->chi2();
  //     std::cout << "imu_edges, i = " << i << ", e = " << chi2 << std::endl;
  //   }

  //   for(size_t i = 0; i < gyr_edges.size(); i++){
  //     EdgeGyr* e = gyr_edges[i];
  //     e->computeError();
  //     const float chi2 = e->chi2();
  //     std::cout << "gyr_edges, i = " << i << ", e = " << chi2 << std::endl;
  //   }

  //   for(size_t i = 0; i < acc_edges.size(); i++){
  //     EdgeAcc* e = acc_edges[i];
  //     e->computeError();
  //     const float chi2 = e->chi2();
  //     std::cout << "acc_edges, i = " << i << ", e = " << chi2 << std::endl;
  //   }

  //   double mono_point_error = 0;
  //   double stereo_point_error = 0;
  //   int mono_point_inlier = 0;
  //   int stereo_point_inlier = 0;    
  //   for(size_t i = 0; i < mono_edges.size(); i++){
  //     EdgeSE3ProjectPoint* e = mono_edges[i];
  //     if(mono_point_constraints[i]->inlier){
  //       e->computeError();
  //       const float chi2 = e->chi2();
  //       mono_point_error += chi2;
  //       mono_point_inlier++;
  //     }
  //   }
  //   for(size_t i = 0; i < stereo_edges.size(); i++){
  //     EdgeSE3ProjectStereoPoint* e = stereo_edges[i];
  //     if(stereo_point_constraints[i]->inlier){
  //       e->computeError();
  //       const float chi2 = e->chi2();
  //       stereo_point_error += chi2;
  //       stereo_point_inlier++;
  //     }
  //   }

  //   std::cout << "mono edge, num = " << mono_point_inlier << ", sum_error = " << mono_point_error << ", aver_error = " << mono_point_error/mono_point_inlier << std::endl;
  //   // std::cout << "stereo edge, num = " << stereo_point_inlier << ", sum_error = " << stereo_point_error << ", aver_error = " << stereo_point_error/stereo_point_inlier << std::endl;
  //   std::cout << "---------------------------------------------------------------" << std::endl;

  // }



  // recover optimized data
  // keyframes
  for(MapOfPoses::iterator it = poses.begin(); it != poses.end(); ++it){
    VertexVIPose* frame_vertex = static_cast<VertexVIPose*>(optimizer.vertex(it->first));
    g2o::SE3Quat SE3quat = frame_vertex->estimate().Tcw.inverse();
    it->second.p = SE3quat.translation();
    it->second.R = SE3quat.rotation().toRotationMatrix();
  }

  // velocities
  for(MapOfVelocity::iterator it = velocities.begin(); it != velocities.end(); it++){
    VertexVelocity* velocity_vertex = static_cast<VertexVelocity*>(optimizer.vertex(it->first+max_line_id));
    it->second.velocity = velocity_vertex->estimate();
  }

  // biases 
  for(MapOfBias::iterator it = biases.begin(); it != biases.end(); it++){
    VertexGyrBias* gyr_bias_vertex = static_cast<VertexGyrBias*>(optimizer.vertex(it->first*2+max_velocity_id));
    VertexAccBias* acc_bias_vertex = static_cast<VertexAccBias*>(optimizer.vertex(it->first*2+max_velocity_id+1));
    it->second.gyr_bias = gyr_bias_vertex->estimate();
    it->second.acc_bias = acc_bias_vertex->estimate();
  }

  return (mono_point_constraints.size() + stereo_point_constraints.size() + mono_line_constraints.size() + stereo_line_constraints.size() - num_outlier);
}

bool IMUInitialization(MapOfPoses& poses, MapOfVelocity& velocities, Bias& bias, std::vector<CameraPtr>& camera_list,
    VectorOfIMUConstraints& imu_constraints, Eigen::Matrix3d& Rwg){
  // 1. optimizer
  g2o::SparseOptimizer optimizer;
  auto linear_solver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
  g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));

  optimizer.setVerbose(false);
  optimizer.setAlgorithm(solver);

  // frame vertex
  int max_frame_id = 0;
  for(auto& kv : poses){
    Eigen::Matrix4d Tcb = camera_list[kv.second.id_camera]->BodyToCamera();
    Eigen::Matrix3d Rcw = kv.second.R.transpose();
    Eigen::Vector3d tcw = -Rcw * kv.second.p;
    VIPose vi_pose(Rcw, tcw, Tcb.block<3, 3>(0, 0), Tcb.block<3, 1>(0, 3));
    VertexVIPose* frame_vertex = new VertexVIPose();
    frame_vertex->setEstimate(vi_pose);
    frame_vertex->setId(kv.first);
    frame_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(frame_vertex);
    max_frame_id = std::max(max_frame_id, kv.first);
  } 
  max_frame_id++; 

  // velocity vertex
  int max_velocity_id = max_frame_id;
  for(auto& kv : velocities){
    VertexVelocity* velocity_vertex = new VertexVelocity(kv.second.velocity);
    int velocity_id = kv.first + max_frame_id;
    max_velocity_id = std::max(max_velocity_id, velocity_id);
    velocity_vertex->setId(velocity_id);
    velocity_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(velocity_vertex);
  }
  max_velocity_id++;

  // bias vertex
  Eigen::Vector3d prior_gyr_bias = bias.gyr_bias; 
  Eigen::Vector3d prior_acc_bias = bias.acc_bias; 
  int max_bias_id = max_velocity_id;
  VertexGyrBias* gyr_bias_vertex = new VertexGyrBias(prior_gyr_bias);
  gyr_bias_vertex->setId(max_bias_id);
  gyr_bias_vertex->setFixed(false);
  optimizer.addVertex(gyr_bias_vertex);
  max_bias_id++;

  VertexAccBias* acc_bias_vertex = new VertexAccBias(prior_acc_bias);
  acc_bias_vertex->setId(max_bias_id);
  acc_bias_vertex->setFixed(false);
  optimizer.addVertex(acc_bias_vertex);
  max_bias_id++;

  VertexGyrBias* prior_gyr_bias_vertex = new VertexGyrBias(prior_gyr_bias);
  prior_gyr_bias_vertex->setId(max_bias_id);
  prior_gyr_bias_vertex->setFixed(true);
  optimizer.addVertex(prior_gyr_bias_vertex);
  max_bias_id++;

  VertexAccBias* prior_acc_bias_vertex = new VertexAccBias(prior_acc_bias);
  prior_acc_bias_vertex->setId(max_bias_id);
  prior_acc_bias_vertex->setFixed(true);
  optimizer.addVertex(prior_acc_bias_vertex);
  max_bias_id++;

  // gravity direction vertex.
  VertexGDirection* gravity_direction_vertex = new VertexGDirection(Rwg);
  gravity_direction_vertex->setId(max_bias_id);
  gravity_direction_vertex->setFixed(false);
  optimizer.addVertex(gravity_direction_vertex);

  // prior bias edges
  g2o::HyperGraph::Vertex *vg = optimizer.vertex(max_velocity_id);
  g2o::HyperGraph::Vertex *va = optimizer.vertex(max_velocity_id + 1);
  g2o::HyperGraph::Vertex *pvg = optimizer.vertex(max_velocity_id + 2);
  g2o::HyperGraph::Vertex *pva = optimizer.vertex(max_velocity_id + 3);

  EdgeGyr* e_gyr_prior = new EdgeGyr();
  EdgeAcc* e_acc_prior = new EdgeAcc();
  e_gyr_prior->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(pvg));
  e_gyr_prior->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg));
  e_gyr_prior->setInformation(1e2 * Eigen::Matrix3d::Identity());
  e_acc_prior->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va));
  e_acc_prior->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(pva));
  e_acc_prior->setInformation(1e5 * Eigen::Matrix3d::Identity());
  optimizer.addEdge(e_gyr_prior);
  optimizer.addEdge(e_acc_prior);

  // imu edges
  std::vector<EdgeIMU*> imu_edges; 
  imu_edges.reserve(imu_constraints.size());
  std::vector<EdgeGyr*> gyr_edges;
  std::vector<EdgeAcc*> acc_edges;
  std::vector<EdgeGyr*> prior_gyr_edges;
  std::vector<EdgeAcc*> prior_acc_edges;

  for(IMUConstraintPtr& ipc : imu_constraints){
    g2o::HyperGraph::Vertex *vp1 = optimizer.vertex(ipc->id_pose1);
    g2o::HyperGraph::Vertex *vv1 = optimizer.vertex(max_frame_id + ipc->id_pose1);
    g2o::HyperGraph::Vertex *vp2 = optimizer.vertex(ipc->id_pose2);
    g2o::HyperGraph::Vertex *vv2 = optimizer.vertex(max_frame_id + ipc->id_pose2);
    g2o::HyperGraph::Vertex *vG = optimizer.vertex(max_bias_id);

    EdgeIMU* e_imu = new EdgeIMU(ipc->preinteration);
    e_imu->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vp1));
    e_imu->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vv1));
    e_imu->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg));
    e_imu->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va));
    e_imu->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vp2));
    e_imu->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vv2));
    e_imu->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vG));

    optimizer.addEdge(e_imu);
    imu_edges.push_back(e_imu);
  }

  // // // check edge
  // std::cout << "------------------------before optimization ----------------------" << std::endl;
  // std::cout << "imu_edges.size = " << imu_edges.size() << std::endl;
  // for(size_t i = 0; i < imu_edges.size(); i++){
  //   EdgeIMU* e = imu_edges[i];
  //   e->computeError();
  //   const float chi2 = e->chi2();
  //   std::cout << "imu_edges, i = " << i << ", e = " << chi2 << std::endl;
  // }

  // {
  //   e_gyr_prior->computeError();
  //   const float chi2 = e_gyr_prior->chi2();
  //   std::cout << "prior_gyr_edges, e = " << chi2 << std::endl;
  // }

  // {
  //   e_acc_prior->computeError();
  //   const float chi2 = e_gyr_prior->chi2();
  //   std::cout << "prior_acc_edges, e = " << chi2 << std::endl;
  // }
  // std::cout << "---------------------------------------------------------------" << std::endl;

  optimizer.setVerbose(false);
  optimizer.initializeOptimization();
  optimizer.optimize(200);

  // // // check edge
  // std::cout << "------------------------after optimization ----------------------" << std::endl;
  // std::cout << "imu_edges.size = " << imu_edges.size() << std::endl;
  // for(size_t i = 0; i < imu_edges.size(); i++){
  //   EdgeIMU* e = imu_edges[i];
  //   e->computeError();
  //   const float chi2 = e->chi2();
  //   std::cout << "imu_edges, i = " << i << ", e = " << chi2 << std::endl;
  // }

  // {
  //   e_gyr_prior->computeError();
  //   const float chi2 = e_gyr_prior->chi2();
  //   std::cout << "prior_gyr_edges, e = " << chi2 << std::endl;
  // }

  // {
  //   e_acc_prior->computeError();
  //   const float chi2 = e_acc_prior->chi2();
  //   std::cout << "prior_acc_edges, e = " << chi2 << std::endl;
  // }
  // std::cout << "---------------------------------------------------------------" << std::endl;

  // recover data
  // velocities
  for(MapOfVelocity::iterator it = velocities.begin(); it != velocities.end(); it++){
    VertexVelocity* velocity_vertex = static_cast<VertexVelocity*>(optimizer.vertex(it->first+max_frame_id));
    it->second.velocity = velocity_vertex->estimate();
  }

  // biases 
  bias.gyr_bias = gyr_bias_vertex->estimate(); 
  bias.acc_bias = acc_bias_vertex->estimate(); 
  std::cout << "gyr_bias = " << bias.gyr_bias.transpose() << ", acc_bias = " << bias.acc_bias.transpose() << std::endl;

  Rwg = NormalizeRotation(gravity_direction_vertex->estimate().Rwg);
  return true;
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
  if(object_points.size() < 8) return 0;

  try{
    cv::solvePnPRansac(object_points, image_points, camera_matrix, dist_coeffs, 
        rotation_vector, translation_vector, false, 100, 20.0, 0.99, cv_inliers);
  }catch(...){
    return 0;
  }

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

bool ComputeGyrBias(std::vector<FramePtr>& frames, Eigen::Vector3d& dbg){
  Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();

  for(size_t i = 0; i < frames.size() - 1; i++){
    FramePtr frame = frames[i];
    FramePtr last_frame = frames[i+1];
    Eigen::Matrix3d Ai = frame->GetIMUPreinteration()->JRg;
    Eigen::Matrix3d delta_R = last_frame->IMUPose().block<3, 3>(0, 0).transpose() * frame->IMUPose().block<3, 3>(0, 0);
    Eigen::Vector3d delta_r;
    SO3Log(frame->GetIMUPreinteration()->dR.transpose() * delta_R, delta_r);
    Eigen::Vector3d bi = delta_r;

    A += Ai.transpose() * Ai;
    b += Ai.transpose() * bi;
  }

  dbg = A.ldlt().solve(b);

  return true;
}

void ValidateGyrBias(std::vector<FramePtr>& frames){
  std::cout << "-----------------ValidateGyrBias--------------" << std::endl;
  for(size_t i = 0; i < frames.size() - 1; i++){
    FramePtr frame = frames[i];
    FramePtr last_frame = frames[i+1];
    Eigen::Matrix3d delta_R = last_frame->IMUPose().block<3, 3>(0, 0).transpose() * frame->IMUPose().block<3, 3>(0, 0);
    Eigen::Vector3d delta_r;
    SO3Log(frame->GetIMUPreinteration()->dR.transpose() * delta_R, delta_r);
    std::cout << "frame_id = " << frame->GetFrameId() << ", delta_r = " << delta_r.transpose() << std::endl;
  }
  std::cout << "----------------------------------------------" << std::endl;
}

bool ComputeVelocity(std::vector<FramePtr>& frames, Eigen::Vector3d& gw){
  int N = frames.size();
  int D = 3 * (N+1);
  Eigen::MatrixXd A{D, D};
  Eigen::VectorXd b{D};

  A.setZero();
  b.setZero();

  for(int i = 0; i < N-1; i++){
    FramePtr frame = frames[i];
    FramePtr last_frame = frames[i+1];

    double dt = frame->GetIMUPreinteration()->dT;
    Eigen::Matrix4d pose_i = last_frame->IMUPose();
    Eigen::Matrix4d pose_j = frame->IMUPose();

    Eigen::MatrixXd tmp_A(6, 9);
    Eigen::VectorXd tmp_b(6);

    tmp_A.setZero();
    tmp_b.setZero();

    tmp_A.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    tmp_A.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
    tmp_A.block<3, 3>(0, 6) = -dt * Eigen::Matrix3d::Identity();
    tmp_A.block<3, 3>(3, 3) = dt * Eigen::Matrix3d::Identity();
    tmp_A.block<3, 3>(3, 6) = 0.5 * dt * dt * Eigen::Matrix3d::Identity();

    tmp_b.block<3, 1>(0, 0) = pose_i.block<3, 3>(0, 0) * frame->GetIMUPreinteration()->dV;
    tmp_b.block<3, 1>(3, 0) = pose_j.block<3, 1>(0, 3) - pose_i.block<3, 1>(0, 3) - pose_i.block<3, 3>(0, 0) * frame->GetIMUPreinteration()->dP;

    Eigen::MatrixXd r_A = tmp_A.transpose() * tmp_A;
    Eigen::VectorXd r_b = tmp_A.transpose() * tmp_b;

    A.block<6, 6>(i*3, i*3) += r_A.topLeftCorner<6, 6>();
    A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
    A.block<6, 3>(i*3, D-3) += r_A.topRightCorner<6, 3>();
    A.block<3, 6>(D-3, i*3) += r_A.bottomLeftCorner<3, 6>();

    b.segment<6>(i*3) += r_b.head<6>();
    b.tail<3>() += r_b.tail<3>();
  }
  A = A * 1000.0;
  b = b * 1000.0;
  Eigen::VectorXd x = A.ldlt().solve(b);
  gw = x.tail<3>();

  std::cout << "x = " << x.transpose() << std::endl;


  for(int i = 0; i < N; i++){
    frames[i]->SetVelocaity(x.segment<3>(i*3));
  }

  // ValidateVelocity(frames, x);

  return true;
}

void ValidateVelocity(std::vector<FramePtr>& frames, Eigen::VectorXd x){
  std::cout << "-----------------ValidateVelocity--------------" << std::endl;

  x.tail(3) = x.tail(3) * (Camera::IMU_G_VALUE / x.tail(3).norm());

  int N = frames.size();
  for(int i = 0; i < N-1; i++){
    FramePtr frame = frames[i];
    FramePtr last_frame = frames[i+1];

    double dt = frame->GetIMUPreinteration()->dT;
    Eigen::Matrix4d pose_i = last_frame->IMUPose();
    Eigen::Matrix4d pose_j = frame->IMUPose();

    Eigen::MatrixXd tmp_A(6, 9);
    Eigen::VectorXd tmp_b(6);

    tmp_A.setZero();
    tmp_b.setZero();

    tmp_A.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    tmp_A.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
    tmp_A.block<3, 3>(0, 6) = -dt * Eigen::Matrix3d::Identity();
    tmp_A.block<3, 3>(3, 3) = dt * Eigen::Matrix3d::Identity();
    tmp_A.block<3, 3>(3, 6) = 0.5 * dt * dt * Eigen::Matrix3d::Identity();

    tmp_b.block<3, 1>(0, 0) = pose_i.block<3, 3>(0, 0) * frame->GetIMUPreinteration()->dV;
    tmp_b.block<3, 1>(3, 0) = pose_j.block<3, 1>(0, 3) - pose_i.block<3, 1>(0, 3) - pose_i.block<3, 3>(0, 0) * frame->GetIMUPreinteration()->dP;

    Eigen::MatrixXd r_A = tmp_A.transpose() * tmp_A;
    Eigen::VectorXd r_b = tmp_A.transpose() * tmp_b;

    Eigen::VectorXd tmp_x(9);
    tmp_x.head(6) = x.segment<6>(i*3);
    tmp_x.tail(3) = x.tail<3>();

    std::cout << "frame_id = " << frame->GetFrameId() << ", Ax-b = " << (tmp_A*tmp_x-tmp_b).transpose() << std::endl;
    Eigen::Vector3d v2 = tmp_x.head(3);
    Eigen::Vector3d v1 = tmp_x.segment<3>(3);
    Eigen::Vector3d gw = tmp_x.tail(3);

    Eigen::Vector3d ev = pose_i.block<3, 3>(0, 0).transpose() * (v2 - v1 - gw * dt) - frame->GetIMUPreinteration()->dV;
    std::cout << "ev = " << ev.transpose() << std::endl;
  }
  std::cout << "----------------------------------------------" << std::endl;
}

void ValidateError(std::vector<FramePtr>& frames, const Eigen::Vector3d& gw){
  std::cout << "-----------------ValidateError----------------" << std::endl;
  int N = frames.size();
  for(int i = 0; i < N-1; i++){
    FramePtr frame = frames[i];
    FramePtr last_frame = frames[i+1];

    PreinterationPtr preinteration = frame->GetIMUPreinteration();
    const Eigen::Matrix3d dR = preinteration->GetUpdatedDeltaRotation();
    const Eigen::Vector3d dV = preinteration->GetUpdatedDeltaVelocity();
    const Eigen::Vector3d dP = preinteration->GetUpdatedDeltaPosition();
    Eigen::Matrix4d pose_1 = last_frame->IMUPose();
    Eigen::Matrix4d pose_2 = frame->IMUPose();
    Eigen::Matrix3d Rwb1 = pose_1.block<3, 3>(0, 0);
    Eigen::Matrix3d Rwb2 = pose_2.block<3, 3>(0, 0);
    Eigen::Vector3d twb1 = pose_1.block<3, 1>(0, 3);
    Eigen::Vector3d twb2 = pose_2.block<3, 1>(0, 3);
    Eigen::Vector3d v2 = frame->GetVelocity();
    Eigen::Vector3d v1 = last_frame->GetVelocity();
    double dt = preinteration->dT;

    Eigen::Vector3d er;
    SO3Log(dR.transpose()*Rwb1.transpose()*Rwb2, er);    
    Eigen::Vector3d ev = Rwb1.transpose() * (v2 - v1 - gw * dt) - dV;
    Eigen::Vector3d ep = Rwb1.transpose() * (twb2 - twb1 - v1*dt - gw*dt*dt/2) - dP;


    std::cout << "frame_id = " << frame->GetFrameId()  << std::endl;
    std::cout << "er = " << er.norm() << ", ev = " << ev.norm() << ", ep = " << ep.norm() << std::endl;
  }
  std::cout << "----------------------------------------------" << std::endl;
}

bool ValidateError(MapOfPoses& poses, MapOfVelocity& velocities, Bias& bias, std::vector<CameraPtr>& camera_list,
    VectorOfIMUConstraints& imu_constraints, Eigen::Matrix3d& Rwg, const Eigen::Vector3d& prior_gyr_bias, const Eigen::Vector3d& prior_acc_bias){
  std::cout << "-----------------ValidateError----------------" << std::endl;
  Eigen::Vector3d g;
  g << 0, 0, -Camera::IMU_G_VALUE;
  Eigen::Vector3d gw = Rwg * g;

  Eigen::Vector3d gyr_bias = bias.gyr_bias;
  Eigen::Vector3d acc_bias = bias.acc_bias;

  for(IMUConstraintPtr& ipc : imu_constraints){
    int id_pose1 = ipc->id_pose1;
    int id_pose2 = ipc->id_pose2;
    PreinterationPtr preinteration = ipc->preinteration;

    Pose3d pose_3d1 = poses[id_pose1];
    Pose3d pose_3d2 = poses[id_pose2];

    Eigen::Matrix4d Tcb = camera_list[pose_3d1.id_camera]->BodyToCamera();
    Eigen::Matrix3d Rcb = Tcb.block<3, 3>(0, 0);
    Eigen::Vector3d tcb = Tcb.block<3, 1>(0, 3);

    Eigen::Matrix3d Rwc1 = pose_3d1.R;
    Eigen::Vector3d twc1 = pose_3d1.p;

    Eigen::Matrix3d Rwb1 = Rwc1 * Rcb;
    Eigen::Vector3d twb1 = twc1 + Rwc1 * tcb;

    Eigen::Matrix3d Rwc2 = pose_3d2.R;
    Eigen::Vector3d twc2 = pose_3d2.p;

    Eigen::Matrix3d Rwb2 = Rwc2 * Rcb;
    Eigen::Vector3d twb2 = twc2 + Rwc2 * tcb;

    Eigen::Vector3d v1 = velocities[id_pose1].velocity;
    Eigen::Vector3d v2 = velocities[id_pose2].velocity;

    double dt = preinteration->dT;
    const Eigen::Matrix3d dR = preinteration->GetDeltaRotation(gyr_bias);
    const Eigen::Vector3d dV = preinteration->GetDeltaVelocity(gyr_bias, acc_bias);
    const Eigen::Vector3d dP = preinteration->GetDeltaPosition(gyr_bias, acc_bias);

    Eigen::Vector3d er;
    SO3Log(dR.transpose()*Rwb1.transpose()*Rwb2, er);    
    Eigen::Vector3d ev = Rwb1.transpose() * (v2 - v1 - gw * dt) - dV;
    Eigen::Vector3d ep = Rwb1.transpose() * (twb2 - twb1 - v1*dt - gw*dt*dt/2) - dP;

    Eigen::Vector3d p1 = -Rwb1 * dV;
    Eigen::Vector3d p2 = v2;
    Eigen::Vector3d p3 = -v1;
    Eigen::Vector3d p4 = -gw * dt;
    std::cout << "p1 = " << p1.transpose() << std::endl;
    std::cout << "p2 = " << p2.transpose() << std::endl;
    std::cout << "p3 = " << p3.transpose() << std::endl;
    std::cout << "p4 = " << p4.transpose() << std::endl;
    Eigen::Vector3d p = p1 + p2 + p3 + p4;
    std::cout << "p = " << p.transpose() << std::endl;

    std::cout << "frame_id = " << id_pose2 << std::endl;
    // std::cout << "v1 = " << v1.transpose() << ", v2 = " << v2.transpose() << std::endl;
    std::cout << "er = " << er.transpose() << ", ev = " << ev.transpose() << ", ep = " << ep.transpose() << std::endl;
  }
  std::cout << "----------------------------------------------" << std::endl;
  return true;
}

void ValidateIMUInitialization(std::vector<FramePtr>& frames){
  Eigen::Vector3d g;
  g << 0, 0, -Camera::IMU_G_VALUE;

  for(size_t i = 0; i < frames.size()-1; i++){
    FramePtr frame1 = frames[i];
    FramePtr frame0 = frames[i+1];
    PreinterationPtr preinteration = frame1->GetIMUPreinteration();

    Eigen::Matrix4d Twb0 = frame0->IMUPose();
    Eigen::Matrix3d Rwb0 = Twb0.block<3, 3>(0, 0);
    Eigen::Vector3d twb0 = Twb0.block<3, 1>(0, 3);
    Eigen::Vector3d vwb0 = frame0->GetVelocity();
    double dt = preinteration->dT;

    Eigen::Matrix4d Twb1 = Eigen::Matrix4d::Identity();

    Twb1.block<3, 3>(0, 0) = NormalizeRotation(Rwb0 * preinteration->GetUpdatedDeltaRotation());
    Twb1.block<3, 1>(0, 3) = twb0 + vwb0 * dt + 0.5 * dt * dt * g + Rwb0 * preinteration->GetUpdatedDeltaPosition();
    Eigen::Vector3d vwb = vwb0 + dt * g + Rwb0 * preinteration->GetUpdatedDeltaVelocity();

    Eigen::Matrix4d Twb1_real = frame1->IMUPose();
    Eigen::Vector3d vwb_real = frame1->GetVelocity();

    std::cout << "i = " << i << std::endl;
    std::cout << "dt = " << dt << std::endl;
    // std::cout << "Twb1 = " << Twb1.block<3, 1>(0, 3).transpose() << ", Twb1_real = " << Twb1_real.block<3, 1>(0, 3).transpose() << std::endl;
    Eigen::Vector3d rwb, rwb_real;
    SO3Log(Twb1.block<3, 3>(0, 0), rwb);
    SO3Log(Twb1_real.block<3, 3>(0, 0), rwb_real);
    // std::cout << "rwb = " << rwb.transpose() << ", rwb_real = " << rwb_real.transpose() << std::endl;
    std::cout << "vwb = " << vwb.transpose() << ", vwb_real = " << vwb_real.transpose() << std::endl;
    // std::cout << "Rwb0 = " << Rwb0 << std::endl;

    Eigen::Vector3d pv1 = vwb0;
    Eigen::Vector3d pv2 = dt * g;
    Eigen::Vector3d pv3 = Rwb0 * preinteration->GetUpdatedDeltaVelocity();
    std::cout << "pv1 = " << pv1.transpose() << std::endl;
    std::cout << "pv2 = " << pv2.transpose() << std::endl;
    std::cout << "pv3 = " << pv3.transpose() << std::endl;

    // // for debug
    // Eigen::Vector3d p2 = twb0;
    // Eigen::Vector3d p3 = vwb0 * dt;
    // Eigen::Vector3d p4 = 0.5 * dt * dt * g;
    // Eigen::Vector3d p5 = Rwb0 * preinteration->GetUpdatedDeltaPosition();
    // std::cout << "p2 = " << p2.transpose() << std::endl;
    // std::cout << "p3 = " << p3.transpose() << std::endl;
    // std::cout << "p4 = " << p4.transpose() << std::endl;
    // std::cout << "p5 = " << p5.transpose() << std::endl;

  }
}

void PoseGraphOptimization(MapOfPoses& poses, std::vector<CameraPtr>& camera_list, 
    VectorOfRelativePoseConstraints& relative_pose_constraints){
  if(poses.empty() || camera_list.empty() || relative_pose_constraints.empty()){
    return;
  }

  // 1. optimizer
  g2o::SparseOptimizer optimizer;
  auto linear_solver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
  g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));

  optimizer.setVerbose(false);
  optimizer.setAlgorithm(solver);

  // 2. frame vertex
  for(auto& kv : poses){
    Eigen::Matrix4d Tcb = camera_list[kv.second.id_camera]->BodyToCamera();
    Eigen::Matrix3d Rcw = kv.second.R.transpose();
    Eigen::Vector3d tcw = -Rcw * kv.second.p;
    VIPose vi_pose(Rcw, tcw, Tcb.block<3, 3>(0, 0), Tcb.block<3, 1>(0, 3));
    VertexVIPose* frame_vertex = new VertexVIPose();
    frame_vertex->setEstimate(vi_pose);
    frame_vertex->setId(kv.first);
    frame_vertex->setFixed(kv.second.fixed);
    optimizer.addVertex(frame_vertex);
  } 

  // 3. add relative pose constraints
  std::vector<EdgeRelativePose*> relative_pose_edges;
  relative_pose_edges.reserve(relative_pose_constraints.size());
  Eigen::Matrix<double, 6, 6> information_matrix = Eigen::Matrix<double, 6, 6>::Identity();
  for(RelativePoseConstraintPtr& rpc : relative_pose_constraints){
    EdgeRelativePose* e = new EdgeRelativePose();
    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(rpc->id_pose1)));
    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(rpc->id_pose2)));
    e->Rc1c2 = rpc->Rc1c2;
    e->tc1c2 = rpc->tc1c2;
    e->setInformation(information_matrix);
    optimizer.addEdge(e);
    relative_pose_edges.emplace_back(e);
  }

  // solve 
  optimizer.initializeOptimization();
  optimizer.optimize(20);

  // recover optimized data
  for(MapOfPoses::iterator it = poses.begin(); it != poses.end(); ++it){
    VertexVIPose* frame_vertex = static_cast<VertexVIPose*>(optimizer.vertex(it->first));
    g2o::SE3Quat SE3quat = frame_vertex->estimate().Tcw.inverse();
    it->second.p = SE3quat.translation();
    it->second.R = SE3quat.rotation().toRotationMatrix();
  }
  return;
}

void GlobalBA(MapPtr _map, const OptimizationConfig& cfg, bool point_outlier_rejection, 
    bool line_outlier_rejection, int first_iterations, int second_iterations){
  std::map<int, MappointPtr>& mappoints = _map->GetAllMappoints();
  std::map<int, MaplinePtr>& maplines = _map->GetAllMaplines();
  std::map<int, FramePtr>& keyframes = _map->GetAllKeyframes();
  CameraPtr camera = _map->GetCameraPtr();

  // 1. optimizer
  g2o::SparseOptimizer optimizer;
  auto linear_solver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
  g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));

  optimizer.setVerbose(false);
  optimizer.setAlgorithm(solver);

  // 2. frame vertex
  int max_frame_id = 0;
  Eigen::Matrix4d Tcb = camera->BodyToCamera();
  Eigen::Matrix3d Rcb = Tcb.block<3, 3>(0, 0);
  Eigen::Vector3d tcb = Tcb.block<3, 1>(0, 3);
  Eigen::Matrix3d Rwg = _map->GetRwg();
  for(auto& kv : keyframes){
    bool fix_this_frame = (kv.first == keyframes.begin()->first);

    Eigen::Matrix4d Twc = kv.second->GetPose();
    Eigen::Matrix3d Rwc = Twc.block<3, 3>(0, 0);
    Eigen::Vector3d twc = Twc.block<3, 1>(0, 3);

    Eigen::Matrix3d Rcw = Rwc.transpose();
    Eigen::Vector3d tcw = -Rcw * twc;
    VIPose vi_pose(Rcw, tcw, Rcb, tcb);
    VertexVIPose* frame_vertex = new VertexVIPose();
    frame_vertex->setEstimate(vi_pose);
    frame_vertex->setId(kv.first);
    frame_vertex->setFixed(fix_this_frame);
    optimizer.addVertex(frame_vertex);
    max_frame_id = std::max(max_frame_id, kv.first);
  } 
  max_frame_id++;

  // 3. point vertex
  int max_point_id = max_frame_id;
  for(auto& kv : mappoints){
    MappointPtr mpt = kv.second;
    if(!mpt || !mpt->IsValid()) continue;

    g2o::VertexPointXYZ* point_vertex = new g2o::VertexPointXYZ();
    point_vertex->setEstimate(mpt->GetPosition());
    int point_id = kv.first + max_frame_id;
    point_vertex->setId((point_id));
    max_point_id = std::max(max_point_id, point_id);
    point_vertex->setMarginalized(true);
    point_vertex->setFixed(false);
    optimizer.addVertex(point_vertex);
  }
  max_point_id++;

  // 4. line vertex
  int max_line_id = max_point_id;
  for(auto& kv : maplines){
    MaplinePtr mpl = kv.second;
    if(!mpl || !mpl->IsValid()) continue;

    g2o::VertexLine3D* line_vertex = new g2o::VertexLine3D();
    line_vertex->setEstimateData(mpl->GetLine3D());
    int line_id = kv.first + max_point_id;
    max_line_id = std::max(max_line_id, line_id);
    line_vertex->setId(line_id);
    line_vertex->setMarginalized(true);
    line_vertex->setFixed(false);
    optimizer.addVertex(line_vertex);
  }
  max_line_id++;

  int max_bias_id = -1;
  int gravity_direction_id = -1;
  if(_map->IMUInit()){
    max_bias_id = max_line_id;
    for(auto& kv : keyframes){
      bool fix_this_frame = (kv.first == keyframes.begin()->first);
      FramePtr frame = kv.second;

      // 5. velocity vertex
      assert(frame->VelocityIsInitialized());
      VertexVelocity* velocity_vertex = new VertexVelocity(frame->GetVelocity());
      int velocity_id = kv.first * 3 + max_line_id;
      velocity_vertex->setId(velocity_id);
      velocity_vertex->setFixed(fix_this_frame);
      optimizer.addVertex(velocity_vertex);

      // 6. bias vertex
      Eigen::Vector3d gyr_bias, acc_bias;
      PreinterationPtr preinteration = frame->GetIMUPreinteration();
      if(fix_this_frame){
        gyr_bias = preinteration->bg + preinteration->dbg;
        acc_bias = preinteration->ba + preinteration->dba;
      }else{
        gyr_bias = preinteration->bg;
        acc_bias = preinteration->ba;
      }
      VertexGyrBias* gyr_bias_vertex = new VertexGyrBias(gyr_bias);
      int gyr_bias_id = kv.first * 3 + 1 + max_line_id;
      gyr_bias_vertex->setId(gyr_bias_id);
      gyr_bias_vertex->setFixed(fix_this_frame);
      optimizer.addVertex(gyr_bias_vertex);

      VertexAccBias* acc_bias_vertex = new VertexAccBias(acc_bias);
      int acc_bias_id = kv.first * 3 + 2 + max_line_id;
      max_bias_id = std::max(max_bias_id, acc_bias_id);
      acc_bias_vertex->setId(acc_bias_id);
      acc_bias_vertex->setFixed(fix_this_frame);
      optimizer.addVertex(acc_bias_vertex);
    }

    // 7. gravity direction vertex, will be fixed in this function.
    gravity_direction_id = max_bias_id + 1;
    VertexGDirection* gravity_direction_vertex = new VertexGDirection(Rwg);
    gravity_direction_vertex->setId(gravity_direction_id);
    gravity_direction_vertex->setFixed(true);
    optimizer.addVertex(gravity_direction_vertex);
  }

  // edges
  std::vector<EdgeSE3ProjectPoint*> mono_edges; 
  std::vector<EdgeSE3ProjectStereoPoint*> stereo_edges;
  const double thHuberMonoPoint = sqrt(cfg.mono_point);
  const double thHuberStereoPoint = sqrt(cfg.stereo_point);

  std::vector<EdgeSE3ProjectLine*> mono_line_edges; 
  std::vector<EdgeStereoSE3ProjectLine*> stereo_line_edges;
  const double thHuberMonoLine = sqrt(cfg.mono_line);
  const double thHuberStereoLine = sqrt(cfg.stereo_line);

  std::vector<EdgeIMU*> imu_edges; 
  std::vector<EdgeGyr*> gyr_edges;
  std::vector<EdgeAcc*> acc_edges;

  double fx = camera->Fx();
  double fy = camera->Fy();
  double cx = camera->Cx();
  double cy = camera->Cy();
  double bf = camera->BF();
  Eigen::Vector3d Kv;
  Kv << -fy * cx, -fx * cy, fx * fy;

  FramePtr last_frame = std::shared_ptr<Frame>(nullptr);
  for(auto& kv : keyframes){
    int frame_id = kv.first;
    FramePtr frame = kv.second;
    
    int frame_vertex_id = frame_id;

    // 8. point edges
    std::vector<MappointPtr>& frame_mpts = frame->GetAllMappoints();
    for(size_t i = 0; i < frame_mpts.size(); i++){
      MappointPtr mpt = frame_mpts[i];
      if(!mpt || !mpt->IsValid()) continue;

      Eigen::Vector3d keypoint;
      if(!frame->GetKeypointPosition(i, keypoint)) continue;
  
      int mpt_vertex_id = max_frame_id + mpt->GetId();
      if(keypoint(2) < 0){
        EdgeSE3ProjectPoint* e = new EdgeSE3ProjectPoint();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mpt_vertex_id)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(frame_vertex_id)));
        e->setMeasurement(keypoint.head<2>());
        e->setInformation(Eigen::Matrix2d::Identity());
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(thHuberMonoPoint);
        e->fx = fx;
        e->fy = fy;
        e->cx = cx;
        e->cy = cy;
        optimizer.addEdge(e);
        mono_edges.push_back(e);
      }else{
        EdgeSE3ProjectStereoPoint* e = new EdgeSE3ProjectStereoPoint();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mpt_vertex_id)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(frame_vertex_id)));
        e->setMeasurement(keypoint);
        e->setInformation(Eigen::Matrix3d::Identity());
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(thHuberStereoPoint);
        e->fx = fx;
        e->fy = fy;
        e->cx = cx;
        e->cy = cy;
        e->bf = bf;
        optimizer.addEdge(e);
        stereo_edges.push_back(e);
      }
    }

    // 9. line edges
    std::vector<MaplinePtr>& frame_mpls = frame->GetAllMaplines();
    for(size_t i = 0; i < frame_mpls.size(); i++){
      MaplinePtr mpl = frame_mpls[i];
      if(!mpl || !mpl->IsValid()) continue;

      Eigen::Vector4d line_left, line_right;
      if(!frame->GetLine(i, line_left)) continue;

      double cov = mpl->ObverserNum() > 3 ? 0.1 : 0.001;
      int mpl_vertex_id = max_point_id + mpl->GetId();
      if(!frame->GetLineRight(i, line_right)){
        EdgeSE3ProjectLine* e = new EdgeSE3ProjectLine();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mpl_vertex_id)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(frame_vertex_id)));
        e->setMeasurement(line_left);
        e->setInformation(Eigen::Matrix2d::Identity() * cov);
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(thHuberMonoLine);

        e->fx = fx;
        e->fy = fy;
        e->Kv = Kv; 
        optimizer.addEdge(e);
        mono_line_edges.push_back(e);
      }else{
        EdgeStereoSE3ProjectLine* e = new EdgeStereoSE3ProjectLine();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(mpl_vertex_id)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(frame_vertex_id)));
        Vector8d line_2d;
        line_2d << line_left, line_right;
        e->setMeasurement(line_2d);
        e->setInformation(Eigen::Matrix4d::Identity() * cov);
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(thHuberStereoLine);

        e->fx = fx;
        e->fy = fy;
        e->b = bf / fx;
        e->Kv = Kv;
        optimizer.addEdge(e);
        stereo_line_edges.push_back(e);
      }
    }

    // 10. imu edges
    if(_map->IMUInit() && last_frame){
      PreinterationPtr preinteration = frame->GetIMUPreinteration();
      if(!preinteration->Valid()) continue;

      EdgeIMU* e_imu = new EdgeIMU(preinteration);

      int last_frame_vertex_id = last_frame->GetFrameId();
      g2o::HyperGraph::Vertex *vp1 = optimizer.vertex(last_frame_vertex_id);
      g2o::HyperGraph::Vertex *vv1 = optimizer.vertex(last_frame_vertex_id * 3 + max_line_id);
      g2o::HyperGraph::Vertex *vg1 = optimizer.vertex(last_frame_vertex_id * 3 + 1 + max_line_id);
      g2o::HyperGraph::Vertex *va1 = optimizer.vertex(last_frame_vertex_id * 3 + 2 + max_line_id);
      g2o::HyperGraph::Vertex *vp2 = optimizer.vertex(frame_vertex_id);
      g2o::HyperGraph::Vertex *vv2 = optimizer.vertex(frame_vertex_id * 3 + max_line_id);
      g2o::HyperGraph::Vertex *vg2 = optimizer.vertex(frame_vertex_id * 3 + 1 + max_line_id);
      g2o::HyperGraph::Vertex *va2 = optimizer.vertex(frame_vertex_id * 3 + 2 + max_line_id);
      g2o::HyperGraph::Vertex *vG = optimizer.vertex(gravity_direction_id);
      if(!vp1 || !vv1 || !vg1 || !va1 || !vp2 || !vv2 || !vg2 || !va2  || !vG) continue;

      e_imu->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vp1));
      e_imu->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vv1));
      e_imu->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg2));
      e_imu->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va2));
      e_imu->setVertex(4, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vp2));
      e_imu->setVertex(5, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vv2));
      e_imu->setVertex(6, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vG));

      if(last_frame_vertex_id == keyframes.begin()->first){
        g2o::RobustKernelHuber *rki = new g2o::RobustKernelHuber;
        e_imu->setRobustKernel(rki);
        e_imu->setInformation(e_imu->information() * 1e-2);
        rki->setDelta(sqrt(16.92));
      }
      optimizer.addEdge(e_imu);
      imu_edges.push_back(e_imu);


      // bias edges
      EdgeGyr* e_gyr = new EdgeGyr();
      EdgeAcc* e_acc = new EdgeAcc();

      e_gyr->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg1));
      e_acc->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va1));
      e_gyr->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vg2));
      e_acc->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(va2));

      Eigen::Matrix3d info_g = preinteration->Cov.block<3,3>(9,9).inverse();
      Eigen::Matrix3d info_a = preinteration->Cov.block<3,3>(12,12).inverse();
      e_gyr->setInformation(info_g);
      e_acc->setInformation(info_a);

      optimizer.addEdge(e_gyr);
      optimizer.addEdge(e_acc);
      gyr_edges.push_back(e_gyr);
      acc_edges.push_back(e_acc);
    }

    last_frame = frame;
  }


  // solve 
  optimizer.initializeOptimization();
  optimizer.optimize(first_iterations);

  if(point_outlier_rejection)
  {
    // check inlier observations
    for(size_t i=0; i < mono_edges.size(); i++){
      EdgeSE3ProjectPoint* e = mono_edges[i];
      if(e->chi2() > cfg.mono_point || !e->isDepthPositive()){
        e->setLevel(1);
      }
      e->setRobustKernel(0); 
    }

    for(size_t i=0; i < stereo_edges.size(); i++){    
      EdgeSE3ProjectStereoPoint* e = stereo_edges[i];
      if(e->chi2() > cfg.stereo_point || !e->isDepthPositive()){
          e->setLevel(1);
      }
      e->setRobustKernel(0);
    }
  }

  if(line_outlier_rejection){
    for(size_t i=0; i < mono_line_edges.size(); i++){
      EdgeSE3ProjectLine* e = mono_line_edges[i];
      if(e->chi2() > cfg.mono_line){
        e->setLevel(1);
      }
      e->setRobustKernel(0);
    }

    for(size_t i=0; i < stereo_line_edges.size(); i++){    
      EdgeStereoSE3ProjectLine* e = stereo_line_edges[i];
      if(e->chi2() > cfg.stereo_line){
          e->setLevel(1);
      }
      e->setRobustKernel(0);
    }
  }

  if(second_iterations > 0){
    // optimize again without the outliers
    optimizer.initializeOptimization(0);
    optimizer.optimize(second_iterations);

    // check outlier observations   
    std::vector<std::pair<FramePtr, MappointPtr>> mappoint_outliers;
    std::vector<std::pair<FramePtr, MaplinePtr>> mapline_outliers;  
    for(size_t i = 0; i < mono_edges.size(); i++){
      EdgeSE3ProjectPoint* e = mono_edges[i];
      e->computeError();
      if(e->chi2() <= cfg.mono_point && e->isDepthPositive()) continue;

      int mpt_vertex_id = e->vertexXn<0>()->id();
      int frame_vertex_id = e->vertexXn<1>()->id();
      int mpt_id = mpt_vertex_id - max_frame_id;
      int frame_id = frame_vertex_id;
      mappoint_outliers.emplace_back(keyframes[frame_id], mappoints[mpt_id]);
    }

    for(size_t i = 0; i < stereo_edges.size(); i++){    
      EdgeSE3ProjectStereoPoint* e = stereo_edges[i];

      e->computeError();
      if(e->chi2() <= cfg.stereo_point && e->isDepthPositive()) continue;

      int mpt_vertex_id = e->vertexXn<0>()->id();
      int frame_vertex_id = e->vertexXn<1>()->id();
      int mpt_id = mpt_vertex_id - max_frame_id;
      int frame_id = frame_vertex_id;
      mappoint_outliers.emplace_back(keyframes[frame_id], mappoints[mpt_id]);
    }

    for(size_t i = 0; i < mono_line_edges.size(); i++){
      EdgeSE3ProjectLine* e = mono_line_edges[i];
      e->computeError();
      if(e->chi2() <= cfg.mono_line) continue;

      int mpl_vertex_id = e->vertexXn<0>()->id();
      int frame_vertex_id = e->vertexXn<1>()->id();
      int mpl_id = mpl_vertex_id - max_point_id;
      int frame_id = frame_vertex_id;
      mapline_outliers.emplace_back(keyframes[frame_id], maplines[mpl_id]);
    }

    for(size_t i = 0; i < stereo_line_edges.size(); i++){    
      EdgeStereoSE3ProjectLine* e = stereo_line_edges[i];

      e->computeError();
      if(e->chi2() <= cfg.stereo_line) continue;

      int mpl_vertex_id = e->vertexXn<0>()->id();
      int frame_vertex_id = e->vertexXn<1>()->id();
      int mpl_id = mpl_vertex_id - max_point_id;
      int frame_id = frame_vertex_id;
      mapline_outliers.emplace_back(keyframes[frame_id], maplines[mpl_id]);
    }

    std::cout << "mappoint_outliers = " << mappoint_outliers.size() << std::endl;
    std::cout << "mapline_outliers = " << mapline_outliers.size() << std::endl;
    _map->RemoveOutliers(mappoint_outliers);
    _map->RemoveLineOutliers(mapline_outliers);
  }

  // recover optimized data
  // keyframes
  for(auto& kv : keyframes){
    int frame_id = kv.first;
    FramePtr frame = kv.second;

    // pose
    int frame_vertex_id = frame_id;
    VertexVIPose* frame_vertex = static_cast<VertexVIPose*>(optimizer.vertex(frame_vertex_id));
    g2o::SE3Quat SE3quat = frame_vertex->estimate().Tcw.inverse();
    Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
    Twc.block<3, 3>(0, 0) = SE3quat.rotation().toRotationMatrix();
    Twc.block<3, 1>(0, 3) = SE3quat.translation();
    frame->SetPose(Twc);

    if(_map->IMUInit()){
      // velocity
      int velocity_vertex_id = frame_vertex_id * 3 + max_line_id;
      VertexVelocity* velocity_vertex = static_cast<VertexVelocity*>(optimizer.vertex(velocity_vertex_id));
      Eigen::Vector3d velocity = velocity_vertex->estimate();
      frame->SetVelocaity(velocity);

      // biases
      int gyr_vertex_id = frame_vertex_id * 3 + 1 + max_line_id;
      int acc_vertex_id = frame_vertex_id * 3 + 2 + max_line_id;
      VertexGyrBias* gyr_bias_vertex = static_cast<VertexGyrBias*>(optimizer.vertex(gyr_vertex_id));
      VertexAccBias* acc_bias_vertex = static_cast<VertexAccBias*>(optimizer.vertex(acc_vertex_id));
      Eigen::Vector3d gyr_bias = gyr_bias_vertex->estimate();
      Eigen::Vector3d acc_bias = acc_bias_vertex->estimate();
      frame->SetBias(gyr_bias, acc_bias);
    }
  } 

  // 3. points 
  for(auto& kv : mappoints){
    MappointPtr mpt = kv.second;
    if(!mpt || !mpt->IsValid()) continue;

    int point_id = kv.first + max_frame_id;
    g2o::VertexPointXYZ* point_vertex = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(point_id));
    if(point_vertex){
      Eigen::Vector3d pw = point_vertex->estimate();
      mpt->SetPosition(pw);
    }
  }

  // 4. line vertex
  for(auto& kv : maplines){
    MaplinePtr mpl = kv.second;
    if(!mpl || !mpl->IsValid()) continue;

    int line_id = kv.first+max_point_id;
    g2o::VertexLine3D* line_vertex = static_cast<g2o::VertexLine3D*>(optimizer.vertex(line_id));
    if(line_vertex){
      g2o::Line3D line_3d = line_vertex->estimate();
      mpl->SetLine3D(line_3d);
      mpl->SetEndpointsValidStatus(_map->UppdateMapline(mpl));
    }
  }

}