#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <ros/ros.h>


#include <fstream>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include "g2o_optimization/edge_project_line.h"
#include "g2o_optimization/edge_project_stereo_line.h"
#include "g2o_optimization/g2o_optimization.h"

int main(int argc, char **argv) {
  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);
  std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linear_solver;
  linear_solver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver)));
  optimizer.setAlgorithm(solver);



  return 0;
}
