#ifndef G2O_OPTIMIZATION_H_
#define G2O_OPTIMIZATION_H_

#include <vector>

#include "read_configs.h"
#include "camera.h"
#include "frame.h"
#include "mappoint.h"
#include "map.h"
#include "g2o_optimization/types.h"

void AddFrameVertex(FramePtr frame, MapOfPoses& poses, int id_camera, bool fix_this_frame);
void AddFrameVertex(FramePtr frame, MapOfPoses& poses, int id_camera, MapOfVelocity& velocities, MapOfBias& biases, 
    VectorOfIMUConstraints& imu_constraints, bool fix_this_frame, bool add_imu_constraint, bool use_updated_bias=false);

// pose, velocaity and bias of same frame have same key, i.e. id.
void LocalmapOptimization(MapOfPoses& poses, MapOfPoints3d& points, MapOfLine3d& lines, 
    MapOfVelocity& velocities, MapOfBias& biases, std::vector<CameraPtr>& camera_list, 
    VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints, 
    VectorOfMonoLineConstraints& mono_line_constraints, VectorOfStereoLineConstraints& stereo_line_constraints,
    VectorOfIMUConstraints& imu_constraints, const Eigen::Matrix3d& Rwg, const OptimizationConfig& cfg);

int FrameOptimization(MapOfPoses& poses, MapOfPoints3d& points, MapOfLine3d& lines,
    MapOfVelocity& velocities, MapOfBias& biases, std::vector<CameraPtr>& camera_list, 
    VectorOfMonoPointConstraints& mono_point_constraints, VectorOfStereoPointConstraints& stereo_point_constraints, 
    VectorOfMonoLineConstraints& mono_line_constraints, VectorOfStereoLineConstraints& stereo_line_constraints,
    VectorOfIMUConstraints& imu_constraints, Eigen::Matrix3d& Rwg, const OptimizationConfig& cfg);

bool IMUInitialization(MapOfPoses& poses, MapOfVelocity& velocities, Bias& bias, std::vector<CameraPtr>& camera_list,
    VectorOfIMUConstraints& imu_constraints, Eigen::Matrix3d& Rwg);

int SolvePnPWithCV(FramePtr frame, std::vector<MappointPtr>& mappoints, Eigen::Matrix4d& pose, std::vector<int>& inliers);

// frames: [n, n-1, n-2....]
bool ComputeGyrBias(std::vector<FramePtr>& frames, Eigen::Vector3d& dbg);
bool ComputeVelocity(std::vector<FramePtr>& frames, Eigen::Vector3d& gw);

// frames: [n, n-1, n-2....], for debugging
void ValidateGyrBias(std::vector<FramePtr>& frames); 
void ValidateVelocity(std::vector<FramePtr>& frames, Eigen::VectorXd x);
void ValidateError(std::vector<FramePtr>& frames, const Eigen::Vector3d& gw);
bool ValidateError(MapOfPoses& poses, MapOfVelocity& velocities, Bias& bias, std::vector<CameraPtr>& camera_list,
    VectorOfIMUConstraints& imu_constraints, Eigen::Matrix3d& Rwg, const Eigen::Vector3d& prior_gyr_bias, const Eigen::Vector3d& prior_acc_bias);
void ValidateIMUInitialization(std::vector<FramePtr>& frames);


// for offline map refinement
void PoseGraphOptimization(MapOfPoses& poses, std::vector<CameraPtr>& camera_list, VectorOfRelativePoseConstraints& relative_pose_constraints);
void GlobalBA(MapPtr _map, const OptimizationConfig& cfg, bool point_outlier_rejection, 
    bool line_outlier_rejection, int first_iterations, int second_iterations);

#endif  // G2O_OPTIMIZATION_H_