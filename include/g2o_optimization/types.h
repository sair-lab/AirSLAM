
#ifndef OPTIMIZATION_3D_TYPES_H_
#define OPTIMIZATION_3D_TYPES_H_

#include <istream>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/slam3d_addons/types_slam3d_addons.h>

#include "utils.h"
#include "imu.h"

struct Pose3d {
  bool fixed;
  int id_camera;
  Eigen::Vector3d p;
  Eigen::Matrix3d R;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Pose3d() {}
  Pose3d& operator =(Pose3d& other){
		fixed = other.fixed;
		id_camera = other.id_camera;
		p = other.p;
		R = other.R;
		return *this;
	}
};
typedef std::map<int, Pose3d, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Pose3d>>> MapOfPoses;


struct Position3d{
  bool fixed;
  Eigen::Vector3d p;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Position3d() {}
  Position3d& operator =(Position3d& other){
		fixed = other.fixed;
		p = other.p;
		return *this;
	}
};
typedef std::map<int, Position3d, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Position3d>>> MapOfPoints3d;


struct MonoPointConstraint {
  int id_pose;
  int id_point;
  int id_camera;
  bool inlier;

  // x_left, y_left
  Eigen::Vector2d keypoint;  
  double pixel_sigma;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MonoPointConstraint() {}
  MonoPointConstraint& operator =(MonoPointConstraint& other){
		id_pose = other.id_pose;
		id_point = other.id_point;
		id_camera = other.id_camera;
		inlier = other.inlier;
		keypoint = other.keypoint;
		pixel_sigma = other.pixel_sigma;
		return *this;
	}
};
typedef std::shared_ptr<MonoPointConstraint> MonoPointConstraintPtr;
typedef std::vector<MonoPointConstraintPtr> VectorOfMonoPointConstraints;


struct StereoPointConstraint {
  int id_pose;
  int id_point;
  int id_camera;
  bool inlier;

  // x_left, y_left, x_right
  Eigen::Vector3d keypoint;  
  double pixel_sigma;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  StereoPointConstraint() {}
  StereoPointConstraint& operator =(StereoPointConstraint& other){
		id_pose = other.id_pose;
		id_point = other.id_point;
		id_camera = other.id_camera;
		inlier = other.inlier;
		keypoint = other.keypoint;
		pixel_sigma = other.pixel_sigma;
		return *this;
	}
};
typedef std::shared_ptr<StereoPointConstraint> StereoPointConstraintPtr;
typedef std::vector<StereoPointConstraintPtr> VectorOfStereoPointConstraints;


struct Line3d{
  bool fixed;
  g2o::Line3D line_3d;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Line3d() {}
  Line3d& operator =(Line3d& other){
		fixed = other.fixed;
		line_3d = other.line_3d;
		return *this;
	}
};
typedef std::map<int, Line3d, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Line3d>>> MapOfLine3d;


struct MonoLineConstraint {
  int id_pose;
  int id_line;
  int id_camera;
  bool inlier;

  // x_left, y_left
  Eigen::Vector4d line_2d;  
  double pixel_sigma;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MonoLineConstraint() {}
  MonoLineConstraint& operator =(MonoLineConstraint& other){
		id_pose = other.id_pose;
		id_line = other.id_line;
		id_camera = other.id_camera;
		inlier = other.inlier;
		line_2d = other.line_2d;
		pixel_sigma = other.pixel_sigma;
		return *this;
	}
};
typedef std::shared_ptr<MonoLineConstraint> MonoLineConstraintPtr;
typedef std::vector<MonoLineConstraintPtr> VectorOfMonoLineConstraints;


struct StereoLineConstraint {
  int id_pose;
  int id_line;
  int id_camera;
  bool inlier;

  Vector8d line_2d;  
  double pixel_sigma;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  StereoLineConstraint() {}
  StereoLineConstraint& operator =(StereoLineConstraint& other){
		id_pose = other.id_pose;
		id_line = other.id_line;
		id_camera = other.id_camera;
		inlier = other.inlier;
		line_2d = other.line_2d;
		pixel_sigma = other.pixel_sigma;
		return *this;
	}
};
typedef std::shared_ptr<StereoLineConstraint> StereoLineConstraintPtr;
typedef std::vector<StereoLineConstraintPtr> VectorOfStereoLineConstraints;


// for imu
struct Velocity{
  bool fixed;
  Eigen::Vector3d velocity;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Velocity() {}
  Velocity& operator =(Velocity& other){
		fixed = other.fixed;
		velocity = other.velocity;
		return *this;
	}
};
typedef std::map<int, Velocity, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Velocity>>> MapOfVelocity;


struct Bias{
  bool fixed;
  Eigen::Vector3d gyr_bias;
  Eigen::Vector3d acc_bias;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Bias() {}
  Bias& operator =(Bias& other){
		fixed = other.fixed;
		gyr_bias = other.gyr_bias;
		acc_bias = other.acc_bias;
		return *this;
	}
};
typedef std::map<int, Bias, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Bias>>> MapOfBias;


struct ImuConstraint{
  int id_pose1;
  int id_pose2;
  int id_camera1;
  int id_camera2;
  PreinterationPtr preinteration;
  
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuConstraint() {}
  ImuConstraint& operator =(ImuConstraint& other){
		id_pose1 = other.id_pose1;
		id_pose2 = other.id_pose2;
		id_camera1 = other.id_camera1;
		id_camera2 = other.id_camera2;
		preinteration = other.preinteration;
		return *this;
	}
};
typedef std::shared_ptr<ImuConstraint> IMUConstraintPtr;
typedef std::vector<IMUConstraintPtr> VectorOfIMUConstraints;

struct RelativePoseConstraint{
  int id_pose1;
  int id_pose2;
  int id_camera1;
  int id_camera2;

  Eigen::Matrix3d Rc1c2;
  Eigen::Vector3d tc1c2;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RelativePoseConstraint() {}
  RelativePoseConstraint& operator =(RelativePoseConstraint& other){
		id_pose1 = other.id_pose1;
		id_pose2 = other.id_pose2;
		id_camera1 = other.id_camera1;
		id_camera2 = other.id_camera2;
		Rc1c2 = other.Rc1c2;
		tc1c2 = other.tc1c2;
		return *this;
	}
};
typedef std::shared_ptr<RelativePoseConstraint> RelativePoseConstraintPtr;
typedef std::vector<RelativePoseConstraintPtr> VectorOfRelativePoseConstraints;

#endif  // OPTIMIZATION_3D_TYPES_H_
