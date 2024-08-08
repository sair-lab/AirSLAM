#include <Eigen/Core> 
#include <Eigen/Geometry> 
#include <opencv2/core/eigen.hpp>
#include <g2o/core/base_binary_edge.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/types/sba/g2o_types_sba_api.h>
#include <g2o/types/sba/vertex_se3_expmap.h>
#include <g2o/types/slam3d_addons/line3d.h>

#include "utils.h"
#include "imu.h"
#include "g2o_optimization/vertex_line3d.h"
#include "g2o_optimization/vertex_imu.h"

// velocity
VertexVelocity::VertexVelocity(){
}

VertexVelocity::VertexVelocity(const Eigen::Vector3d& velocity){
  setEstimate(velocity);
}

bool VertexVelocity::read(std::istream& is) {
  Eigen::Vector3d lv;
  bool state = g2o::internal::readVector(is, lv);
  setEstimate(lv);
  return state;
}

bool VertexVelocity::write(std::ostream& os) const {
  return g2o::internal::writeVector(os, _estimate);
}

void VertexVelocity::oplusImpl(const double* update_) {
  Eigen::Vector3d update;
  update << update_[0], update_[1], update_[2];

  // std::cout << "before, id = " << id() << ", v = " << estimate().transpose() << ", update = " << update.transpose() << std::endl;

  setEstimate(estimate()+update);
  // std::cout << "after, id = " << id() << ", v = " << estimate().transpose() << ", update = " << update.transpose() << std::endl;
}

int VertexVelocity::estimateDimension() const { 
  return 3; 
}


// gyroscope bias
VertexGyrBias::VertexGyrBias(){
}

VertexGyrBias::VertexGyrBias(const Eigen::Vector3d& gyr_bias){
  setEstimate(gyr_bias);
}

bool VertexGyrBias::read(std::istream& is){
  Eigen::Vector3d lv;
  bool state = g2o::internal::readVector(is, lv);
  setEstimate(lv);
  return state;
}

bool VertexGyrBias::write(std::ostream& os) const {
  return g2o::internal::writeVector(os, _estimate);
}

void VertexGyrBias::setToOriginImpl(){ 
  _estimate.setZero(); 
}

void VertexGyrBias::oplusImpl(const double* update_){
  Eigen::Vector3d update;
  update << update_[0], update_[1], update_[2];
  setEstimate(estimate()+update);
}

int VertexGyrBias::estimateDimension() const { 
  return 3; 
}


// accelerometer bias
VertexAccBias::VertexAccBias(){
}

VertexAccBias::VertexAccBias(const Eigen::Vector3d& gyr_bias){
  setEstimate(gyr_bias);
}

bool VertexAccBias::read(std::istream& is){
  Eigen::Vector3d lv;
  bool state = g2o::internal::readVector(is, lv);
  setEstimate(lv);
  return state;
}

bool VertexAccBias::write(std::ostream& os) const{
  return g2o::internal::writeVector(os, _estimate);
}

void VertexAccBias::setToOriginImpl(){ 
  _estimate.setZero(); 
}

void VertexAccBias::oplusImpl(const double* update_){
  Eigen::Vector3d update;
  update << update_[0], update_[1], update_[2];
  setEstimate(estimate()+update);
}

int VertexAccBias::estimateDimension() const { 
  return 3; 
}


// gravity direction
GDirection::GDirection(){
}

GDirection::GDirection(const Eigen::Matrix3d& Rwg_) : Rwg(Rwg_){
}

void GDirection::Update(const double *update_){
  Eigen::Vector3d update;
  update << update_[0], update_[1], 0.0;
  Eigen::Matrix3d dR;
  SO3Exp(update, dR);
  Rwg = Rwg * dR;
}

VertexGDirection::VertexGDirection(){
}

VertexGDirection::VertexGDirection(const Eigen::Matrix3d& Rwg){
  setEstimate(GDirection(Rwg));
}

bool VertexGDirection::read(std::istream &is){ 
  return false; 
}

bool VertexGDirection::write(std::ostream &os) const { 
  return false; 
}

void VertexGDirection::setToOriginImpl(){
}

void VertexGDirection::oplusImpl(const double *update_){
  _estimate.Update(update_);
  updateCache();
}