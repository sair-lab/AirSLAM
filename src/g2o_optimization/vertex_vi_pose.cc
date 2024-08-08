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
#include "g2o_optimization/vertex_vi_pose.h"

VIPose::VIPose() : n_it(0){
  Rcw.setIdentity();
  Rcb.setIdentity();
  Rbc.setIdentity();
  Rwb.setIdentity();

  tcw.setZero();
  tcb.setZero();
  tbc.setZero();
  twb.setZero();
}

VIPose::VIPose(const Eigen::Matrix3d& Rcw_, const Eigen::Vector3d& tcw_, const Eigen::Matrix3d& Rcb_, const Eigen::Vector3d& tcb_) : n_it(0){
  SetParam(Rcw_, tcw_, Rcb_, tcb_);
}


VIPose& VIPose::operator =(const VIPose& other){
  Rcw = other.Rcw;
  Rcb = other.Rcb;
  Rbc = other.Rbc;
  Rwb = other.Rwb;
  tcw = other.tcw;
  tcb = other.tcb;
  tbc = other.tbc;
  twb = other.twb;
  Tcw.setTranslation(other.Tcw.translation());
  Tcw.setRotation(other.Tcw.rotation());
  n_it = other.n_it;
  return *this;
}

void VIPose::SetParam(const Eigen::Matrix3d& Rcw_, const Eigen::Vector3d& tcw_, const Eigen::Matrix3d& Rcb_, const Eigen::Vector3d& tcb_){
  Rcw = Rcw_;
  tcw = tcw_;
  Rcb = Rcb_;
  Rbc = Rcb_.transpose();

  tcb = tcb_;
  tbc = -Rbc * tcb_;

  Rwb = Rcw.transpose() * Rcb;
  twb = Rcw.transpose() * (tcb - tcw);

  Eigen::Quaterniond qcw(Rcw);
  Tcw.setRotation(qcw);
  Tcw.setTranslation(tcw);
}

void VIPose::Update(const double* v){
  Eigen::Vector3d dr, dt;
  for(size_t i = 0; i < 3; ++i){
    dr(i) = v[i];
    dt(i) = v[i+3];
  } 

  Eigen::Matrix3d dR;
  SO3Exp(dr, dR);

  twb += Rwb * dt;
  Rwb = Rwb * dR;

  n_it++;
  if(n_it >= 3){
    Rwb = NormalizeRotation(Rwb);
    n_it = 0;
  }

  Rcw = Rcb * Rwb.transpose();
  tcw = tcb - Rcw * twb;

  Eigen::Quaterniond qcw(Rcw);
  Tcw.setRotation(qcw);
  Tcw.setTranslation(tcw);
}

VertexVIPose::VertexVIPose() : g2o::BaseVertex<6, VIPose>(){
}

bool VertexVIPose::read(std::istream& is) {
  return false; 
}

bool VertexVIPose::write(std::ostream& os) const {
  return false; 
}