#include <istream>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "imu.h"
#include "camera.h"

void Hat(Eigen::Matrix3d& m, const Eigen::Vector3d& v){
  m << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
}

Eigen::Matrix3d NormalizeRotation(const Eigen::Matrix3d &R){
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
}

void ComputerDeltaR(const Eigen::Vector3d& rv, Eigen::Matrix3d& delta_R, Eigen::Matrix3d& Jr){
  double d = rv.norm();
  double d2 = d * d;
  Eigen::Matrix3d rv_hat;
  Hat(rv_hat, rv);
  if(d < IMU_EPS){
    delta_R = Eigen::Matrix3d::Identity() + rv_hat;
    Jr = Eigen::Matrix3d::Identity();
  }else{
    delta_R = Eigen::Matrix3d::Identity() + (sin(d)/d)*rv_hat + ((1.0-cos(d))/d2)*rv_hat*rv_hat;
    Jr = Eigen::Matrix3d::Identity() - ((1.0-cos(d))/d2)*rv_hat + ((d-sin(d))/(d2*d))*rv_hat*rv_hat;
  }
}

Eigen::Vector3d VectorInterpolation(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1, double t0, double t1, double t){
  return ((t1-t)*v0+(t-t0)*v1)/(t1-t0);
}

void SO3Exp(const Eigen::Vector3d& v, Eigen::Matrix3d& R){
  double theta = v.norm();
  Eigen::Matrix3d Omega;
  Hat(Omega, v);
  Eigen::Matrix3d Omega2 = Omega * Omega;
  Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
  if(theta < IMU_EPS){
    R = I3 + Omega + 0.5 * Omega2;
  }else{
    double sin_theta = std::sin(theta);
    double theta2 = theta * theta;
    double cos_theta = std::cos(theta);
    R = I3 + (sin_theta / theta) * Omega + ((1 - cos_theta) / theta2) * Omega2;
  }
  R = NormalizeRotation(R);
}

void SO3Log(const Eigen::Matrix3d& R, Eigen::Vector3d& v){
  double d = 0.5 * (R(0, 0) + R(1, 1) + R(2, 2) - 1);
  Eigen::Vector3d delta_R;
  delta_R << R(2, 1) - R(1, 2),  R(0, 2) - R(2, 0), R(1, 0) - R(0, 1);
  if(std::abs(d) > 0.99999){
    v = 0.5 * delta_R;
  } else {
    double theta = std::acos(d);
    v = theta / (2 * std::sqrt(1 - d * d)) * delta_R;
  }
}

Preinteration::Preinteration(){
  Initialize();
  start_time = -1;
  end_time = -1;
  noise_matrix.setIdentity(6);
  walk_matrix.setIdentity(6);
  ba.setZero();
  bg.setZero();
}

Preinteration::Preinteration(const Eigen::Vector3d& ba_, const Eigen::Vector3d& bg_){
  Initialize();
  start_time = -1;
  end_time = -1;
  noise_matrix.setIdentity(6);
  walk_matrix.setIdentity(6);
  ba = ba_;
  bg = bg_;
}

Preinteration& Preinteration::operator=(const Preinteration& preinteration){
  start_time = preinteration.start_time; 
  end_time = preinteration.end_time; 
  noise_matrix = preinteration.noise_matrix; 
  walk_matrix = preinteration.walk_matrix; 
  ba = preinteration.ba; 
  bg = preinteration.bg;   
  dba = preinteration.dba; 
  dbg = preinteration.dbg; 
  dT = preinteration.dT; 
  dR = preinteration.dR; 
  dP = preinteration.dP; 
  dV = preinteration.dV; 
  JRg = preinteration.JRg; 
  JVg = preinteration.JVg; 
  JVa = preinteration.JVa; 
  JPg = preinteration.JPg; 
  JPa = preinteration.JPa; 
  Cov = preinteration.Cov; 
  dt_list = preinteration.dt_list; 
  gyr_list = preinteration.gyr_list; 
  acc_list = preinteration.acc_list; 
  return *this;
}

void Preinteration::Initialize(){
  dba.setZero();
  dbg.setZero();
  dP.setZero();
  dV.setZero();
  JRg.setZero();
  JVg.setZero();
  JVa.setZero();
  JPg.setZero();
  JPa.setZero();
  Cov.setZero();

  dT = 0;
  dR.setIdentity();
}

void Preinteration::SetNoiseAndWalk(double gyr_noise, double acc_noise, double gyr_walk, double acc_walk){
  double gyr_noise2 = gyr_noise * gyr_noise;
  double acc_noise2 = acc_noise * acc_noise;
  double gyr_walk2 = gyr_walk * gyr_walk;
  double acc_walk2 = acc_walk * acc_walk;

  noise_matrix.diagonal() << gyr_noise2, gyr_noise2, gyr_noise2, acc_noise2, acc_noise2, acc_noise2;
  walk_matrix.diagonal() << gyr_walk2, gyr_walk2, gyr_walk2, acc_walk2, acc_walk2, acc_walk2;
}

void Preinteration::SetBias(const Eigen::Vector3d& gyr_bias, const Eigen::Vector3d& acc_bias, bool to_repropagate){
  bg = gyr_bias;
  ba = acc_bias;

  dbg = Eigen::Vector3d::Zero();
  dba = Eigen::Vector3d::Zero();

  if(to_repropagate){
    Initialize();
    Repropagate();
  }
}

void Preinteration::UpdateBias(const Eigen::Vector3d& gyr_bias, const Eigen::Vector3d& acc_bias){
  dbg = gyr_bias - bg;
  dba = acc_bias - ba;
}

void Preinteration::Propagate(double dt, const Eigen::Vector3d &acc_m, const Eigen::Vector3d &gyr_m, bool save_m){
  // minus bias
  Eigen::Vector3d acc = acc_m - ba;
  Eigen::Vector3d gyr = gyr_m - bg;

  // update position and velocity
  dP = dP + dV*dt + 0.5f*dR*acc*dt*dt;
  dV = dV + dR*acc*dt;

  // A and B are used to compute covariance
  Matrix9d A;
  A.setIdentity();
  Eigen::Matrix<double, 9, 6> B;
  B.setZero();

  Eigen::Matrix3d acc_hat;
  Hat(acc_hat, acc);
  A.block<3,3>(3,0) = -dR*dt*acc_hat;
  A.block<3,3>(6,0) = -0.5*dR*dt*dt*acc_hat;
  A.block<3,3>(6,3) = Eigen::DiagonalMatrix<double,3>(dt, dt, dt);
  B.block<3,3>(3,3) = dR*dt;
  B.block<3,3>(6,3) = 0.5*dR*dt*dt;

  // update jacobians
  JPa = JPa + JVa*dt -0.5f*dR*dt*dt;
  JPg = JPg + JVg*dt -0.5f*dR*dt*dt*acc_hat*JRg;
  JVa = JVa - dR*dt;
  JVg = JVg - dR*dt*acc_hat*JRg;

  // update dR
  Eigen::Matrix3d delta_R, Jr;
  ComputerDeltaR(gyr*dt, delta_R, Jr);
  dR = NormalizeRotation(dR * delta_R);

  A.block<3,3>(0,0) = delta_R.transpose();
  B.block<3,3>(0,0) = Jr * dt;

  // update covariance
  Cov.block<9,9>(0,0) = A * Cov.block<9,9>(0,0) * A.transpose() + B * noise_matrix * B.transpose();
  Cov.block<6,6>(9,9) += walk_matrix;

  // update rotation jacobian wrt bias correction
  JRg = delta_R.transpose() * JRg - Jr * dt;

  // total integrated time
  dT += dt;

  // save measurements
  if(save_m){
    dt_list.push_back(dt);
    gyr_list.push_back(gyr_m);
    acc_list.push_back(acc_m);
  }
}

void Preinteration::Repropagate(){
  for(size_t i = 0; i < dt_list.size(); i++){
    Propagate(dt_list[i], acc_list[i], gyr_list[i], false);
  }
}

void Preinteration::AddBatchData(const ImuDataList& batch_imu_data, double t0, double t1){
  if(batch_imu_data.empty()) return;
  start_time = start_time > 0 ? start_time : t0;
  assert(std::abs(end_time-t0)<1e-5 || end_time < 0);
  end_time = t1;
  size_t i = 0;

  Eigen::Vector3d mid_gyr, mid_acc;
  double mid_t, dt;
  for(; i < batch_imu_data.size()-1; i++){
    if(batch_imu_data[i+1].timestamp < t0){
      continue;
    }else if(batch_imu_data[i].timestamp < t0){
      mid_t = 0.5 * (t0 + batch_imu_data[i+1].timestamp);
      dt = batch_imu_data[i+1].timestamp - t0;
    }else if(batch_imu_data[i].timestamp > t1){
      break;
    }else if(batch_imu_data[i+1].timestamp > t1){
      mid_t = 0.5 * (t1 + batch_imu_data[i].timestamp);
      dt = t1 - batch_imu_data[i].timestamp;
    }else{
      mid_t = 0.5 * (batch_imu_data[i+1].timestamp + batch_imu_data[i].timestamp);
      dt = batch_imu_data[i+1].timestamp - batch_imu_data[i].timestamp;
    }
    
    mid_gyr = VectorInterpolation(batch_imu_data[i].gyr, batch_imu_data[i+1].gyr, batch_imu_data[i].timestamp, batch_imu_data[i+1].timestamp, mid_t);
    mid_acc = VectorInterpolation(batch_imu_data[i].acc, batch_imu_data[i+1].acc, batch_imu_data[i].timestamp, batch_imu_data[i+1].timestamp, mid_t);

    Propagate(dt, mid_acc, mid_gyr);
  }
}

const Eigen::Matrix3d Preinteration::GetDeltaRotation(const Eigen::Vector3d& gyr_bias){
  Eigen::Matrix3d ddR;
  SO3Exp(JRg * (gyr_bias - bg), ddR);
  return NormalizeRotation(dR * ddR);
}

const Eigen::Vector3d Preinteration::GetDeltaPosition(const Eigen::Vector3d& gyr_bias, const Eigen::Vector3d& acc_bias){
  return dP + JPg * (gyr_bias - bg) + JPa * (acc_bias - ba);
}

const Eigen::Vector3d Preinteration::GetDeltaVelocity(const Eigen::Vector3d& gyr_bias, const Eigen::Vector3d& acc_bias){
  return dV + JVg * (gyr_bias - bg) + JVa * (acc_bias - ba);
}

const Eigen::Matrix3d Preinteration::GetUpdatedDeltaRotation(){
  Eigen::Matrix3d ddR;
  SO3Exp(JRg * dbg, ddR);
  return NormalizeRotation(dR * ddR);
}

const Eigen::Vector3d Preinteration::GetUpdatedDeltaPosition(){
  return dP + JPg * dbg + JPa * dba;
}

const Eigen::Vector3d Preinteration::GetUpdatedDeltaVelocity(){
  return dV + JVg * dbg + JVa * dba;
}

const void Preinteration::GetUpdatedBias(Eigen::Vector3d& gyr_bias, Eigen::Vector3d& acc_bias){
  gyr_bias = bg + dbg;
  acc_bias = ba + dba;
}

bool Preinteration::Valid(){
  return (start_time >=0) && (end_time > start_time);
}

void Preinteration::Reset(){
  Initialize();
  start_time = -1;
  end_time = -1;
  ba.setZero();
  bg.setZero();

  dt_list.clear();
  gyr_list.clear();
  acc_list.clear();
}

void Preinteration::Predict(const Eigen::Matrix4d& Twb0, const Eigen::Vector3d& vwb0, Eigen::Matrix4d& Twb1, Eigen::Vector3d& vwb1){
  if(Valid()){
    Eigen::Matrix3d Rwb0 = Twb0.block<3, 3>(0, 0);
    Eigen::Vector3d twb0 = Twb0.block<3, 1>(0, 3);
    const Eigen::Vector3d g(0, 0, -Camera::IMU_G_VALUE);

    Twb1 = Eigen::Matrix4d::Identity();
    Twb1.block<3, 3>(0, 0) = NormalizeRotation(Rwb0 * GetUpdatedDeltaRotation());
    Twb1.block<3, 1>(0, 3) = twb0 + vwb0 * dT + 0.5 * dT * dT * g + Rwb0 * GetUpdatedDeltaPosition();
    vwb1 = vwb0 + dT * g + Rwb0 * GetUpdatedDeltaVelocity();
  }else{
    Twb1 = Twb0;
    vwb1 = vwb0;
  }
}