#include <Eigen/Core> 
#include <Eigen/Geometry> 
#include <opencv2/core/eigen.hpp>
#include <g2o/types/slam3d/isometry3d_mappings.h>

#include "utils.h"
#include "camera.h"
#include "g2o_optimization/edge_imu.h"
#include "g2o_optimization/vertex_imu.h"
#include "g2o_optimization/vertex_vi_pose.h"

EdgeIMU::EdgeIMU(PreinterationPtr preinteration_): preinteration(preinteration_){
  resize(7);
  g << 0, 0, -Camera::IMU_G_VALUE;

  Matrix9d info = preinteration->Cov.block<9,9>(0,0).cast<double>().inverse();
  info = (info + info.transpose()) / 2;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,9,9> > es(info);
  Vector9d eigs = es.eigenvalues();
  for(int i=0; i<9; i++){
    if(eigs[i] < 1e-12) eigs[i]=0;
  }

  info = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
  setInformation(info);
  // std::cout << "info = " << info << std::endl;
}

void EdgeIMU::computeError() {
  const VertexVIPose* vp1 = static_cast<const VertexVIPose*>(_vertices[0]);
  const VertexVelocity* vv1= static_cast<const VertexVelocity*>(_vertices[1]);
  const VertexGyrBias* vg2= static_cast<const VertexGyrBias*>(_vertices[2]);
  const VertexAccBias* va2= static_cast<const VertexAccBias*>(_vertices[3]);
  const VertexVIPose* vp2 = static_cast<const VertexVIPose *>(_vertices[4]);
  const VertexVelocity* vv2= static_cast<const VertexVelocity*>(_vertices[5]);
  const VertexGDirection* vg = static_cast<const VertexGDirection*>(_vertices[6]);

  const Eigen::Vector3d gyr_bias = vg2->estimate();
  const Eigen::Vector3d acc_bias = va2->estimate();
  const Eigen::Matrix3d dR = preinteration->GetDeltaRotation(gyr_bias);
  const Eigen::Vector3d dV = preinteration->GetDeltaVelocity(gyr_bias, acc_bias);
  const Eigen::Vector3d dP = preinteration->GetDeltaPosition(gyr_bias, acc_bias);

  Eigen::Vector3d gw = vg->estimate().Rwg * g;
  double dt = preinteration->dT;
  Eigen::Vector3d er;

  SO3Log(dR.transpose()*vp1->estimate().Rwb.transpose()*vp2->estimate().Rwb, er);
  Eigen::Vector3d ev = vp1->estimate().Rwb.transpose() * (vv2->estimate() - vv1->estimate() - gw*dt) - dV;
  Eigen::Vector3d ep = vp1->estimate().Rwb.transpose() * (vp2->estimate().twb - vp1->estimate().twb - vv1->estimate()*dt - gw*dt*dt/2) - dP;

  _error << er, ev, ep;

  // std::cout << "---------------------------- EdgeIMU::computeError ------------------------------------" << std::endl;
  // std::cout << "id1 = " << vp1->id() << ", id2 = " << vp2->id() << std::endl;
  // std::cout << "vp1->estimate().twb = " << vp1->estimate().twb.transpose() << std::endl;
  // std::cout << "vp2->estimate().twb = " << vp2->estimate().twb.transpose() << std::endl;
  // std::cout << "vv1->estimate().twb = " << vv1->estimate().transpose() << std::endl;
  // std::cout << "dt = " << dt << std::endl;
  // std::cout << "gw = " << gw << std::endl;
  // std::cout << "dP = " << dP.transpose() << std::endl;

  // Eigen::Vector3d ep_tmp1 = vp2->estimate().twb - vp1->estimate().twb - vv1->estimate()*dt - gw*dt*dt/2;
  // std::cout << "ep_tmp1 = " << ep_tmp1.transpose() << std::endl;

  // std::cout << "ep_tmp1.norm() = " << ep_tmp1.norm() << std::endl;
  // std::cout << "dP.norm() = " << dP.norm() << std::endl;


  // // std::cout << "vg->estimate().Rwg = " << vg->estimate().Rwg << std::endl;
  // std::cout << "acc_bias = " << acc_bias.transpose() << std::endl;


  // std::cout << "frame_id = " << vp2->id() << std::endl;
  // std::cout << "er = " << er.transpose() << ", ev = " << ev.transpose() << ", ep = " << ep.transpose() << std::endl;
  // std::cout << "v = " << vv2->estimate().transpose() << ", p = " << vp2->estimate().twb.transpose() << std::endl;
  // std::cout << "---------------------------------------------------------------------------------------" << std::endl;
}

void EdgeIMU::PrintError() {
  const VertexVIPose* vp1 = static_cast<const VertexVIPose*>(_vertices[0]);
  const VertexVelocity* vv1= static_cast<const VertexVelocity*>(_vertices[1]);
  const VertexGyrBias* vg2= static_cast<const VertexGyrBias*>(_vertices[2]);
  const VertexAccBias* va2= static_cast<const VertexAccBias*>(_vertices[3]);
  const VertexVIPose* vp2 = static_cast<const VertexVIPose *>(_vertices[4]);
  const VertexVelocity* vv2= static_cast<const VertexVelocity*>(_vertices[5]);
  const VertexGDirection* vg = static_cast<const VertexGDirection*>(_vertices[6]);

  const Eigen::Vector3d gyr_bias = vg2->estimate();
  const Eigen::Vector3d acc_bias = va2->estimate();
  const Eigen::Matrix3d dR = preinteration->GetDeltaRotation(gyr_bias);
  const Eigen::Vector3d dV = preinteration->GetDeltaVelocity(gyr_bias, acc_bias);
  const Eigen::Vector3d dP = preinteration->GetDeltaPosition(gyr_bias, acc_bias);

  Eigen::Vector3d gw = vg->estimate().Rwg * g;
  double dt = preinteration->dT;
  Eigen::Vector3d er;

  SO3Log(dR.transpose()*vp1->estimate().Rwb.transpose()*vp2->estimate().Rwb, er);
  Eigen::Vector3d ev = vp1->estimate().Rwb.transpose() * (vv2->estimate() - vv1->estimate() - gw*dt) - dV;
  Eigen::Vector3d ep = vp1->estimate().Rwb.transpose() * (vp2->estimate().twb - vp1->estimate().twb - vv1->estimate()*dt - gw*dt*dt/2) - dP;

  // std::cout << "---------------------------- EdgeIMU::computeError ------------------------------------" << std::endl;
  // std::cout << "id1 = " << vp1->id() << ", id2 = " << vp2->id() << std::endl;
  // std::cout << "vp1->estimate().twb = " << vp1->estimate().twb.transpose() << std::endl;
  // std::cout << "vp2->estimate().twb = " << vp2->estimate().twb.transpose() << std::endl;
  // std::cout << "vv1->estimate().twb = " << vv1->estimate().transpose() << std::endl;
  // std::cout << "dt = " << dt << std::endl;
  // std::cout << "gw = " << gw << std::endl;
  // std::cout << "dP = " << dP.transpose() << std::endl;

  // Eigen::Vector3d ep_tmp1 = vp2->estimate().twb - vp1->estimate().twb - vv1->estimate()*dt - gw*dt*dt/2;
  // std::cout << "ep_tmp1 = " << ep_tmp1.transpose() << std::endl;

  // std::cout << "ep_tmp1.norm() = " << ep_tmp1.norm() << std::endl;
  // std::cout << "dP.norm() = " << dP.norm() << std::endl;


  // // std::cout << "vg->estimate().Rwg = " << vg->estimate().Rwg << std::endl;
  // std::cout << "acc_bias = " << acc_bias.transpose() << std::endl;

  // std::cout << "frame_id = " << vp2->id() << std::endl;
  std::cout << "er = " << er.transpose() << ", ev = " << ev.transpose() << ", ep = " << ep.transpose() << std::endl;
  // std::cout << "v = " << vv2->estimate().transpose() << ", p = " << vp2->estimate().twb.transpose() << std::endl;
  // std::cout << "---------------------------------------------------------------------------------------" << std::endl;
}

void EdgeGyr::computeError(){
  const VertexGyrBias* vg1= static_cast<const VertexGyrBias*>(_vertices[0]);
  const VertexGyrBias* vg2= static_cast<const VertexGyrBias*>(_vertices[1]);
  _error = vg2->estimate() - vg1->estimate();

  // std::cout << "---------------------------- EdgeGyr::computeError ------------------------------------" << std::endl;
  // std::cout << "id1 = " << vg1->id() << ", id2 = " << vg2->id() << std::endl;
  // std::cout << "vg2->estimate() = " << vg2->estimate().transpose() << ", vg1->estimate() = " << vg1->estimate().transpose() << ", _error = " << _error.transpose() << std::endl;
  // std::cout << "---------------------------------------------------------------------------------------" << std::endl;
}

void EdgeAcc::computeError(){
  const VertexAccBias* va1= static_cast<const VertexAccBias*>(_vertices[0]);
  const VertexAccBias* va2= static_cast<const VertexAccBias*>(_vertices[1]);
  _error = va2->estimate() - va1->estimate();
}