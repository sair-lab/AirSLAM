
#ifndef IMU_H_
#define IMU_H_

#include <istream>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>

#include "utils.h"

const double IMU_EPS = 1e-4; 

struct ImuData {
  double timestamp;
  Eigen::Vector3d gyr;
  Eigen::Vector3d acc;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuData() {}
  ImuData& operator =(const ImuData& other){
		timestamp = other.timestamp;
		gyr = other.gyr;
		acc = other.acc;
		return *this;
	}
};
typedef std::vector<ImuData> ImuDataList;

void Hat(Eigen::Matrix3d& m, const Eigen::Vector3d& v);
Eigen::Matrix3d NormalizeRotation(const Eigen::Matrix3d &R);
void ComputerDeltaR(const Eigen::Vector3d& rv, Eigen::Matrix3d& delta_R, Eigen::Matrix3d& Jr);
Eigen::Vector3d VectorInterpolation(const Eigen::Vector3d& v0, const Eigen::Vector3d& v1, double t0, double t1, double t);
void SO3Exp(const Eigen::Vector3d& v, Eigen::Matrix3d& R);
void SO3Log(const Eigen::Matrix3d& R, Eigen::Vector3d& v);
// void SE3Exp(const Vector6d& v, Eigen::Matrix3d& R, Eigen::Vector3d& t);

class Preinteration{
public:
  Preinteration();
  Preinteration(const Eigen::Vector3d& ba_, const Eigen::Vector3d& bg_);
  Preinteration& operator=(const Preinteration& preinteration); // deep copy
  void Initialize();
  void SetNoiseAndWalk(double gyr_noise, double acc_noise, double gyr_walk, double acc_walk);
  void SetBias(const Eigen::Vector3d& gyr_bias, const Eigen::Vector3d& acc_bias, bool to_repropagate = true);
  void UpdateBias(const Eigen::Vector3d& gyr_bias, const Eigen::Vector3d& acc_bias);  
  void Propagate(double dt, const Eigen::Vector3d &acc_m, const Eigen::Vector3d &gyr_m, bool save_m = true);
  void Repropagate();
  void AddBatchData(const ImuDataList& batch_imu_data, double t0, double t1);

  const Eigen::Matrix3d GetDeltaRotation(const Eigen::Vector3d& gyr_bias);
  const Eigen::Vector3d GetDeltaPosition(const Eigen::Vector3d& gyr_bias, const Eigen::Vector3d& acc_bias);
  const Eigen::Vector3d GetDeltaVelocity(const Eigen::Vector3d& gyr_bias, const Eigen::Vector3d& acc_bias);
  const Eigen::Matrix3d GetUpdatedDeltaRotation();
  const Eigen::Vector3d GetUpdatedDeltaPosition();
  const Eigen::Vector3d GetUpdatedDeltaVelocity();
  const void GetUpdatedBias(Eigen::Vector3d& gyr_bias, Eigen::Vector3d& acc_bias);

  bool Valid();
  void Reset();
  void Predict(const Eigen::Matrix4d& Twb0, const Eigen::Vector3d& vwb0, Eigen::Matrix4d& Twb1, Eigen::Vector3d& vwb1);


public:
  double start_time, end_time;

  Eigen::DiagonalMatrix<double, 6> noise_matrix, walk_matrix;
  Eigen::Vector3d ba, bg;
  Eigen::Vector3d dba, dbg;
  double dT;
  Eigen::Matrix3d dR;
  Eigen::Vector3d dP, dV;
  Eigen::Matrix3d JRg, JVg, JVa, JPg, JPa;
  Matrix15d Cov;

  std::vector<double> dt_list;
  std::vector<Eigen::Vector3d> gyr_list;
  std::vector<Eigen::Vector3d> acc_list;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version){
    ar & start_time;
    ar & end_time;

    SerializeDiagonalMatrix6d(ar, noise_matrix, version);
    SerializeDiagonalMatrix6d(ar, walk_matrix, version);
    ar & boost::serialization::make_array(ba.data(), ba.size());
    ar & boost::serialization::make_array(bg.data(), bg.size());
    ar & boost::serialization::make_array(dba.data(), dba.size());
    ar & boost::serialization::make_array(dbg.data(), dbg.size());
    ar & dT;
    ar & boost::serialization::make_array(dR.data(), dR.size());
    ar & boost::serialization::make_array(dP.data(), dP.size());
    ar & boost::serialization::make_array(dV.data(), dV.size());
    ar & boost::serialization::make_array(JRg.data(), JRg.size());
    ar & boost::serialization::make_array(JVg.data(), JVg.size());
    ar & boost::serialization::make_array(JVa.data(), JVa.size());
    ar & boost::serialization::make_array(JPg.data(), JPg.size());
    ar & boost::serialization::make_array(JPa.data(), JPa.size());
    ar & boost::serialization::make_array(Cov.data(), Cov.size());

    ar & dt_list;
    SerializeEigenVector3dList(ar, gyr_list, version);
    SerializeEigenVector3dList(ar, acc_list, version);
  }
};

typedef std::shared_ptr<Preinteration> PreinterationPtr;

#endif  // IMU_H_
