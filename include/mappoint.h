#ifndef MAPPOINT_H_
#define MAPPOINT_H_

#include <limits>
#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

static constexpr int kMaxInt = std::numeric_limits<int>::max();

struct Obverser {
  int frame_id;
  int keypoint_index;

  inline Obverser() : frame_id(kMaxInt), keypoint_index(kMaxInt) {}
  inline Obverser(int& _frame_id, int _keypoint_index)
      : frame_id(_frame_id), keypoint_index(_keypoint_index) {}

  inline bool operator==(const Obverser& other) const {
    return frame_id == other.frame_id && keypoint_index == other.keypoint_index;
  }
  inline bool operator!=(const Obverser& other) const {
    return frame_id != other.frame_id || keypoint_index != other.keypoint_index;
  }
  inline bool operator<(const Obverser& other) const {
    if (frame_id == other.frame_id) {
      return keypoint_index < other.keypoint_index;
    } else {
      return frame_id < other.frame_id;
    }
  }
  inline bool isValid() const {
    return frame_id != kMaxInt && keypoint_index != kMaxInt;
  }
};

class Mappoint{
public:
  enum Type {
    UnTriangulated = 0,
    Good = 1,
    Bad = 2,
  };

  Mappoint();
  Mappoint(int& mappoint_id);
  Mappoint(int& mappoint_id, Eigen::Vector3d& p);
  Mappoint(int& mappoint_id, Eigen::Vector3d& p, Eigen::Matrix<double, 256, 1>& d);
  void SetId(int id);
  int GetId();
  void SetType(Type& type);
  Type GetType();
  void SetBad();
  bool IsBad();
  void SetGood();
  bool IsValid();

  void SetPosition(Eigen::Vector3d& p);
  Eigen::Vector3d& GetPosition();
  void SetDescriptor(const Eigen::Matrix<double, 256, 1>& descriptor);
  Eigen::Matrix<double, 256, 1>& GetDescriptor(); 

  void AddObverser(const int& frame_id, const int& keypoint_index);
  void RemoveObverser(const int& frame_id);
  int ObverserNum();
  std::map<int, int>& GetAllObversers();
  int GetKeypointIdx(int frame_id);

private:
  int _id;
  Type _type;
  Eigen::Vector3d _position;
  Eigen::Matrix<double, 256, 1> _descriptor;
  std::map<int, int> _obversers;  // frame_id - keypoint_index 
};

typedef std::shared_ptr<Mappoint> MappointPtr;

#endif  // MAPPOINT_H