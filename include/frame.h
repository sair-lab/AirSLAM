#ifndef FRAME_H_
#define FRAME_H_

#include <string>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <opencv2/opencv.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "utils.h"
#include "mappoint.h"
#include "mapline.h"
#include "camera.h"
#include "imu.h"
#include "3rdparty/DBoW2/include/DBoW2/TemplatedVocabulary.h"

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class Frame{
public:
  Frame();
  Frame(int frame_id, bool pose_fixed, CameraPtr camera, double timestamp);
  // Frame& operator=(const Frame& other);

  void SetFrameId(int frame_id);
  int GetFrameId();
  double GetTimestamp();
  void SetPoseFixed(bool pose_fixed);
  bool PoseFixed();
  void SetPose(const Eigen::Matrix4d& pose);
  Eigen::Matrix4d& GetPose();

  // point features
  bool FindGrid(float& x, float& y, int& grid_x, int& grid_y);
  void AddFeatures(Eigen::Matrix<float, 259, Eigen::Dynamic>& features_left, 
      Eigen::Matrix<float, 259, Eigen::Dynamic>& features_right, std::vector<Eigen::Vector4d>& lines_left, 
      std::vector<Eigen::Vector4d>& lines_right, std::vector<cv::DMatch>& stereo_matches);
  void AddLeftFeatures(Eigen::Matrix<float, 259, Eigen::Dynamic>& features_left, std::vector<Eigen::Vector4d>& lines_left);
  int AddRightFeatures(Eigen::Matrix<float, 259, Eigen::Dynamic>& features_right, std::vector<Eigen::Vector4d>& lines_right, std::vector<cv::DMatch>& stereo_matches);

  Eigen::Matrix<float, 259, Eigen::Dynamic>& GetAllFeatures();

  size_t FeatureNum();

  bool GetKeypointPosition(size_t idx, Eigen::Vector3d& keypoint_pos);
  std::vector<cv::KeyPoint>& GetAllKeypoints();
  cv::KeyPoint& GetKeypoint(size_t idx);
  int GetInlierFlag(std::vector<bool>& inliers_feature_message);

  double GetRightPosition(size_t idx);
  std::vector<double>& GetAllRightPosition(); 

  bool GetDescriptor(size_t idx, Eigen::Matrix<float, 256, 1>& descriptor) const;

  double GetDepth(size_t idx);
  std::vector<double>& GetAllDepth();
  void SetDepth(size_t idx, double depth);

  void SetTrackIds(std::vector<int>& track_ids);
  std::vector<int>& GetAllTrackIds();
  void SetTrackId(size_t idx, int track_id);
  int GetTrackId(size_t idx);

  MappointPtr GetMappoint(size_t idx);
  std::vector<MappointPtr>& GetAllMappoints();
  void InsertMappoint(size_t idx, MappointPtr mappoint);

  bool BackProjectPoint(size_t idx, Eigen::Vector3d& p3D);
  CameraPtr GetCamera();
  void FindNeighborKeypoints(Eigen::Vector3d& p2D, std::vector<int>& indices, double r, bool filter = true) const;

  
  // line features
  size_t LineNum();
  void SetLineTrackId(size_t idx, int line_track_id);
  int GetLineTrackId(size_t idx);
  const std::vector<int>& GetAllLineTrackId();
  bool GetLine(size_t idx, Eigen::Vector4d& line);
  bool GetLineRight(size_t idx, Eigen::Vector4d& line);
  const std::vector<Eigen::Vector4d>& GatAllLines();
  const std::vector<Eigen::Vector4d>& GatAllRightLines();
  bool GetRightLineStatus(size_t idx);
  const std::vector<bool>& GetAllRightLineStatus();
  void InsertMapline(size_t idx, MaplinePtr mapline);
  std::vector<MaplinePtr>& GetAllMaplines();
  const std::vector<MaplinePtr>& GetConstAllMaplines();
  std::map<int, double> GetPointsOnLine(size_t idx);
  const std::vector<std::map<int, double>>& GetPointsOnLines();
  bool TriangulateStereoLine(size_t idx, Vector6d& endpoints);
  void RemoveMapline(MaplinePtr mapline);
  void RemoveMapline(int idx);

  void AddJunctions(Eigen::Matrix<float, 259, Eigen::Dynamic>& junctions);
  Eigen::Matrix<float, 259, Eigen::Dynamic>& GetJunctions();
  int JunctionNum();

  void RemoveMappoint(MappointPtr mappoint);
  void RemoveMappoint(int idx);

  // for IMU
  Eigen::Matrix4d IMUPose();
  void SetIMUPose(const Eigen::Matrix4d& pose);
  void SetIMUPreinteration(const Preinteration& preinteration);
  PreinterationPtr GetIMUPreinteration();
  bool VelocityIsInitialized();
  void SetVelocaity(const Eigen::Vector3d& velocity);
  Eigen::Vector3d GetVelocity();
  void SetPreviousFrame(const std::shared_ptr<Frame> previous_frame);
  std::shared_ptr<Frame> PreviousFrame();
  void SetBias(const Eigen::Vector3d& gyr_bias, const Eigen::Vector3d& acc_bias, bool to_repropagate = true);
  void UpdateBias(const Eigen::Vector3d& gyr_bias, const Eigen::Vector3d& acc_bias);
  void GetBias(Eigen::Vector3d& gyr_bias, Eigen::Vector3d& acc_bias);
  void Repropagate();

  // for loop detection and re-localization
  void DetectSentences(std::vector<DBoW2::WordId>& word_of_features);
  void FindSameSentences(const std::vector<std::vector<DBoW2::WordId>>& other_sentenses, 
      std::vector<int>& word_num_of_same_sentence);
  int ComputeSentenseSimilarity(const std::vector<DBoW2::WordId>& other_word_of_features);

  const std::map<DBoW2::WordId, std::vector<int>>& GetSentenseIdsOfWord();
  const std::vector<std::vector<DBoW2::WordId>>& GetSentenses();

  void FindJunctionConnections();
  const std::vector<std::set<int>>& GetJunctionConnections();

  
public:
  int tracking_frame_id;
  int local_map_optimization_frame_id;
  int local_map_optimization_fix_frame_id;

  // debug
  std::vector<int> line_left_to_right_match;
  std::vector<std::map<int, double>> relation_left;
  std::vector<std::map<int, double>> relation_right;

private:
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version){
    ar & _frame_id;
    ar & _timestamp;
    ar & _pose_fixed;
    ar & boost::serialization::make_array(_pose.data(), _pose.size());

    SerializeFeatures(ar, _features, version);
    SerializeKeypoints(ar, _keypoints, version);
    ar & _feature_grid;
    ar & _grid_width_inv;
    ar & _grid_height_inv;
    ar & _u_right;
    ar & _depth;
    ar & _track_ids;
    ar & _mappoints;

    SerializeEigenVector4dList(ar, _lines, version);
    SerializeEigenVector4dList(ar, _lines_right, version);
    ar & _lines_right_valid;
    ar & _points_on_lines;
    ar & _line_track_ids;
    ar & _maplines;

    SerializeFeatures(ar, _junctions, version);
    ar & _connected_junctions;

    ar & _camera;

    ar & boost::serialization::make_array(_imu_pose.data(), _imu_pose.size());
    ar & _init_v;
    ar & boost::serialization::make_array(_velocity.data(), _velocity.size());
    ar & _preinteration;

    ar & _sentence_ids_of_word;
    ar & _sentences;
  }

private:
  int _frame_id;
  double _timestamp;
  bool _pose_fixed;
  Eigen::Matrix4d _pose;

  // point features
  Eigen::Matrix<float, 259, Eigen::Dynamic> _features;
  std::vector<cv::KeyPoint> _keypoints;
  std::vector<int> _feature_grid[FRAME_GRID_COLS][FRAME_GRID_ROWS];
  double _grid_width_inv;
  double _grid_height_inv;
  std::vector<double> _u_right;
  std::vector<double> _depth;
  std::vector<int> _track_ids;
  std::vector<MappointPtr> _mappoints;

  // line features
  std::vector<Eigen::Vector4d> _lines;
  std::vector<Eigen::Vector4d> _lines_right;
  std::vector<bool> _lines_right_valid;
  std::vector<std::map<int, double>> _points_on_lines;
  std::vector<int> _line_track_ids;
  std::vector<MaplinePtr> _maplines;

  // junctions
  Eigen::Matrix<float, 259, Eigen::Dynamic> _junctions;
  std::vector<std::set<int>> _connected_junctions;

  // camera
  CameraPtr _camera;

  // for imu
  Eigen::Matrix4d _imu_pose;
  bool _init_v;   
  Eigen::Vector3d _velocity;
  PreinterationPtr _preinteration;
  std::shared_ptr<Frame> _previous_frame;

  // for re-localization, word id <-> sentecnse indeces
  std::map<DBoW2::WordId, std::vector<int>> _sentence_ids_of_word;
  std::vector<std::vector<DBoW2::WordId>> _sentences;
};

typedef std::shared_ptr<Frame> FramePtr;

#endif  // FRAME_H_