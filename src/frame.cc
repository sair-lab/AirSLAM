#include "frame.h"
#include <assert.h>

#include "line_processor.h"

Frame::Frame(){
}

Frame::Frame(int frame_id, bool pose_fixed, CameraPtr camera, double timestamp):
    tracking_frame_id(-1), local_map_optimization_frame_id(-1), local_map_optimization_fix_frame_id(-1),
    _frame_id(frame_id), _pose_fixed(pose_fixed), _camera(camera), _timestamp(timestamp), _init_v(false){
  _grid_width_inv = static_cast<double>(FRAME_GRID_COLS)/static_cast<double>(_camera->ImageWidth());
  _grid_height_inv = static_cast<double>(FRAME_GRID_ROWS)/static_cast<double>(_camera->ImageHeight());
}

// Frame& Frame::operator=(const Frame& other){
//   tracking_frame_id = other.tracking_frame_id;
//   local_map_optimization_frame_id = other.local_map_optimization_frame_id;
//   local_map_optimization_fix_frame_id = other.local_map_optimization_fix_frame_id;
//   _frame_id = other._frame_id;
//   _timestamp = other._timestamp;
//   _pose_fixed = other._pose_fixed;
//   _pose = other._pose;

//   _features = other._features;
//   _keypoints = other._keypoints;
//   for(int i=0;i<FRAME_GRID_COLS;i++){
//     for(int j=0; j<FRAME_GRID_ROWS; j++){
//       _feature_grid[i][j] = other._feature_grid[i][j];
//     }
//   }
//   _grid_width_inv = other._grid_width_inv;
//   _grid_height_inv = other._grid_height_inv;
//   _u_right = other._u_right;
//   _depth = other._depth;
//   _track_ids = other._track_ids;
//   _mappoints = other._mappoints;
//   _camera = other._camera;

//   _imu_pose = other._imu_pose;
//   _preinteration = other._preinteration;
//   _previous_frame = other._previous_frame;
//   return *this;
// }

void Frame::SetFrameId(int frame_id){
  _frame_id = frame_id;
}

int Frame::GetFrameId(){
  return _frame_id;
}

double Frame::GetTimestamp(){
  return _timestamp;
}

void Frame::SetPoseFixed(bool pose_fixed){
  _pose_fixed = pose_fixed;
}

bool Frame::PoseFixed(){
  return _pose_fixed;
}

void Frame::SetPose(const Eigen::Matrix4d& pose){
  _pose = pose;
  Eigen::Matrix4d Tcb = _camera->BodyToCamera();
  _imu_pose = pose * Tcb;
}

Eigen::Matrix4d& Frame::GetPose(){
  return _pose;
}

bool Frame::FindGrid(float& x, float& y, int& grid_x, int& grid_y){
  grid_x = std::round(x * _grid_width_inv);
  grid_y = std::round(y * _grid_height_inv);

  grid_x = std::min(std::max(0, grid_x), (FRAME_GRID_COLS-1));
  grid_y = std::min(std::max(0, grid_y), (FRAME_GRID_ROWS-1));

  return !(grid_x < 0 || grid_x >= FRAME_GRID_COLS || grid_y < 0 || grid_y >= FRAME_GRID_ROWS);
}

void Frame::AddFeatures(Eigen::Matrix<float, 259, Eigen::Dynamic>& features_left, 
    Eigen::Matrix<float, 259, Eigen::Dynamic>& features_right, std::vector<Eigen::Vector4d>& lines_left, 
    std::vector<Eigen::Vector4d>& lines_right, std::vector<cv::DMatch>& stereo_matches){
  AddLeftFeatures(features_left, lines_left);
  AddRightFeatures(features_right, lines_right, stereo_matches);
}

void Frame::AddLeftFeatures(Eigen::Matrix<float, 259, Eigen::Dynamic>& features_left, 
    std::vector<Eigen::Vector4d>& lines_left){
  _features = features_left;

  // fill in keypoints and assign features to grids
  size_t features_left_size = _features.cols();
  for(size_t i = 0; i < features_left_size; ++i){
    float score = _features(0, i);
    float x = _features(1, i);
    float y = _features(2, i);
    _keypoints.emplace_back(x, y, 8, -1, score);

    int grid_x, grid_y;
    bool found = FindGrid(x, y, grid_x, grid_y);
    assert(found);
    _feature_grid[grid_x][grid_y].push_back(i);
  } 

  // initialize u_right and depth
  _u_right = std::vector<double>(features_left_size, -1);
  _depth = std::vector<double>(features_left_size, -1);

  // initialize track_ids and mappoints
  std::vector<int> track_ids(features_left_size, -1);
  SetTrackIds(track_ids);
  std::vector<MappointPtr> mappoints(features_left_size, nullptr);
  _mappoints = mappoints;

  // assign points to lines
  _lines = lines_left;
  std::vector<std::map<int, double>> points_on_line_left;
  std::vector<int> line_matches;
  AssignPointsToLines(lines_left, features_left, points_on_line_left);
  _points_on_lines = points_on_line_left;

  // initialize line track ids and maplines
  size_t line_num = lines_left.size();
  std::vector<int> line_track_ids(line_num, -1);
  _line_track_ids = line_track_ids;
  std::vector<MaplinePtr> maplines(line_num, nullptr);
  _maplines = maplines;

  // for debug
  relation_left = points_on_line_left;
}

int Frame::AddRightFeatures(Eigen::Matrix<float, 259, Eigen::Dynamic>& features_right, 
    std::vector<Eigen::Vector4d>& lines_right, std::vector<cv::DMatch>& stereo_matches){

  // filter matches from superglue
  std::vector<cv::DMatch> matches;
  double min_x_diff = _camera->MinXDiff();
  double max_x_diff = _camera->MaxXDiff();
  const double max_y_diff = _camera->MaxYDiff();
  for(cv::DMatch& match : stereo_matches){
    int idx_left = match.queryIdx;
    int idx_right = match.trainIdx;

    double dx = std::abs(_features(1, idx_left) - features_right(1, idx_right));
    double dy = std::abs(_features(2, idx_left) - features_right(2, idx_right));

    if(dx > min_x_diff && dx < max_x_diff && dy <= max_y_diff){
      matches.emplace_back(match);
    }
  }

  // Triangulate stereo points
  int good_stereo_point = 0;
  for(cv::DMatch& match : matches){
    int idx_left = match.queryIdx;
    int idx_right = match.trainIdx;

    assert(idx_left < _u_right.size());
    double parallax = _features(1, idx_left) - features_right(1, idx_right);

    if(parallax < _camera->MaxXDiff() && parallax > _camera->MinXDiff()){
      _u_right[idx_left] = features_right(1, idx_right);
      _depth[idx_left] = _camera->BF() / parallax;
      good_stereo_point++;
    }
  }

  // assign points to lines
  std::vector<std::map<int, double>> points_on_line_right;
  AssignPointsToLines(lines_right, features_right, points_on_line_right);

  // match stereo lines
  std::vector<int> line_matches;
  size_t line_num = _lines.size();
  _lines_right.resize(line_num);
  _lines_right_valid.resize(line_num);
  MatchLines(_points_on_lines, points_on_line_right, matches, _features.cols(), features_right.cols(), line_matches);
  for(size_t i = 0; i < line_num; i++){
    if(line_matches[i] > 0){
      _lines_right[i] = lines_right[line_matches[i]];
      _lines_right_valid[i] = true;
    }else{
      _lines_right_valid[i] = false;
    }
  }

  // for debug
  line_left_to_right_match = line_matches;
  relation_right = points_on_line_right;

  return good_stereo_point;
}

Eigen::Matrix<float, 259, Eigen::Dynamic>& Frame::GetAllFeatures(){
  return _features;
}

size_t Frame::FeatureNum(){
  return _features.cols();
}

bool Frame::GetKeypointPosition(size_t idx, Eigen::Vector3d& keypoint_pos){
  if(idx > _features.cols()) return false;
  keypoint_pos.head(2) = _features.block<2, 1>(1, idx).cast<double>();
  keypoint_pos(2) = _u_right[idx];
  return true;
}

std::vector<cv::KeyPoint>& Frame::GetAllKeypoints(){
  return _keypoints;
}

cv::KeyPoint& Frame::GetKeypoint(size_t idx){
  assert(idx < _keypoints.size());
  return _keypoints[idx];
}

int Frame::GetInlierFlag(std::vector<bool>& inliers_feature_message){
  int num_inliers = 0;
  inliers_feature_message.resize(_mappoints.size());
  for(size_t i = 0; i < _mappoints.size(); i++){
    if(_mappoints[i] && !_mappoints[i]->IsBad()){
      inliers_feature_message[i] = true;
      num_inliers++;
    }else{
      inliers_feature_message[i] = false;
    }
  }
  return num_inliers;
}

double Frame::GetRightPosition(size_t idx){
  assert(idx < _u_right.size());
  return _u_right[idx];
}

std::vector<double>& Frame::GetAllRightPosition(){
  return _u_right;
} 

bool Frame::GetDescriptor(size_t idx, Eigen::Matrix<float, 256, 1>& descriptor) const{
  if(idx > _features.cols()) return false;
  descriptor = _features.block<256, 1>(3, idx);
  return true;
}

double Frame::GetDepth(size_t idx){
  assert(idx < _depth.size());
  return _depth[idx];
}

std::vector<double>& Frame::GetAllDepth(){
  return _depth;
}

void Frame::SetDepth(size_t idx, double depth){
  assert(idx < _depth.size());
  _depth[idx] = depth;
}

void Frame::SetTrackIds(std::vector<int>& track_ids){
  _track_ids = track_ids;
}

std::vector<int>& Frame::GetAllTrackIds(){
  return _track_ids;
}

void Frame::SetTrackId(size_t idx, int track_id){
  _track_ids[idx] = track_id;
}

int Frame::GetTrackId(size_t idx){
  assert(idx < _track_ids.size());
  return _track_ids[idx];
}

MappointPtr Frame::GetMappoint(size_t idx){
  assert(idx < _mappoints.size());
  return _mappoints[idx];
}

std::vector<MappointPtr>& Frame::GetAllMappoints(){
  return _mappoints;
}

void Frame::InsertMappoint(size_t idx, MappointPtr mappoint){
  assert(idx < FeatureNum());
  _mappoints[idx] = mappoint;
}

bool Frame::BackProjectPoint(size_t idx, Eigen::Vector3d& p3D){
  if(idx >= _depth.size() || _depth[idx] <= 0) return false;
  Eigen::Vector3d p2D;
  if(!GetKeypointPosition(idx, p2D)) return false;
  if(!_camera->BackProjectStereo(p2D, p3D)) return false;
  return true;
}

CameraPtr Frame::GetCamera(){
  return _camera;
}

void Frame::FindNeighborKeypoints(Eigen::Vector3d& p2D, std::vector<int>& indices, double r, bool filter) const{
  double x = p2D(0);
  double y = p2D(1);
  double xr = p2D(2);
  const int min_grid_x = std::max(0, (int)std::floor((x-r)*_grid_width_inv));
  const int max_grid_x = std::min((int)(FRAME_GRID_COLS-1), (int)std::ceil((x+r)*_grid_width_inv));
  const int min_grid_y = std::max(0, (int)std::floor((y-r)*_grid_height_inv));
  const int max_grid_y = std::min((int)(FRAME_GRID_ROWS-1), (int)std::ceil((y+r)*_grid_height_inv));
  if(min_grid_x >= FRAME_GRID_COLS || max_grid_x < 0 || min_grid_y >= FRAME_GRID_ROWS || max_grid_y <0) return;

  for(int gx = min_grid_x; gx <= max_grid_x; gx++){
    for(int gy = min_grid_y; gy <= max_grid_y; gy++){
      if(_feature_grid[gx][gy].empty()) continue;
      for(auto& idx : _feature_grid[gx][gy]){
        if(filter && _mappoints[idx] && !_mappoints[idx]->IsBad()) continue;

        const double dx = _keypoints[idx].pt.x - x;
        const double dy = _keypoints[idx].pt.y - y;
        const double dxr = (xr > 0) ? (_u_right[idx] - xr) : 0;
        if(std::abs(dx) < r && std::abs(dy) < r && std::abs(dxr) < r){
          indices.push_back(idx);
        }
      }
    }
  }
}

size_t Frame::LineNum(){
  return _lines.size();
}

void Frame::SetLineTrackId(size_t idx, int line_track_id){
  if(idx < _lines.size()){
    _line_track_ids[idx] = line_track_id;
  }
}

int Frame::GetLineTrackId(size_t idx){
  if(idx < _lines.size()){
    return _line_track_ids[idx];
  }else{
    return -1;
  }
}

bool Frame::GetLine(size_t idx, Eigen::Vector4d& line){
  if(idx >= _lines.size()) return false;

  line = _lines[idx];
  return true;
}

bool Frame::GetLineRight(size_t idx, Eigen::Vector4d& line){
  if(idx >= _lines.size() || !_lines_right_valid[idx]) return false;
  
  line = _lines_right[idx];
  return true;
}

const std::vector<int>& Frame::GetAllLineTrackId(){
  return _line_track_ids;
}

const std::vector<Eigen::Vector4d>& Frame::GatAllLines(){
  return _lines;
}

const std::vector<Eigen::Vector4d>& Frame::GatAllRightLines(){
  return _lines_right;
}

bool Frame::GetRightLineStatus(size_t idx){
  if(idx >=0 && idx < _lines_right_valid.size()){
    return _lines_right_valid[idx];
  }
  return false;
}

const std::vector<bool>& Frame::GetAllRightLineStatus(){
  return _lines_right_valid;
}

void Frame::InsertMapline(size_t idx, MaplinePtr mapline){
  if(idx < _lines.size()){
    _maplines[idx] = mapline;
  }
}

std::vector<MaplinePtr>& Frame::GetAllMaplines(){
  return _maplines;
}

const std::vector<MaplinePtr>& Frame::GetConstAllMaplines(){
  return _maplines;
}

std::map<int, double> Frame::GetPointsOnLine(size_t idx){
  if(idx >= _points_on_lines.size()){
    std::map<int, double> null_map;
    return null_map;
  }
  return _points_on_lines[idx];
}

const std::vector<std::map<int, double>>& Frame::GetPointsOnLines(){
  return _points_on_lines;
}

bool Frame::TriangulateStereoLine(size_t idx, Vector6d& endpoints){
  if(idx >= _lines.size() || !_lines_right_valid[idx]) return false;
  return TriangulateByStereo(_lines[idx], _lines_right[idx], _pose, _camera, endpoints);
}

void Frame::RemoveMapline(MaplinePtr mapline){
  RemoveMapline(mapline->GetLineIdx(_frame_id));
}

void Frame::RemoveMapline(int idx){
  if(idx < _maplines.size() && idx >=0){
    _maplines[idx] = nullptr;
  }
}

void Frame::AddJunctions(Eigen::Matrix<float, 259, Eigen::Dynamic>& junctions){
  _junctions = junctions;
}

Eigen::Matrix<float, 259, Eigen::Dynamic>& Frame::GetJunctions(){
  return _junctions;
}

int Frame::JunctionNum(){
  return _junctions.cols();
}

void Frame::RemoveMappoint(MappointPtr mappoint){
  RemoveMappoint(mappoint->GetKeypointIdx(_frame_id));
}

void Frame::RemoveMappoint(int idx){
  if(idx < _mappoints.size() && idx >=0){
    _mappoints[idx] = nullptr;
  }
}

Eigen::Matrix4d Frame::IMUPose(){
  return _imu_pose;
}

void Frame::SetIMUPose(const Eigen::Matrix4d& pose){
  _imu_pose = pose;
  Eigen::Matrix4d Tbc = _camera->CameraToBody();
  _pose = _imu_pose * Tbc;
}

void Frame::SetIMUPreinteration(const Preinteration& preinteration){
  Preinteration frame_preinteration = preinteration;
  _preinteration = std::make_shared<Preinteration>(frame_preinteration);
}

PreinterationPtr Frame::GetIMUPreinteration(){
  return _preinteration;
}

bool Frame::VelocityIsInitialized(){
  return _init_v;
}

void Frame::SetVelocaity(const Eigen::Vector3d& velocity){
  _velocity = velocity;
  _init_v = true;
}

Eigen::Vector3d Frame::GetVelocity(){
  return _velocity;
}

void Frame::SetPreviousFrame(std::shared_ptr<Frame> previous_frame){
  _previous_frame = previous_frame;
}

std::shared_ptr<Frame> Frame::PreviousFrame(){
  return _previous_frame;
}

void Frame::SetBias(const Eigen::Vector3d& gyr_bias, const Eigen::Vector3d& acc_bias, bool to_repropagate){
  _preinteration->SetBias(gyr_bias, acc_bias, to_repropagate);
}

void Frame::UpdateBias(const Eigen::Vector3d& gyr_bias, const Eigen::Vector3d& acc_bias){
  _preinteration->UpdateBias(gyr_bias, acc_bias);
}

void Frame::GetBias(Eigen::Vector3d& gyr_bias, Eigen::Vector3d& acc_bias){
  _preinteration->GetUpdatedBias(gyr_bias, acc_bias);
}

void Frame::Repropagate(){
  _preinteration->Repropagate();
}

void Frame::DetectSentences(std::vector<DBoW2::WordId>& word_of_features){
  assert((int)word_of_features.size() == _features.cols());
  _sentences.clear();
  _sentences.resize(_points_on_lines.size());
  for(int i = 0; i < _points_on_lines.size(); i++){
    if(_points_on_lines[i].size() < 2) continue;

    for(auto& kv : _points_on_lines[i]){
      int kpt_idx = kv.first;
      DBoW2::WordId word_id = word_of_features[kpt_idx];
      if(word_id < UINT_MAX){
        _sentence_ids_of_word[word_id].push_back(i);
        _sentences[i].push_back(word_id);
      }
    }
  }
}


void Frame::FindSameSentences(const std::vector<std::vector<DBoW2::WordId>>& other_sentenses, 
    std::vector<int>& word_num_of_same_sentence){
  
  for(const std::vector<DBoW2::WordId>& other_sentense : other_sentenses){
    if(other_sentense.size() < 2) continue;

    std::map<int, int> sentense_id_to_word_num;
    std::map<DBoW2::WordId, std::vector<int>>::iterator it;
    for(DBoW2::WordId word_id : other_sentense){
      it = _sentence_ids_of_word.find(word_id);
      if(it == _sentence_ids_of_word.end()) continue;

      for(int sentence_id : it->second){
        if(sentense_id_to_word_num.count(sentence_id) > 0){
          sentense_id_to_word_num[sentence_id] += 1;
        }else{
          sentense_id_to_word_num[sentence_id] = 1;
        }
      }
    }

    int max_word_num = 0;
    for(auto& kv : sentense_id_to_word_num){
      max_word_num = std::max(max_word_num, kv.second);
    }

    if(max_word_num >= 2){
      word_num_of_same_sentence.push_back(max_word_num);
    }
  }
}

int Frame::ComputeSentenseSimilarity(const std::vector<DBoW2::WordId>& other_word_of_features){
  int word_num = 0;
  for(const DBoW2::WordId& word_id : other_word_of_features){
    if(word_id < UINT_MAX && _sentence_ids_of_word.find(word_id) != _sentence_ids_of_word.end()){
      word_num++;
    }
  }
  return word_num;
}

const std::map<DBoW2::WordId, std::vector<int>>& Frame::GetSentenseIdsOfWord(){
  return _sentence_ids_of_word;
}

const std::vector<std::vector<DBoW2::WordId>>& Frame::GetSentenses(){
  return _sentences;
}

void Frame::FindJunctionConnections(){
  _connected_junctions.resize(_junctions.cols());

  const int W = _camera->ImageWidth();
  const int H = _camera->ImageHeight();
  std::vector<std::vector<int>> junction_map(H, std::vector<int>(W, -1));
  for(int i = 0; i < _junctions.cols(); ++i){
    int x = (int)(_junctions(1, i)+0.5);
    int y = (int)(_junctions(2, i)+0.5);
    junction_map[y][x] = i;
  }

  const int WS = 2; // window size
  std::function<int(double, double)> match_junction = [&](double x, double y){
    int junction_id = -1;
    int d_min = 2 * WS + 1;

    int xi = int(x+0.5);
    int yi = int(y+0.5);
    for(int i = std::max(yi - WS, 0); i <= std::min(yi + WS, H-1); i++){
      for(int j = std::max(xi - WS, 0); j <= std::min(xi + WS, W-1); j++){
        if(junction_map[i][j] >= 0){
          int d = std::abs(yi - i) + std::abs(xi - j);
          if(d < d_min){
            junction_id = junction_map[i][j];
            d_min = d;

            if(d_min == 0){
              return junction_id;
            }
          }
        }
      }
    }

    return junction_id;
  };

  for(const Eigen::Vector4d& line : _lines){
    int junction_id1 = match_junction(line(0), line(1));
    if(junction_id1 < 0) continue;

    int junction_id2 = match_junction(line(2), line(3));
    if(junction_id2 < 0) continue;

    _connected_junctions[junction_id1].insert(junction_id2);
    _connected_junctions[junction_id2].insert(junction_id1);
  }
}

const std::vector<std::set<int>>& Frame::GetJunctionConnections(){
  return _connected_junctions;
} 