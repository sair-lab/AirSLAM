#ifndef MAPLINE_H_
#define MAPLINE_H_

#include <limits>
#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/slam3d_addons/types_slam3d_addons.h>

#include "utils.h"


class Mapline{
public:
  enum Type {
    UnTriangulated = 0,
    Good = 1,
    Bad = 2,
  };

  Mapline();
  Mapline(int& mappoint_id);
  void SetId(int id);
  int GetId();
  void SetType(Type& type);
  Type GetType();
  void SetBad();
  bool IsBad();
  void SetGood();
  bool IsValid();

  void SetEndpoints(Vector6d& p);
  Vector6d& GetEndpoints();
  void SetEndpointsUpdateStatus(bool status);
  bool ToEndpointsUpdate();
  void SetLine3D(Line3D& line_3d);
  ConstLine3DPtr GetLine3DPtr();
  Line3D GetLine3D();

  void AddObverser(const int& frame_id, const int& line_index);
  void RemoveObverser(const int& frame_id);
  int ObverserNum();
  std::map<int, int>& GetAllObversers();
  int GetLineIdx(int frame_id);

public:

private:
  int _id;
  Type _type;
  bool _to_update_endpoints;
  Vector6d _endpoints;
  Line3DPtr _line_3d;
  std::map<int, int> _obversers;  // frame_id - line_index 
};

typedef std::shared_ptr<Mapline> MaplinePtr;

#endif  // MAPLINE_H_