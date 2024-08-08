#include <iostream> 

#include "mappoint.h"

Mappoint::Mappoint(): tracking_frame_id(-1), last_frame_seen(-1), local_map_optimization_frame_id(-1),
     _type(Type::UnTriangulated){
}

Mappoint::Mappoint(int& mappoint_id): tracking_frame_id(-1), last_frame_seen(-1),
    local_map_optimization_frame_id(-1), _id(mappoint_id), _type(Type::UnTriangulated){
  if(mappoint_id < 0) exit(0);
}

Mappoint::Mappoint(int& mappoint_id, Eigen::Vector3d& p): tracking_frame_id(-1), last_frame_seen(-1), 
    local_map_optimization_frame_id(-1), _id(mappoint_id), _type(Type::Good), _position(p){
}

Mappoint::Mappoint(int& mappoint_id, Eigen::Vector3d& p, Eigen::Matrix<float, 256, 1>& d):
    tracking_frame_id(-1), last_frame_seen(-1), local_map_optimization_frame_id(-1), 
    _id(mappoint_id), _type(Type::Good), _position(p), _descriptor(d){

}

void Mappoint::SetId(int id){
  _id = id;
}

int Mappoint::GetId(){
  return _id;
}

void Mappoint::SetType(const Type& type){
  _type = type;
}

Mappoint::Type Mappoint::GetType(){
  return _type;
}

void Mappoint::SetBad(){
  _type = Type::Bad;
  _obversers.clear();
}

bool Mappoint::IsBad(){
  return (_type == Type::Bad);
}

void Mappoint::SetGood(){
  _type = Type::Good;
}

bool Mappoint::IsValid(){
  return (_type == Type::Good);
}

void Mappoint::AddObverser(const int& frame_id, const int& keypoint_index){
  _obversers[frame_id] = keypoint_index;
}

void Mappoint::RemoveObverser(const int& frame_id){
  std::map<int, int>::iterator it = _obversers.find(frame_id);
  if(it != _obversers.end()){
    _obversers.erase(it);
  }
}

int Mappoint::ObverserNum(){
  int obverser_num = 0;
  for(auto& kv : _obversers){
    if(kv.second >= 0){
      obverser_num++;
    }
  }
  return obverser_num;
}

void Mappoint::SetPosition(const Eigen::Vector3d& p){
  _position = p;
  if(_type == Type::UnTriangulated){
    _type = Type::Good;
  }
}

Eigen::Vector3d Mappoint::GetPosition(){
  return _position;
}

void Mappoint::SetDescriptor(const Eigen::Matrix<float, 256, 1>& descriptor){
  _descriptor = descriptor;
}

Eigen::Matrix<float, 256, 1>& Mappoint::GetDescriptor(){
  return _descriptor;
}

std::map<int, int>& Mappoint::GetAllObversers(){
  return _obversers;
}

int Mappoint::GetKeypointIdx(int frame_id){
  if(_obversers.count(frame_id) > 0) return _obversers[frame_id];
  return -1;
}