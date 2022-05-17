#include "mappoint.h"

Mappoint::Mappoint(): _type(Type::UnTriangulated){
}

Mappoint::Mappoint(int& mappoint_id): 
    _id(mappoint_id), _type(Type::UnTriangulated){
}

Mappoint::Mappoint(int& mappoint_id, Eigen::Vector3d& p):
    _id(mappoint_id), _type(Type::Good), _position(p){
}

void Mappoint::SetId(int id){
  _id = id;
}

int Mappoint::GetId(){
  return _id;
}

void Mappoint::SetType(Type& type){
  _type = type;
}

Mappoint::Type Mappoint::GetType(){
  return _type;
}

void Mappoint::SetBad(){
  _type = Type::Bad;
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
  if(_obversers.count(frame_id) > 0) 
    _obversers[frame_id] = -1;
}

void Mappoint::SetPosition(Eigen::Vector3d& p){
  _position = p;
  if(_type == Type::UnTriangulated){
    _type = Type::Good;
  }
}

Eigen::Vector3d& Mappoint::GetPosition(){
  return _position;
}

std::map<int, int>& Mappoint::GetAllObversers(){
  return _obversers;
}

int Mappoint::GetKeypointIdx(int frame_id){
  if(_obversers.count(frame_id) > 0) return _obversers[frame_id];
  return -1;
}