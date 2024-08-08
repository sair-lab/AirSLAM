#ifndef DATABASE_H_
#define DATABASE_H_

#include <vector>
#include <numeric>
#include <fstream>
#include <string>
#include <list>
#include <set>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>

#include "3rdparty/DBoW2/include/DBoW2/TemplatedVocabulary.h"
#include "3rdparty/DBoW2/include/DBoW2/QueryResults.h"
#include "3rdparty/DBoW2/include/DBoW2/ScoringObject.h"
#include "3rdparty/DBoW2/include/DBoW2/BowVector.h"
#include "3rdparty/DBoW2/include/DBoW2/FeatureVector.h"
#include <boost/serialization/shared_ptr.hpp>

#include "include/bow/FSuperpoint.h"
#include "frame.h"

typedef DBoW2::TemplatedVocabulary<DBoW2::FSuperpoint::TDescriptor, DBoW2::FSuperpoint> SuperpointVocabulary;
typedef std::shared_ptr<SuperpointVocabulary> SuperpointVocabularyPtr;
typedef std::map<FramePtr, std::vector<int>> FrameFeatures;

namespace boost{
namespace serialization{

template<class Archive>
void serialize(Archive & ar, SuperpointVocabulary::Node & node, const unsigned int){
  ar & node.id;
  ar & node.weight;
  ar & node.children;
  ar & node.parent;
  ar & boost::serialization::make_array(node.descriptor.data(), node.descriptor.size());
  ar & node.word_id;
}


template<class Archive>
void serialize(Archive & ar, SuperpointVocabulary & voc, const unsigned int){
  ar & voc.m_k;
  ar & voc.m_L;
  ar & voc.m_weighting;
  ar & voc.m_scoring;
  ar & voc.m_nodes;
  ar & voc.m_words;
  voc.createScoringObject();
}

template<class Archive>
void serialize(Archive & ar, DBoW2::BowVector & v, const unsigned int){
  ar & boost::serialization::base_object<std::map<DBoW2::WordId, DBoW2::WordValue> >(v);
}

} // serialization
} // boost


class Database{
public:
  Database();
  Database(SuperpointVocabularyPtr voc);
  Database(const std::string voc_path);

  void LoadVocabulary(const std::string voc_path);
  void LoadVocabulary(SuperpointVocabularyPtr voc);

  void FrameToBow(FramePtr frame, DBoW2::WordIdToFeatures& word_features, DBoW2::BowVector& bow_vector);
  void FrameToBow(const Eigen::Matrix<float, 259, Eigen::Dynamic>& features_eigen, DBoW2::WordIdToFeatures& word_features, DBoW2::BowVector& bow_vector);
  void FrameToBow(FramePtr frame, DBoW2::WordIdToFeatures& word_features, DBoW2::BowVector& bow_vector, std::vector<DBoW2::WordId>& word_of_features);
  void FrameToBow(const Eigen::Matrix<float, 259, Eigen::Dynamic>& features_eigen, 
    DBoW2::WordIdToFeatures& word_features, DBoW2::BowVector& bow_vector, std::vector<DBoW2::WordId>& word_of_features);

  void AddFrame(FramePtr frame);
  void AddFrame(FramePtr frame, const DBoW2::WordIdToFeatures& word_features, const DBoW2::BowVector& bow_vector);
  void Query(const DBoW2::BowVector& bow_vector, std::map<FramePtr, int>& frame_sharing_words);
  double Score(const DBoW2::BowVector& bow_vector1, const DBoW2::BowVector& bow_vector2);

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version){
    ar & _inverted_file;
    ar & _frame_bow_vectors;
  }

// private:
  SuperpointVocabularyPtr _voc;
  std::vector<FrameFeatures> _inverted_file;
  std::map<FramePtr, DBoW2::BowVector> _frame_bow_vectors;
};

typedef std::shared_ptr<Database> DatabasePtr;

#endif // DATABASE_H_