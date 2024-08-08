#include "bow/database.h"
#include <climits>

Database::Database(){
}

Database::Database(SuperpointVocabularyPtr voc): _voc(voc){
  _inverted_file.resize(_voc->size());
}

Database::Database(const std::string voc_path){
  LoadVocabulary(voc_path);
}

void Database::LoadVocabulary(const std::string voc_path){
  SuperpointVocabulary voc_load;
  std::ifstream ifs(voc_path, std::ios::binary);
  boost::archive::binary_iarchive ia(ifs);
  ia >> voc_load;
  _voc = std::make_shared<SuperpointVocabulary>(voc_load);
  
  if(_inverted_file.empty()){
    _inverted_file.resize(_voc->size());
  }
}

void Database::LoadVocabulary(SuperpointVocabularyPtr voc){
  _voc = voc;
  if(_inverted_file.empty()){
    _inverted_file.resize(_voc->size());
  }
}

void Database::FrameToBow(FramePtr frame, DBoW2::WordIdToFeatures& word_features, DBoW2::BowVector& bow_vector){
  const Eigen::Matrix<float, 259, Eigen::Dynamic>& features_eigen = frame->GetAllFeatures();
  FrameToBow(features_eigen, word_features, bow_vector);
}

void Database::FrameToBow(const Eigen::Matrix<float, 259, Eigen::Dynamic>& features_eigen, 
    DBoW2::WordIdToFeatures& word_features, DBoW2::BowVector& bow_vector){
  int N = features_eigen.cols();
  std::vector<Eigen::Matrix<float, 256, 1>> features;
  features.reserve(N);
  for(int i = 0; i < N; i++){
    features.emplace_back(features_eigen.block(3, i, 256, 1));
  }
  _voc->transform(features, bow_vector, word_features);
}

void Database::FrameToBow(FramePtr frame, DBoW2::WordIdToFeatures& word_features, DBoW2::BowVector& bow_vector, 
    std::vector<DBoW2::WordId>& word_of_features){
  const Eigen::Matrix<float, 259, Eigen::Dynamic>& features_eigen = frame->GetAllFeatures();
  FrameToBow(features_eigen, word_features, bow_vector, word_of_features);
}


void Database::FrameToBow(const Eigen::Matrix<float, 259, Eigen::Dynamic>& features_eigen, 
    DBoW2::WordIdToFeatures& word_features, DBoW2::BowVector& bow_vector, std::vector<DBoW2::WordId>& word_of_features){
  int N = features_eigen.cols();
  if(N == 0) return; 

  // normalize 
  DBoW2::LNorm norm;
  bool must = _voc->m_scoring_object->mustNormalize(norm);
  for(int i = 0; i < N; i++){
    DBoW2::WordId id;
    DBoW2::WordValue w; // w is the idf value if TF_IDF, 1 if TF

    _voc->transform(features_eigen.block(3, i, 256, 1), id, w);
    if(w > 0){
      bow_vector.addWeight(id, w);
      word_features[id].emplace_back(i);
      word_of_features.push_back(id);
    }else{
      word_of_features.push_back(UINT_MAX);
    }
  }

  if(bow_vector.empty()) return;

  if(must){
    bow_vector.normalize(norm);
  }else{
    const double nd = bow_vector.size();
    for(DBoW2::BowVector::iterator vit = bow_vector.begin(); vit != bow_vector.end(); vit++){
      vit->second /= nd;
    }
  }
}

void Database::AddFrame(FramePtr frame){
  DBoW2::WordIdToFeatures word_features;
  DBoW2::BowVector bow_vector;
  FrameToBow(frame, word_features, bow_vector);
  AddFrame(frame, word_features, bow_vector);
}

void Database::AddFrame(FramePtr frame, const DBoW2::WordIdToFeatures& word_features, const DBoW2::BowVector& bow_vector){
  _frame_bow_vectors[frame] = bow_vector;

  // update inverted file
  for(auto& kv : word_features){
    const DBoW2::WordId& word_id = kv.first;
    _inverted_file[word_id][frame] = kv.second;
  }
}

void Database::Query(const DBoW2::BowVector& bow_vector, std::map<FramePtr, int>& frame_sharing_words){
  DBoW2::BowVector::const_iterator vit;
  for(vit = bow_vector.begin(); vit != bow_vector.end(); ++vit){
    const DBoW2::WordId word_id = vit->first;
    for(const auto& kv : _inverted_file[word_id]){
      FramePtr f = kv.first;
      if(frame_sharing_words.find(f) == frame_sharing_words.end()){
        frame_sharing_words[f] = 0;
      }
      frame_sharing_words[f]++;
    }
  }
}

double Database::Score(const DBoW2::BowVector& bow_vector1, const DBoW2::BowVector& bow_vector2){
  return _voc->score(bow_vector1, bow_vector2);
}