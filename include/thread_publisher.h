#ifndef THREAD_PUBLISHER_H_
#define THREAD_PUBLISHER_H_

#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <queue>
#include <boost/bind.hpp>
#include <functional>

template <class T>
class ThreadPublisher{
public:
  ThreadPublisher(){ 
    shutdown_requested = false;
  }

  ~ThreadPublisher() {}

  void Register(std::function<void(const std::shared_ptr<const T>&)> cb){
    callbacks.push_back(cb);
  }

  void Start(){
    publish_thread = std::thread(boost::bind(&ThreadPublisher::Process, this));
  }

  void Publish(const std::shared_ptr<const T> msg){
    std::unique_lock<std::mutex> locker(msg_mutex);
    msgs.push(msg);
    locker.unlock();
    msg_cond.notify_one();
  }

  void Process(){
    while(!shutdown_requested){
      std::shared_ptr<const T> msg;
      std::unique_lock<std::mutex> locker(msg_mutex);
      while(msgs.empty()){
        if(shutdown_requested){
          locker.unlock();
          break;
        }else{
          msg_cond.wait(locker);
        }
      }
      
      if(shutdown_requested){
        break;
      }else{
        if(0){
          msg = msgs.back();
          while(!msgs.empty()){
            msgs.pop();
          }
        }else{
          msg = msgs.front();
          msgs.pop();
        }
      }
      locker.unlock();

      for(auto callback : callbacks){
        callback(msg);
      }
    }
  }

  void ShutDown(){
    shutdown_requested = true;
    msg_cond.notify_one();
    if(publish_thread.joinable()){
      publish_thread.join();
    }
  }

private:
  std::mutex msg_mutex;
  std::condition_variable msg_cond;
  std::queue<std::shared_ptr<const T> > msgs;

  std::thread publish_thread;
  std::vector<std::function<void(const std::shared_ptr<const T>&)>> callbacks;

  bool shutdown_requested;
};

#endif  // THREAD_PUBLISHER_H_