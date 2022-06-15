// #include "thread_publisher.h"

// template <typename T>
// ThreadPublisher<T>::ThreadPublisher(){ 
//   shutdown_requested = false;
// }

// template <typename T>
// ThreadPublisher<T>::~ThreadPublisher(){
// }

// template <typename T>
// void ThreadPublisher<T>::Register(
//     std::function<void(const std::shared_ptr<const T>&)> cb){
//   callbacks.push_back(cb);
// }

// template <typename T>
// void ThreadPublisher<T>::Start(){
//   publish_thread = std::thread(boost::bind(&ThreadPublisher::Process, this));
// }

// template <typename T>
// void ThreadPublisher<T>::Publish(const std::shared_ptr<const T> msg){
//   std::unique_lock<std::mutex> locker(msg_mutex);
//   msgs.push(msg);
//   locker.unlock();
//   msg_cond.notify_one();
// }

// template <typename T>
// void ThreadPublisher<T>::Process(){
//   while(!shutdown_requested){
//     std::shared_ptr<const T> msg;
//     std::unique_lock<std::mutex> locker(msg_mutex);
//     while(msgs.empty()){
//       if(shutdown_requested){
//         locker.unlock();
//         break;
//       }else{
//         msg_cond.wait(locker);
//       }
//     }
    
//     if(shutdown_requested){
//       break;
//     }else{
//       if(0){
//         msg = msgs.back();
//         while(!msgs.empty()){
//           msgs.pop();
//         }
//       }else{
//         msg = msgs.front();
//         msgs.pop();
//       }
//     }
//     locker.unlock();

//     for(auto callback : callbacks){
//       callback(msg);
//     }
//   }
// }

// template <typename T>
// void ThreadPublisher<T>::ShutDown(){
//   shutdown_requested = true;
//   msg_cond.notify_one();
//   if(publish_thread.joinable()){
//     publish_thread.join();
//   }
// }