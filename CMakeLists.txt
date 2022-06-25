cmake_minimum_required(VERSION 3.5)
project(air_slam)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_transport
  nav_msgs
  roscpp
  rospy
  std_msgs
  sensor_msgs
)

find_package(OpenCV 4.2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Boost REQUIRED)
find_package(G2O REQUIRED)
# find_package(gflags REQUIRED)
find_package(Gflags REQUIRED)
find_package(Glog REQUIRED)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES ${PROJECT_NAME}_lib
 CATKIN_DEPENDS geometry_msgs image_transport nav_msgs roscpp rospy std_msgs
)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIR}
  ${GFLAGS_INCLUDE_DIRS} 
  ${GLOG_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}_lib SHARED
  src/g2o_optimization/g2o_optimization.cc
  src/super_point.cpp
  src/super_glue.cpp
  src/utils.cc
  src/camera.cc
  src/dataset.cc
  src/frame.cc
  src/point_matching.cc
  src/mappoint.cc
  # src/thread_publisher.cc
  src/ros_publisher.cc
  src/map.cc
  src/map_builder.cc
  src/timer.cc
)

target_link_libraries(${PROJECT_NAME}_lib
  nvinfer
  nvonnxparser
  ${PROJECT_SOURCE_DIR}/Thirdparty/TensorRTBuffer/lib/libTensorRTBuffer.so
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${Boost_LIBRARIES}
  ${G2O_LIBRARIES}
  ${GFLAGS_LIBRARIES} 
  ${GLOG_LIBRARIES}
  yaml-cpp
)

message(${G2O_LIBRARIES})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_lib ${catkin_LIBRARIES})

