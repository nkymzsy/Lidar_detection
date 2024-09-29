cmake_minimum_required(VERSION 3.10.1)
project(pillar_detect LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "release")
set(CMAKE_CXX_FLAGS "-std=c++17")

#解压的libtorch的绝对路径
set(Torch_DIR /home/data/code/catkin_ws/lib/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(EIGEN3_INCLUDE_DIR "/home/data/code/catkin_ws/lib/eigen-3.4.0")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  pcl_conversions
  pcl_ros
)

message(STATUS "Found Eigen version: ${EIGEN3_VERSION}")

find_package(PCL REQUIRED QUIET)

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
 ${PCL_INCLUDE_DIRS}
 ${CMAKE_CURRENT_SOURCE_DIR}
)

catkin_package(
  DEPENDS PCL
  CATKIN_DEPENDS
  std_msgs
  sensor_msgs
)

add_subdirectory(lib)

add_executable(demo demo/demo.cpp)
target_link_libraries(demo ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${TORCH_LIBRARIES} model)

add_executable(train demo/train.cpp)
target_link_libraries(train ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${TORCH_LIBRARIES} model)