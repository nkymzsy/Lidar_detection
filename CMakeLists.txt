cmake_minimum_required(VERSION 3.0.2)
project(pillar_detect)

set(CMAKE_BUILD_TYPE "relase")
set(CMAKE_CXX_FLAGS "-std=c++17")


set(Torch_DIR /home/data/code/catkin_ws/lib/libtorch/share/cmake/Torch)  #解压的libtorch的绝对路径
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  pcl_conversions
  pcl_ros
)

find_package(PCL REQUIRED QUIET)


include_directories(
 include
 ${catkin_INCLUDE_DIRS}
 ${PCL_INCLUDE_DIRS}
 ${TORCH_INCLUDE_DIRS}
)


catkin_package(
  INCLUDE_DIRS include
  DEPENDS PCL
  CATKIN_DEPENDS
  std_msgs
  sensor_msgs
)

add_executable(demo demo.cpp /home/data/code/catkin_ws/src/pillar_detect/src/DetectNet.cpp)
target_link_libraries(demo ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${TORCH_LIBRARIES})

add_executable(train train.cpp /home/data/code/catkin_ws/src/pillar_detect/src/DetectNet.cpp)
target_link_libraries(train ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${TORCH_LIBRARIES})