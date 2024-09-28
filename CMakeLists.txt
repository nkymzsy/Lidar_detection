cmake_minimum_required(VERSION 3.10.1)
project(pillar_detect LANGUAGES CUDA CXX)

set(CMAKE_BUILD_TYPE "relase")
set(CMAKE_CXX_FLAGS "-std=c++17")


set(Torch_DIR /home/data/code/catkin_ws/lib/libtorch/share/cmake/Torch)  #解压的libtorch的绝对路径
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -expt-extended-lambda")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; -diag-suppress=20012,20014")


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



add_library(PillarsBuilderCuda SHARED /home/data/code/catkin_ws/src/pillar_detect/src/PillarsBuilderCuda.cu /home/data/code/catkin_ws/src/pillar_detect/src/DetectNet.cpp)
target_link_libraries(PillarsBuilderCuda)
target_include_directories(PillarsBuilderCuda PUBLIC ${EIGEN3_INCLUDE_DIR})

add_executable(demo demo.cpp)
target_link_libraries(demo ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${TORCH_LIBRARIES} PillarsBuilderCuda)

add_executable(train train.cpp)
target_link_libraries(train ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${TORCH_LIBRARIES} PillarsBuilderCuda)