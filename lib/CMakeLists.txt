cmake_minimum_required(VERSION 3.10.1)
project(pillar_detect LANGUAGES CUDA CXX)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -expt-extended-lambda")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}; -diag-suppress=20012,20014")

add_library(model SHARED  ${CMAKE_CURRENT_SOURCE_DIR}/src/PillarsBuilderCuda.cu  ${CMAKE_CURRENT_SOURCE_DIR}/src/DetectNet.cpp)
target_link_libraries(model)
target_include_directories(model PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include/ ${CMAKE_CURRENT_SOURCE_DIR}/include/model ${TORCH_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} /home/data/code/catkin_ws/src/pillar_detect/tools)