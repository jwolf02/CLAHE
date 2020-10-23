cmake_minimum_required(VERSION 3.17)
project(CLAHE CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS} -O3 -Wall -mtune=native -march=native")

find_package(OpenCV REQUIRED)

set(2D_SOURCES          src/2D/main.cu
                        src/common.hpp
                        src/cuda_helpers.cuh
                        src/2D/clahe.cu
                        src/2D/clahe.cuh)

add_executable(CLAHE2D ${2D_SOURCES})
target_link_libraries(CLAHE2D PUBLIC ${OpenCV_LIBS})
target_include_directories(CLAHE2D PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/2D ${OpenCV_INCLUDE_DIRS})

set_target_properties(CLAHE2D PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(3D_SOURCES          src/3D/main.cpp
                        src/common.hpp
                        src/cuda_helpers.cuh
                        src/3D/clahe3d.cuh
                        src/3D/clahe3d.cu
                        src/3D/clahe3d.hpp
                        src/3D/clahe3d.cpp)

add_executable(CLAHE3D ${3D_SOURCES})
target_link_libraries(CLAHE3D PUBLIC ${OpenCV_LIBS})
target_include_directories(CLAHE3D PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/3D ${OpenCV_INCLUDE_DIRS})

set_target_properties(CLAHE3D PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

