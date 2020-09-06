cmake_minimum_required(VERSION 3.17)
project(CLAHE CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS} -O3 -Wall -mtune=native -march=native")

option(CUDA_SHARED_MEMORY "host and device shared a unified main memory" OFF)

find_package(OpenCV REQUIRED)

set(SOURCES             src/main.cu)

add_executable(CLAHE ${SOURCES})
target_link_libraries(CLAHE PUBLIC ${OpenCV_LIBS})
target_include_directories(CLAHE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${OpenCV_INCLUDE_DIRS})
if (CUDA_SHARED_MEMORY)
    message("-- Building for shared memory platform")
    add_definitions(-DCUDA_SHARED_MEMORY)
endif()

set_target_properties(
        CLAHE
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)