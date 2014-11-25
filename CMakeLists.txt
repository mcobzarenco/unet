project(unet)
cmake_minimum_required (VERSION 2.8)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -pedantic -Wall -Wno-deprecated-register -g -O3")

# Package Dependencies
# ==============================================================================
find_package(
  Boost 1.53
    COMPONENTS
      program_options
    REQUIRED
)

find_package(Glog REQUIRED)

include_directories(
  ${GLOG_INCLUDE_DIR}
  /usr/include/eigen3
  src
)

if(MKL_ROOT)
  message("-- MKL_ROOT is set (${MKL_ROOT}), compiling with MKL")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -openmp -DMKL_LP64 -DEIGEN_USE_MKL_ALL -mkl -static-intel")
    include_directories(${MKL_ROOT}/include)
endif()

add_subdirectory(src)
add_subdirectory(test)
