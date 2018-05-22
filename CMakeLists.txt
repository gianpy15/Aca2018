cmake_minimum_required(VERSION 3.10)
project(Aca2018)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS -lmkldnn)
set(CMAKE_CXX_FLAGS -lmklml_intel)
set(CMAKE_CXX_FLAGS  -liomp5)


include_directories(/usr/local/lib)
include_directories(/usr/local/include)


add_executable(Aca2018 source/main.cpp)
add_executable(Tests source/tests/test_mkl_dnn.cpp)