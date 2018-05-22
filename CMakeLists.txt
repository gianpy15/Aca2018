cmake_minimum_required(VERSION 3.10)
project(Aca2018)

add_executable(Aca2018 source/main.cpp)
add_executable(Intel source/from_intel_example.cpp)
add_executable(Tests source/tests/test_mkl_dnn.cpp)

set(CMAKE_CXX_STANDARD 11)

target_link_libraries(Intel mkldnn)
include_directories(/usr/local/include)

