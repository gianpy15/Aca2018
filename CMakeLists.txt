cmake_minimum_required(VERSION 3.10)
project(Aca2018)

set(CMAKE_CXX_STANDARD 11)

add_executable(Aca2018 source/main.cpp source/neural_network/AbsNet.cpp source/neural_network/FPNetwork.cpp source/neural_network/INTNetwork.cpp source/neural_network/INTNetwork.h)
add_executable(Intel source/from_intel_example.cpp)
add_executable(IntelQuant source/intel_int8_example.cpp)
add_executable(Tests source/tests/test_mkl_dnn.cpp)

target_link_libraries(IntelQuant mkldnn)
target_link_libraries(Intel mkldnn)
target_link_libraries(Aca2018 mkldnn)
include_directories(/usr/local/include)
