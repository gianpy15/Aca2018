cmake_minimum_required(VERSION 3.10)
project(Aca2018)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native")

# Download and unpack googletest at configure time
configure_file(tests/CMakeLists.txt googletest-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download" )
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download" )

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This adds
# the following targets: gtest, gtest_main, gmock
# and gmock_main
add_subdirectory("${CMAKE_BINARY_DIR}/googletest-src"
        "${CMAKE_BINARY_DIR}/googletest-build")

# The gtest/gmock targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if(CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include"
            "${gmock_SOURCE_DIR}/include")
endif()


link_directories(resources/hdf5/HDF_Group/HDF5/1.10.1/lib)

add_executable(Aca2018 source/main.cpp source/neural_network/AbsNet.cpp source/neural_network/FPNetwork.cpp
        source/neural_network/INTNetwork.cpp source/neural_network/INTNetwork.h source/monitoring/mem_monitoring.cpp
        source/monitoring/mem_monitoring.h source/mem_management/mem_base.h source/mem_management/mem_base.cpp
        source/mem_management/mem_management.h source/mem_management/mem_management.cpp source/io/h5io.h
        source/io/h5io.cpp source/logging/logging.cpp source/logging/logging.h source/io/DataSetIO.h
        source/io/DataSetIO.cpp source/measurements.cpp source/measurements.h source/monitoring/cpu_monitoring.cpp source/monitoring/cpu_monitoring.h)
add_executable(Intel source/from_intel_example.cpp)
add_executable(IntelQuant source/intel_int8_example.cpp)
add_executable(Tests tests/FPNetTest.cpp source/neural_network/AbsNet.cpp source/neural_network/FPNetwork.cpp
        source/neural_network/INTNetwork.cpp source/neural_network/INTNetwork.h source/monitoring/mem_monitoring.cpp
        source/monitoring/mem_monitoring.h source/mem_management/mem_base.h source/mem_management/mem_base.cpp
        source/mem_management/mem_management.h source/mem_management/mem_management.cpp source/io/h5io.h source/io/h5io.cpp)
add_executable(IOtest tests/IOtest.cpp source/io/h5io.h source/io/h5io.cpp source/io/DataSetIO.cpp source/io/DataSetIO.h)
add_executable(ReadData source/io/h5io.cpp source/io/h5io.h)

target_link_libraries(IntelQuant mkldnn)
target_link_libraries(Intel mkldnn)
target_link_libraries(Aca2018 mkldnn hdf5_hl_cpp hdf5_cpp hdf5 szip z dl gtest gmock)
target_link_libraries(Tests hdf5_hl_cpp hdf5_cpp hdf5 szip z dl gtest gmock mkldnn)
target_link_libraries(ReadData hdf5_hl_cpp hdf5_cpp hdf5 szip z dl)
target_link_libraries(IOtest hdf5_hl_cpp hdf5_cpp hdf5 szip z dl gtest gmock)


include_directories(cmake-build-debug/googletest-src/googletest/include)
include_directories(resources/hdf5/HDF_Group/HDF5/1.10.1/include)
include_directories(/usr/local/include)
include_directories(source/neural_network)
include_directories(source/io)