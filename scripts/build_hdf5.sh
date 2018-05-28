#!/usr/bin/env bash
cd ../resources
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.1/src/CMake-hdf5-1.10.1.tar.gz
tar -xvzf CMake-hdf5-1.10.1.tar.gz
mv CMake-hdf5-1.10.1 hdf5
cd hdf5
./build-unix.sh
./HDF5-1.10.1-Linux.sh
cd ..
rm CMake-hdf5-1.10.1.tar.gz