//
// Created by luca on 26/05/18.
//

#ifndef ACA2018_MEM_BASE_H
#define ACA2018_MEM_BASE_H

#include "mkldnn_types.h"
#include "mkldnn.hpp"

using namespace mkldnn;

struct membase{
    memory * memref;
    float scale;

    membase(memory* mem, float scales);
    membase(memory::primitive_desc &pd, void* data, float scales);
    membase(memory::primitive_desc &pd, void* data);
    membase(const memory::dims&, memory::format, memory::data_type , void*data, float scales);
    membase(const memory::dims&, memory::format, memory::data_type , void*data);
    std::vector<int> get_shape();
    ~membase();
};

engine get_glob_engine();

#endif //ACA2018_MEM_BASE_H