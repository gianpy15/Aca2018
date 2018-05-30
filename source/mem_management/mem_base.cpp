//
// Created by luca on 26/05/18.
//

#include <mkldnn_types.h>
#include "mkldnn_types.h"
#include "mkldnn.hpp"
#include "mem_base.h"
#include <iostream>

using namespace mkldnn;

engine* glob_eng = nullptr;

engine get_glob_engine(){
    if (glob_eng == nullptr){
        glob_eng = new engine(engine::cpu, 0);
    }
    return *glob_eng;
}

membase::membase(memory * mem, float scales) {
    memref = mem;
    scale = scales;
}

membase::membase(memory::primitive_desc &pd, void *data, float scales):
        membase(data==nullptr? new memory(pd) : new memory(pd, data), scales) {}

membase::membase(memory::primitive_desc &pd, void *data): membase(pd, data, 1.f) {}

membase::membase(const memory::dims &dims, memory::format fmt, memory::data_type dtype, void *data, float scales){
    auto d = memory::desc(dims, dtype, fmt);
    auto pd = memory::primitive_desc(d, get_glob_engine());
    memref = data==nullptr? new memory(pd) : new memory(pd, data);
    scale = scales;
}

membase::membase(const memory::dims &dims, memory::format fmt, memory::data_type dtype, void *data):
        membase(dims, fmt, dtype, data, 1.f) {}

membase::~membase() {
    delete memref;
}

std::vector<int> membase::get_shape() {
    int ndims = memref->get_primitive_desc().desc().data.ndims;
    int* elems = memref->get_primitive_desc().desc().data.dims;
    std::vector<int> dims(ndims);
    for (int i=0; i<ndims; i++) {
        dims[i] = elems[i]; // NO IDEA WHY THE DIMS ARRAY DESCRIPTOR IS FULL OF ZEROES!
    }
    std::cerr << "Called membase::get_shape. This function is not working, don't call it!" << std::endl;
    return dims;
}