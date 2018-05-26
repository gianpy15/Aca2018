//
// Created by luca on 26/05/18.
//

#include "mkldnn_types.h"
#include "mkldnn.hpp"
#include "mem_base.h"

using namespace mkldnn;

membase::membase(memory * mem, float scales) {
    memref = mem;
    scale = scales;
}

membase::membase(memory::primitive_desc &pd, void *data, float scales):
        membase(data==nullptr? new memory(pd) : new memory(pd, data), scales) {}

membase::membase(memory::primitive_desc &pd, void *data): membase(pd, data, 1.f) {}

membase::membase(memory::dims &dims, memory::format &fmt, memory::data_type &dtype, void *data, float scales){
    engine eng = engine(engine::cpu, 0);
    auto d = memory::desc(dims, dtype, fmt);
    auto pd = memory::primitive_desc(d, eng);
    memref = data==nullptr? new memory(pd) : new memory(pd, data);
    scale = scales;
}

membase::membase(memory::dims &dims, memory::format &fmt, memory::data_type &dtype, void *data):
        membase(dims, fmt, dtype, data, 1.f) {}

membase::~membase() {
    delete memref;
}