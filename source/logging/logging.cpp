//
// Created by luca on 01/06/18.
//

#include "logging.h"
#include <iostream>
#include <mkldnn_types.h>

bool shallLog = true;

void log(std::string str){
    if (!shallLog)
        return;
    std::cout << str << std::endl;
}

void log(int n){
    if (!shallLog)
        return;
    std::cout << n << std::endl;
}

void log(std::vector<int> v){
    if (!shallLog)
        return;
    std::cout << "std::vector<int>{";
    for(int i=0; i<v.size()-1; i++)
        std::cout << v[i] << ", ";
    std::cout << v[v.size()-1] <<"}" << std::endl;
}

void log(std::string label, int n){
    if (!shallLog)
        return;
    std::cout <<label << ": " << n << std::endl;
}

void log(mkldnn::memory::dims d, int size){
    if (!shallLog)
        return;
    std::cout << "mkldnn::memory::dims{";
    for (int i=0; i<size-1; i++){
        std::cout << d[i] << ", ";
    }
    std::cout << d[size-1] << "}" << std::endl;
}

void log(membase* mb){
    if (!shallLog)
        return;
    std::cout << "membase{" << std::endl;
    std::cout << "\tshape (unreliable): "; log(mb->get_shape(), mb->memref->get_primitive_desc().desc().data.ndims);
    std::cout << "\tscale: " << mb->scale << std::endl;
    std::cout << "\tsize: " << mb->memref->get_primitive_desc().get_size() << std::endl;
    std::cout << "\tformat: " << mb->memref->get_primitive_desc().desc().data.format << std::endl;
    std::cout << "\tprim_kind: " << mb->memref->get_primitive_desc().desc().data.primitive_kind << std::endl;
    std::cout << "\tndims: " << mb->memref->get_primitive_desc().desc().data.ndims << std::endl;
    std::cout << "\tdtype: " << mb->dtype() << std::endl;
    std::cout << "\tengine: " << mb->memref->get_primitive_desc().get_engine().get() << std::endl;
    std::cout << "}" << std::endl;
}