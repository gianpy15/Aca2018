//
// Created by luca on 01/06/18.
//

#include "logging.h"
#include <iostream>
#include <fstream>
#include <mkldnn_types.h>

bool shallLog = true;

void log(std::string str){
    if (!shallLog)
        return;
    std::cout << str << std::endl;
}

void log(long long n){
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

void log(std::string label, long long n){
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
    std::cout << "\tengine ptr: " << mb->memref->get_primitive_desc().get_engine().get() << std::endl;
    std::cout << "\tdata ptrr: " << mb->memref->get_data_handle() << std::endl;
    std::cout << "\tlayout_bdims ptr: " << mb->memref->get_primitive_desc().desc().data.layout_desc.blocking.block_dims << std::endl;
    std::cout << "\toffset padding (data) ptr: " << mb->memref->get_primitive_desc().desc().data.layout_desc.blocking.offset_padding_to_data << std::endl;
    std::cout << "\tlayout_pdims ptr: " << mb->memref->get_primitive_desc().desc().data.layout_desc.blocking.padding_dims << std::endl;
    std::cout << "\toffset padding ptr: " << mb->memref->get_primitive_desc().desc().data.layout_desc.blocking.offset_padding << std::endl;
    std::cout << "\tstrides ptr: " << mb->memref->get_primitive_desc().desc().data.layout_desc.blocking.strides << std::endl;


    std::cout << "}" << std::endl;
}

std::string error_message(int status){
    switch (status){
        case 0:
            return "The operation was successful.";
        case 1:
            return "The operation failed due to an out-of-memory condition.";
        case 2:
            return "The operation failed and should be retried.";
        case 3:
            return "The operation failed because of incorrect function arguments.";
        case 4:
            return "The operation failed because a primitive was not ready for execution.";
        case 5:
            return "The operation failed because requested functionality is not implemented.";
        case 6:
            return "Primitive iterator passed over last primitive descriptor.";
        case 7:
            return "Primitive or engine failed on execution.";
        case 8:
            return "Queried element is not required for given primitive.";
        default:
            return "Unknown status code " + status;
    }
}


std::ofstream& openStream(std::string& path){
    auto os = new std::ofstream("../resources/" + path + ".csv", std::ios::app);
    return *os;
}

Logger::Logger(std::string&& path): logfile(openStream(path)), lineBegin(true){}

void Logger::endLine() {
    logfile << std::endl;
    lineBegin = true;
}

Logger::~Logger() {
    if (!lineBegin)
        endLine();
    logfile.close();
}