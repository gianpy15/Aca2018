//
// Created by luca on 26/05/18.
//

#include <mkldnn_types.h>
#include "mkldnn.hpp"
#include "mem_management.h"
#include <iostream>
#include "../logging/logging.h"


/*
 * PARAMETERS MANAGER HERE
 */

ParametersManager::~ParametersManager(){
    for (auto memobj: allocated_memory){
        delete memobj;
    }
    for (auto memobj: setup_memory){
        delete memobj;
    }
    allocated_memory.clear();
    setup_memory.clear();
}

membase* ParametersManager::allocate_parameters(memory::primitive_desc& dst_desc, float dst_scale, membase* src_data){
    if (dst_desc == src_data->memref->get_primitive_desc() and dst_scale == src_data->scale){
        allocated_memory.push_back(src_data);
        return src_data;
    }
    setup_memory.push_back(src_data);
    auto dst_mem = new membase(dst_desc, nullptr, dst_scale);
    allocated_memory.push_back(dst_mem);
    primitive_attr dst_attr;
    dst_attr.set_int_output_round_mode(round_mode::round_nearest);
    dst_attr.set_output_scales(0, {dst_scale / src_data->scale});

    auto reorder_pd = reorder::primitive_desc(src_data->memref->get_primitive_desc(),
            dst_desc, dst_attr);
    setup_ops.push_back(std::move(reorder(reorder_pd, *src_data->memref, *dst_mem->memref)));
    return dst_mem;
}

membase* ParametersManager::allocate_parameters(memory::primitive_desc& dst_desc, membase* src_data){
    return allocate_parameters(dst_desc, 1.f, src_data);
}

void ParametersManager::setup_done() {
    for (auto memobj: setup_memory){
        delete memobj;
    }
    setup_memory.clear();
}

size_t ParametersManager::memory_usage(){
    size_t acc = 0;
    auto passed_handles = std::vector<void*>();
    for (auto memobj: setup_memory){
        if (std::find(passed_handles.begin(), passed_handles.end(), memobj->memref->get_data_handle()) == passed_handles.end()){
            acc += memobj->memref->get_primitive_desc().get_size();
            passed_handles.push_back(memobj->memref->get_data_handle());
        }
    }
    for (auto memobj: allocated_memory){
        if (std::find(passed_handles.begin(), passed_handles.end(), memobj->memref->get_data_handle()) == passed_handles.end()){
            acc += memobj->memref->get_primitive_desc().get_size();
            passed_handles.push_back(memobj->memref->get_data_handle());
        }
    }
    return acc;
}

/*
 * DATA PIPELINE MANAGER HERE
 */

DataPipelineManager::~DataPipelineManager(){
    for (auto memobj: allocated_memory){
        delete memobj;
    }
    allocated_memory.clear();
}

membase * DataPipelineManager::get_last_output() {return last_output;}

membase * DataPipelineManager::allocate_dst(memory::primitive_desc &dst_desc, float scale) {

    membase * ret = nullptr;

    if (last_output == nullptr){
        ret = new membase(dst_desc, nullptr, scale);
        last_output = ret;
        allocated_memory.push_back(ret);
        return ret;
    }
    // if there are previously allocated objects just use them
    for (membase *memobj : allocated_memory) {
        // only pay attention not to reuse the last output memory as it is going to be the source memory
        if (memobj->memref->get_data_handle() != last_output->memref->get_data_handle() &&
            memobj->memref->get_primitive_desc().get_size() == dst_desc.get_size()) {
            // if one is okay, keep it
            auto pd = memobj->memref->get_primitive_desc();
            ret = new membase(pd, memobj->memref->get_data_handle(), scale);
            break;
        }
    }

    // if no existing compatible memory has already been allocated, allocate some fresh memory
    if (ret == nullptr) {
        ret = new membase(dst_desc, nullptr, scale);
    }

    last_output = ret;
    allocated_memory.push_back(ret);
    return ret;
}

membase * DataPipelineManager::allocate_src(memory::primitive_desc &src_desc, float scale) {
    membase * src_mem = nullptr;

    if (last_output == nullptr){
        src_mem = new membase(src_desc, nullptr, scale);
        last_output = src_mem;
        allocated_memory.push_back(src_mem);
        return src_mem;
    }
    // if there are previously allocated objects...
    // first check the last output: if it has the right size, no new allocation is needed
    if (last_output->memref->get_primitive_desc().get_size() == src_desc.get_size()){
        // if it is perfect, we won't even move data: it will be the source!
        if (last_output->memref->get_primitive_desc() == src_desc && last_output->scale == scale){
            return last_output;
        }
        // otherwhise just rescale and reformat on itself
        src_mem = new membase(src_desc, last_output->memref->get_data_handle(), scale);
    }
    else {
        // if the last output is not okay, look for others
        for (membase *memobj : allocated_memory) {
            if (memobj->memref->get_primitive_desc().get_size() == src_desc.get_size()) {
                // if one is okay, keep it and issue a reorder (to move data at least)
                auto pd = memobj->memref->get_primitive_desc();
                src_mem = new membase(pd, memobj->memref->get_data_handle(), scale);
                break;
            }
        }
    }

    // if no existing compatible memory has already been allocated, allocate some fresh memory
    if (src_mem == nullptr) {
        src_mem = new membase(src_desc, nullptr, scale);
    }

    primitive_attr attr;
    attr.set_int_output_round_mode(round_mode::round_nearest);
    attr.set_output_scales(0, {scale / last_output->scale});
    //log("Dst");
    //log(src_mem);
    //log("Src");
    //log(last_output);
    auto reorder_pd = reorder::primitive_desc(last_output->memref->get_primitive_desc(),
                                              src_desc, attr);
    inference_ops.push_back(std::move(reorder(reorder_pd, *last_output->memref, *src_mem->memref)));
    last_output = src_mem;
    allocated_memory.push_back(src_mem);
    return src_mem;
}

size_t DataPipelineManager::memory_usage(){
    size_t acc = 0;
    auto passed_handles = std::vector<void*>();
    for (auto memobj: allocated_memory){
        if (std::find(passed_handles.begin(), passed_handles.end(), memobj->memref->get_data_handle()) == passed_handles.end()){
            acc += memobj->memref->get_primitive_desc().get_size();
            passed_handles.push_back(memobj->memref->get_data_handle());
        }
    }
    passed_handles.clear();
    return acc;
}
