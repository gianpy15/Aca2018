//
// Created by luca on 26/05/18.
//

#ifndef ACA2018_MEM_MANAGEMENT_H
#define ACA2018_MEM_MANAGEMENT_H

#include "mkldnn.hpp"
#include "mem_base.h"
using namespace mkldnn;

class ParametersManager {
public:
    explicit ParametersManager(std::vector<primitive>& setup_ops): setup_ops(setup_ops) {}
    ~ParametersManager();
    membase *allocate_parameters(memory::primitive_desc &dst_desc, membase *src_data);

    membase *allocate_parameters(memory::primitive_desc &dst_desc, float dst_scale, membase *src_data);
    void setup_done();
    size_t memory_usage();
private:
    std::vector<primitive>& setup_ops;
    std::vector<membase*> setup_memory = std::vector<membase*>();
    std::vector<membase*> allocated_memory = std::vector<membase*>();
};

class DataPipelineManager{
public:
    explicit DataPipelineManager(std::vector<primitive>& inference_ops): inference_ops(inference_ops) {}
    ~DataPipelineManager();
    size_t memory_usage();
    membase* get_last_output();
    membase* allocate_src(memory::primitive_desc& src_desc, float scale=1.f);
    membase* allocate_dst(memory::primitive_desc& dst_desc, float scale=1.f);
    membase* last_output = nullptr;
private:
    std::vector<primitive>& inference_ops;
    std::vector<membase*> allocated_memory = std::vector<membase*>();

};
#endif //ACA2018_MEM_MANAGEMENT_H
