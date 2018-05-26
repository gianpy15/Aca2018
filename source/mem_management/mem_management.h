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
    ~ParametersManager();
    membase *allocate_parameters(memory::primitive_desc &dst_desc, membase *src_data, std::vector<primitive> setup_ops);

    membase *allocate_parameters(memory::primitive_desc &dst_desc, float dst_scale, membase *src_data,
                                 std::vector<primitive> setup_ops);
private:
    std::vector<membase*> allocated_memory = std::vector<membase*>();
};

class DataPipelineManager{
public:
    ~DataPipelineManager();
    membase* get_last_output();
    membase* allocate_src(memory::primitive_desc& src_desc, float scale, std::vector<primitive> inference_ops);
    membase* allocate_dst(memory::primitive_desc& dst_desc, float scale);
private:
    std::vector<membase*> allocated_memory = std::vector<membase*>();
    membase* last_output = nullptr;

};
#endif //ACA2018_MEM_MANAGEMENT_H
