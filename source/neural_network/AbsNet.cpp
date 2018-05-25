//
// Created by gianpaolo on 5/23/18.
//
#include <mkldnn_types.h>
#include "AbsNet.h"

void AbsNet::run_net(int times) {
    if (!net.empty()) {
        try {
            auto start_time = std::chrono::high_resolution_clock::now();
            for (int i=0; i<times; i++) {
                stream(stream::kind::eager).submit(net).wait();
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            std::cout << "Runs: " << times << std::endl
                      << "Time elapsed(ms): " << (double)(end_time-start_time).count()/1e+6 << std::endl;
        } catch (error &e) {
            std::cerr << "status: " << e.status << std::endl;
            std::cerr << "message: " << e.message << std::endl;
            throw;
        }
    }
}

void AbsNet::run_net(){
    run_net(1);
}

void AbsNet::setup_net() {
    if (!net_weights.empty()) {
        try {
            stream(stream::kind::eager).submit(net_weights).wait();
        } catch (error &e) {
            std::cerr << "status: " << e.status << std::endl;
            std::cerr << "message: " << e.message << std::endl;
            throw;
        }
    }

    for (auto memobj : temporary_memobjs){
        free(memobj->get_data_handle());
        delete memobj;
    }
    temporary_memobjs.clear();

    for (auto memobj : tmp_vecs){
        memobj->clear();
    }
    tmp_vecs.clear();
}

AbsNet::AbsNet(const memory::dims &input_size): input_tz(input_size) {

}

AbsNet *AbsNet::addConv2D(int channels_out, const int *kernel_size, const int *strides, Padding padding) {
    memory::dims in_shape = last_output_shape;

    /**
     * Channels_out: the number of channels outputs by the convolution
     * in_shape[1]: should be the number of channels of the input tensor
     * kernel_size[0:1]: is the effective dimension of the kernel function
     */

    memory::dims conv_weights_tz = {channels_out, in_shape[1], kernel_size[0], kernel_size[1] };
    memory::dims conv_bias_tz = { channels_out };
    memory::dims conv_strides = { strides[0], strides[1] };
    memory::dims out_shape;
    memory::dims padding_tz;

    if (padding == Padding::SAME){
        out_shape = {in_shape[0],
                     channels_out,
                     in_shape[2]/strides[0],
                     in_shape[3]/strides[1]};
        padding_tz = {(kernel_size[0] - 1)/2, (kernel_size[1] - 1)/2};
    } else {
        out_shape = {in_shape[0],
                     channels_out,
                     (int)ceil((in_shape[2]-kernel_size[0]+1)/(double)strides[0]),
                     (int)ceil((in_shape[3]-kernel_size[1]+1)/(double)strides[1])};
        padding_tz = {0, 0};
    }
    try {
        createConv2D(in_shape, conv_weights_tz, conv_bias_tz, conv_strides, out_shape, padding_tz);
    } catch (error &e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
        throw;
    }

    return this;
}

AbsNet *AbsNet::addPool2D(const int *kernel_size, Pooling pooling_algorithm, Padding padding) {
    memory::dims in_shape = last_output_shape;
    memory::dims pool_out_shape;
    memory::dims pool_kernel = { kernel_size[0], kernel_size[1] };
    memory::dims pool_strides = { kernel_size[0], kernel_size[1] };
    memory::dims pool_padding;

    algorithm pool_alg = pooling_algorithm == MAX ? algorithm::pooling_max : algorithm::pooling_avg;

    if (padding == Padding::SAME){
        pool_out_shape = {in_shape[0],
                          in_shape[1],
                          in_shape[2]/(pool_strides[0]),
                          in_shape[3]/(pool_strides[1])},
                pool_padding = {(kernel_size[0] - 1)/2, (kernel_size[1] - 1)/2};
    } else {
        pool_out_shape = {in_shape[0],
                          in_shape[1],
                          (int)ceil((in_shape[2]-kernel_size[0]+1)/(double)(pool_strides[0])),
                          (int)ceil((in_shape[3]-kernel_size[1]+1)/(double)(pool_strides[1]))};
        pool_padding = {0, 0};
    }
    // std::cout << "Initialized pool dimensions" << std::endl;
    try {
        createPool2D(pool_out_shape, pool_kernel, pool_strides, pool_padding, pool_alg);
    } catch (error &e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
        throw;
    }

    return this;
}

size_t AbsNet::total_memory_usage() {
    size_t acc = 0;
    for (auto memobj: memobjs){
        acc += memobj->get_primitive_desc().get_size();
    }
    for (auto memobj: tmp_vecs){
        acc += memobj->size() * sizeof(float);
    }
    for (auto memobj: temporary_memobjs){
        acc += memobj->get_primitive_desc().get_size();
    }
    return acc;
}

std::vector<float>* AbsNet::generate_vec(const memory::dims& dims) {

    size_t size=1;
    for (auto elem: dims)
        size *= elem;
    auto vec = new std::vector<float>(size);
    tmp_vecs.push_back(vec);
    return vec;
}

AbsNet::~AbsNet() {

    for (auto memobj : temporary_memobjs){
        //delete memobj->get_data_handle();
        delete memobj;
    }
    temporary_memobjs.clear();

    for (auto memobj : memobjs){
        //delete memobj->get_data_handle();
        delete memobj;
    }
    memobjs.clear();

    for (auto memobj : tmp_vecs){
        memobj->clear();
    }
    tmp_vecs.clear();

}

AbsNet *AbsNet::addFC(int outputs) {
    int inputs = last_output_shape[1];
    int batch_size = last_output_shape[0];
    memory::dims weights_shape = { inputs, outputs };
    memory::dims biases_shape = { outputs };
    memory::dims output_shape = { batch_size, outputs };

    std::cout << "initialized fc dimensions" << std::endl;

    try {
        createFC(output_shape, weights_shape, biases_shape);
    } catch (error &e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
        throw;
    }

    return this;
}

void AbsNet::createConv2D(const memory::dims& conv_src_tz,
                          const memory::dims& conv_weights_tz,
                          const memory::dims& conv_bias_tz,
                          const memory::dims& conv_strides,
                          const memory::dims& conv_dst_tz,
                          const memory::dims& padding){
    auto conv_weights = generate_vec(conv_weights_tz);
    auto conv_bias = generate_vec(conv_bias_tz);

    /* create memory for user data */
    auto conv_weights_memory
            = new memory({ { { conv_weights_tz }, memory::data_type::f32,
                             memory::format::oihw },
                           cpu_engine },
                         conv_weights->data());

    auto conv_bias_memory
            = new memory({ { { conv_bias_tz }, memory::data_type::f32,
                             memory::format::x },
                           cpu_engine },
                         conv_bias->data());
    createConv2D(conv_src_tz,
                 conv_weights_tz,
                 conv_bias_tz,
                 conv_strides,
                 conv_dst_tz,
                 padding,
                 conv_weights_memory,
                 conv_bias_memory);
}

void AbsNet::createFC(const memory::dims& fc_dst_tz, const memory::dims& fc_weights_tz, const memory::dims& fc_bias_tz) {
    auto weights = generate_vec(fc_weights_tz);
    auto bias = generate_vec(fc_bias_tz);

    /* create memory for user data */
    auto weights_memory
            = new memory({ { { fc_weights_tz }, memory::data_type::f32,
                             memory::format::nc },
                           cpu_engine },
                         weights->data());

    auto bias_memory
            = new memory({ { { fc_bias_tz }, memory::data_type::f32,
                             memory::format::x },
                           cpu_engine },
                         bias->data());

    createFC(fc_dst_tz, fc_weights_tz, fc_bias_tz, weights_memory, bias_memory);
}

memory * AbsNet::make_reorder(std::vector<primitive>& netops, memory *src, memory *dst,
                                const int mask, const std::vector<float>& scales) {

    if(src->get_primitive_desc().desc().data.data_type != mkldnn_f32
       || dst->get_primitive_desc().desc().data.data_type != mkldnn_f32){
        primitive_attr dst_attr;
        dst_attr.set_int_output_round_mode(round_mode::round_nearest);
        dst_attr.set_output_scales(mask, scales);
        auto reorder_pd
                = reorder::primitive_desc(src->get_primitive_desc(),
                                          dst->get_primitive_desc(), dst_attr);
        primitive * ret = new reorder(reorder_pd, *src, *dst);
        netops.push_back(*ret);
        return dst;
    }
    else{
        primitive * ret = new reorder(*src, *dst);
        netops.push_back(*ret);
        return dst;
    }
}

memory * AbsNet::make_reorder(std::vector<primitive>& netops, memory *src, memory *dst) {
    return make_reorder(netops, src, dst, 0, std::vector<float>(0));
}

memory * AbsNet::make_conditional_reorder(std::vector<primitive> &netops, memory *src, memory::primitive_desc &dst,
                                          std::vector<memory *> &memtracker) {

    if (dst != src->get_primitive_desc()) {
        auto mem = new memory(dst);
        memtracker.push_back(src);
        auto ret = new reorder(*src, *mem);
        netops.push_back(*ret);
        return mem;
    }

    return src;
}