//
// Created by gianpaolo on 5/23/18.
//

#include <numeric>
#include <cmath>
#include <iostream>
#include "FPNetwork.h"

FPNetwork::FPNetwork(const memory::dims &input_size) : AbsNet(input_size) {
    input_tz = input_size;
    std::vector<float> user_src(std::accumulate(
            input_size.begin(), input_size.end(), 1,
            std::multiplies<uint32_t>()));
    auto user_src_memory
            = new memory({ { { input_size }, memory::data_type::f32,
                         memory::format::nchw },
                       cpu_engine },
                     user_src.data());
    last_output = user_src_memory;
    last_output_shape = input_tz;
}

AbsNet * FPNetwork::addConv2D(int channels_out, const int *kernel_size, const int *strides, Padding padding) {
    std::cout << "CONV SETUP CHECKPOINT 0" << std::endl;

    memory::dims in_shape = last_output_shape;

    /**
     * Channels_out: the number of channels outputs by the convolution
     * in_shape[1]: should be the number of channels of the input tensor
     * kernel_size[0:1]: is the effective dimension of the kernel function
     */
    std::cout << "CONV SETUP CHECKPOINT 1" << std::endl;

    memory::dims conv_weights_tz = { channels_out, in_shape[1], kernel_size[0], kernel_size[1] };
    memory::dims conv_bias_tz = { channels_out };
    std::cout << "CONV SETUP CHECKPOINT 2" << std::endl;
    memory::dims conv_strides = { strides[0], strides[1] };
    std::cout << "CONV SETUP CHECKPOINT 3" << std::endl;
    memory::dims out_shape;
    memory::dims padding_tz;

    std::cout << "CONV SETUP CHECKPOINT 4" << std::endl;

    if (padding == Padding::SAME){
        out_shape = {in_shape[0],
                     channels_out,
                     in_shape[2]/strides[0],
                     in_shape[3]/strides[1]};
        padding_tz = {(kernel_size[0] - 1)/2, (kernel_size[1] - 1)/2};
    } else {
        out_shape = {in_shape[0],
                     channels_out,
                     ceil((in_shape[2]-kernel_size[0]+1)/(float)strides[0]),
                     ceil((in_shape[3]-kernel_size[1]+1)/(float)strides[1])};
        padding_tz = {0, 0};
    }
    std::cout << "Initialized conv dimensions" << std::endl;
    try {
        createConv2D(in_shape, conv_weights_tz, conv_bias_tz, conv_strides, out_shape, padding_tz);
    } catch (error &e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
        throw;
    }

    return this;
}

void FPNetwork::createConv2D(memory::dims conv_src_tz,
                             memory::dims conv_weights_tz,
                             memory::dims conv_bias_tz,
                             memory::dims conv_strides,
                             memory::dims conv_dst_tz,
                             memory::dims padding) {

    /* AlexNet: conv
    * {batch, 96, 27, 27} (x) {2, 128, 48, 5, 5} -> {batch, 256, 27, 27}
    * strides: {1, 1}
    */
    std::cout << "CONV CHECKPOINT 0" << std::endl;

    std::vector<float> conv_weights(std::accumulate(
            conv_weights_tz.begin(), conv_weights_tz.end(), 1,
            std::multiplies<uint32_t>()));
    std::vector<float> conv_bias(std::accumulate(conv_bias_tz.begin(),
                                                  conv_bias_tz.end(), 1, std::multiplies<uint32_t>()));

    /* create memory for user data */
    auto conv_user_weights_memory
            = new memory({ { { conv_weights_tz }, memory::data_type::f32,
                         memory::format::oihw },
                       cpu_engine },
                     conv_weights.data());

    auto conv_user_bias_memory
            = new memory({ { { conv_bias_tz }, memory::data_type::f32,
                         memory::format::x },
                       cpu_engine },
                     conv_bias.data());

    std::cout << "CONV CHECKPOINT 1" << std::endl;

    /* create memory descriptors for convolution data w/ no specified format
     */
    auto conv_src_md = memory::desc(
            { conv_src_tz }, memory::data_type::f32, memory::format::any);
    auto conv_bias_md = memory::desc(
            { conv_bias_tz }, memory::data_type::f32, memory::format::any);
    auto conv_weights_md = memory::desc({ conv_weights_tz },
                                         memory::data_type::f32, memory::format::any);
    auto conv_dst_md = memory::desc(
            { conv_dst_tz }, memory::data_type::f32, memory::format::any);

    std::cout << "CONV CHECKPOINT 2" << std::endl;

    /* create a convolution */
    auto conv_desc = convolution_forward::desc(
            prop_kind::forward_inference, convolution_direct, conv_src_md,
            conv_weights_md, conv_bias_md, conv_dst_md, conv_strides,
            padding, padding, padding_kind::zero);
    auto conv_prim_desc
            = new convolution_forward::primitive_desc(conv_desc, cpu_engine);

    std::cout << "CONV CHECKPOINT 2" << std::endl;

    auto conv_src_memory = last_output;
    if (memory::primitive_desc(conv_prim_desc->src_primitive_desc())
        != conv_src_memory->get_primitive_desc()) {
        std::cout << "Reordering source memory" << std::endl;
        conv_src_memory = new memory(conv_prim_desc->src_primitive_desc());
        net.push_back(reorder(*last_output, *conv_src_memory));
    }

    auto conv_weights_memory = conv_user_weights_memory;
    if (memory::primitive_desc(conv_prim_desc->weights_primitive_desc())
        != conv_user_weights_memory->get_primitive_desc()) {
        std::cout << "Reordering weights memory" << std::endl;
        conv_weights_memory
                = new memory(conv_prim_desc->weights_primitive_desc());
        net_weights.push_back(
                reorder(*conv_user_weights_memory, *conv_weights_memory));
    }

    std::cout << "CONV CHECKPOINT 3" << std::endl;

    auto conv_dst_memory = new memory(conv_prim_desc->dst_primitive_desc());

    /* create convolution primitive and add it to net */
    net.push_back(convolution_forward(*conv_prim_desc, *conv_src_memory,
                                      *conv_weights_memory, *conv_user_bias_memory,
                                      *conv_dst_memory));

    std::cout << "CONV CHECKPOINT 4" << std::endl;

    /* AlexNet: ReLu
    * {batch, 256, 27, 27} -> {batch, 256, 27, 27}
    */
    const float negative2_slope = 1.0f;

    /* create relu primitive and add it to net */
    auto relu2_desc = new eltwise_forward::desc(prop_kind::forward_inference,
                                            algorithm::eltwise_relu,
                                            conv_dst_memory->get_primitive_desc().desc(), negative2_slope);
    auto relu2_prim_desc
            = new eltwise_forward::primitive_desc(*relu2_desc, cpu_engine);

    std::cout << "CONV CHECKPOINT 5" << std::endl;

    net.push_back(eltwise_forward(*relu2_prim_desc, *conv_dst_memory, *conv_dst_memory));

    last_output = conv_dst_memory;
    last_output_shape = conv_dst_tz;
}
