//
// Created by gianpaolo on 5/23/18.
//

#include <numeric>
#include <cmath>
#include <iostream>
#include "FPNetwork.h"

FPNetwork::FPNetwork(const memory::dims &input_size) : AbsNet(input_size) {
    input_tz = input_size;
    auto user_src = new std::vector<float>(std::accumulate(
            input_size.begin(), input_size.end(), 1,
            std::multiplies<uint32_t>()));
    auto user_src_memory
            = new memory({ { { input_size }, memory::data_type::f32,
                         memory::format::nchw },
                       cpu_engine },
                     user_src->data());
    last_output = user_src_memory;
    last_output_shape = input_tz;
}

AbsNet * FPNetwork::createNet(const memory::dims &input_size) {
    return new FPNetwork(input_size);
}

AbsNet * FPNetwork::addConv2D(int channels_out, const int *kernel_size, const int *strides, Padding padding) {
    memory::dims in_shape = last_output_shape;

    /**
     * Channels_out: the number of channels outputs by the convolution
     * in_shape[1]: should be the number of channels of the input tensor
     * kernel_size[0:1]: is the effective dimension of the kernel function
     */

    memory::dims conv_weights_tz = { channels_out, in_shape[1], kernel_size[0], kernel_size[1] };
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
                     ceil((in_shape[2]-kernel_size[0]+1)/(float)strides[0]),
                     ceil((in_shape[3]-kernel_size[1]+1)/(float)strides[1])};
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

void FPNetwork::createConv2D(memory::dims conv_src_tz, memory::dims conv_weights_tz, memory::dims conv_bias_tz,
                             memory::dims conv_strides, memory::dims conv_dst_tz, memory::dims padding) {

    auto conv_weights = new std::vector<float>(std::accumulate(
            conv_weights_tz.begin(), conv_weights_tz.end(), 1,
            std::multiplies<uint32_t>()));
    auto conv_bias = new std::vector<float>(std::accumulate(conv_bias_tz.begin(),
                                                  conv_bias_tz.end(), 1, std::multiplies<uint32_t>()));

    /* create memory for user data */
    auto conv_user_weights_memory
            = new memory({ { { conv_weights_tz }, memory::data_type::f32,
                         memory::format::oihw },
                       cpu_engine },
                     conv_weights->data());

    auto conv_user_bias_memory
            = new memory({ { { conv_bias_tz }, memory::data_type::f32,
                         memory::format::x },
                       cpu_engine },
                     conv_bias->data());

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

    /* create a convolution */
    auto conv_desc = convolution_forward::desc(
            prop_kind::forward_inference, convolution_direct, conv_src_md,
            conv_weights_md, conv_bias_md, conv_dst_md, conv_strides,
            padding, padding, padding_kind::zero);
    auto conv_prim_desc
            = convolution_forward::primitive_desc(conv_desc, cpu_engine);

    auto conv_src_memory = last_output;
    if (memory::primitive_desc(conv_prim_desc.src_primitive_desc())
        != conv_src_memory->get_primitive_desc()) {
        std::cout << "Reordering source memory" << std::endl;
        conv_src_memory = new memory(conv_prim_desc.src_primitive_desc());
        net.push_back(reorder(*last_output, *conv_src_memory));
    }

    auto conv_weights_memory = conv_user_weights_memory;
    if (memory::primitive_desc(conv_prim_desc.weights_primitive_desc())
        != conv_user_weights_memory->get_primitive_desc()) {
        std::cout << "Reordering weights memory" << std::endl;
        conv_weights_memory
                = new memory(conv_prim_desc.weights_primitive_desc());
        net_weights.push_back(
                reorder(*conv_user_weights_memory, *conv_weights_memory));
    }

    auto conv_dst_memory = new memory(conv_prim_desc.dst_primitive_desc());

    /* create convolution primitive and add it to net */
    net.push_back(convolution_forward(conv_prim_desc, *conv_src_memory,
                                      *conv_weights_memory, *conv_user_bias_memory,
                                      *conv_dst_memory));

    /* AlexNet: ReLu
    * {batch, 256, 27, 27} -> {batch, 256, 27, 27}
    */
    const float negative2_slope = 1.0f;

    /* create relu primitive and add it to net */
    auto relu2_desc = eltwise_forward::desc(prop_kind::forward_inference,
                                            algorithm::eltwise_relu,
                                            conv_dst_memory->get_primitive_desc().desc(), negative2_slope);
    auto relu2_prim_desc
            = eltwise_forward::primitive_desc(relu2_desc, cpu_engine);

    net.push_back(eltwise_forward(relu2_prim_desc, *conv_dst_memory, *conv_dst_memory));

    last_output = conv_dst_memory;
    last_output_shape = conv_dst_tz;
}


AbsNet *FPNetwork::addPool2D(const int *kernel_size, Pooling pooling_algorithm, Padding padding) {
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
                     ceil((in_shape[2]-kernel_size[0]+1)/(float)(pool_strides[0])),
                     ceil((in_shape[3]-kernel_size[1]+1)/(float)(pool_strides[1]))};
        pool_padding = {0, 0};
    }
    std::cout << "Initialized pool dimensions" << std::endl;
    try {
        createPool2D(pool_out_shape, pool_kernel, pool_strides, pool_padding, pool_alg);
    } catch (error &e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
        throw;
    }

    return this;
}


void FPNetwork::createPool2D(memory::dims pool_dst_tz, memory::dims pool_kernel, memory::dims pool_strides,
                             memory::dims pool_padding, algorithm pool_algorithm) {

    auto pool1_dst_md = memory::desc({ pool_dst_tz }, memory::data_type::f32, memory::format::any);

    /* create a pooling */
    auto pool1_desc = pooling_forward::desc(prop_kind::forward_inference,
                                            pool_algorithm, last_output->get_primitive_desc().desc(),
                                            pool1_dst_md, pool_strides, pool_kernel, pool_padding,
                                            pool_padding, padding_kind::zero);
    auto pool1_pd = pooling_forward::primitive_desc(pool1_desc, cpu_engine);
    auto pool_dst_memory = new memory(pool1_pd.dst_primitive_desc());

    /* create pooling primitive an add it to net */
    net.push_back(
            pooling_forward(pool1_pd, *last_output, *pool_dst_memory));

    last_output = pool_dst_memory;
    last_output_shape = pool_dst_tz;
}

void FPNetwork::setup_net() {
    if (!net_weights.empty()) {
        try {
            stream(stream::kind::eager).submit(net_weights).wait();
        } catch (error &e) {
            std::cerr << "status: " << e.status << std::endl;
            std::cerr << "message: " << e.message << std::endl;
            throw;
        }
    }
}
