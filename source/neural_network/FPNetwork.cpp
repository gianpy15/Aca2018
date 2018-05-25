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
        conv_src_memory = new memory(conv_prim_desc.src_primitive_desc());
        net.push_back(reorder(*last_output, *conv_src_memory));
    }

    auto conv_weights_memory = conv_user_weights_memory;
    if (memory::primitive_desc(conv_prim_desc.weights_primitive_desc())
        != conv_user_weights_memory->get_primitive_desc()) {
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

void FPNetwork::createFC(memory::dims fc_dst_tz, memory::dims fc_weights_tz, memory::dims fc_bias_tz) {
    std::vector<float> fc_weights(std::accumulate(fc_weights_tz.begin(),
                                                   fc_weights_tz.end(), 1, std::multiplies<uint32_t>()));
    std::vector<float> fc_bias(std::accumulate(fc_bias_tz.begin(),
                                                fc_bias_tz.end(), 1, std::multiplies<uint32_t>()));

    /* create memory for user data */
    auto fc_user_weights_memory = new memory({ { { fc_weights_tz }, memory::data_type::f32,
                                                  memory::format::nc },  cpu_engine }, fc_weights.data());

    auto fc_user_bias_memory = new memory({ { { fc_bias_tz }, memory::data_type::f32,
                                               memory::format::x }, cpu_engine }, fc_bias.data());

    /* create memory descriptors for convolution data w/ no specified format
     */
    auto fc_bias_md = memory::desc({ fc_bias_tz }, memory::data_type::f32, memory::format::any);
    auto fc_weights_md = memory::desc({ fc_weights_tz }, memory::data_type::f32, memory::format::any);
    auto fc_dst_md = memory::desc({ fc_dst_tz }, memory::data_type::f32, memory::format::any);

    /* create a inner_product */
    auto fc_desc = inner_product_forward::desc(prop_kind::forward_inference, last_output->get_primitive_desc().desc(), fc_weights_md, fc_bias_md, fc_dst_md);
    auto fc_prim_desc = inner_product_forward::primitive_desc(fc_desc, cpu_engine);

    auto fc_weights_memory = fc_user_weights_memory;
    if (memory::primitive_desc(fc_prim_desc.weights_primitive_desc())
        != fc_user_weights_memory->get_primitive_desc()) {
        fc_weights_memory = new memory(fc_prim_desc.weights_primitive_desc());
        net.push_back(reorder(*fc_user_weights_memory, *fc_weights_memory));
    }

    auto fc_dst_memory = new memory(fc_prim_desc.dst_primitive_desc());

    /* create convolution primitive and add it to net */
    net.push_back(inner_product_forward(fc_prim_desc, *last_output,
                                        *fc_weights_memory, *fc_user_bias_memory, *fc_dst_memory));

    last_output = fc_dst_memory;
    last_output_shape = fc_dst_tz;
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
