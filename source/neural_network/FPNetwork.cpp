//
// Created by gianpaolo on 5/23/18.
//

#include <numeric>
#include <cmath>
#include <iostream>
#include "FPNetwork.h"

FPNetwork::FPNetwork(const memory::dims &input_size) : AbsNet(input_size) {
    input_tz = input_size;
    auto user_src = generate_vec(input_size);
    auto user_src_memory
            = new memory({ { { input_size }, memory::data_type::f32,
                         memory::format::nchw },
                       cpu_engine },
                     user_src->data());
    memobjs.push_back(user_src_memory);
    last_output = user_src_memory;
    last_output_shape = input_tz;
}

AbsNet * FPNetwork::createNet(const memory::dims &input_size) {
    return new FPNetwork(input_size);
}

void FPNetwork::createConv2D(memory::dims conv_src_tz, memory::dims conv_weights_tz, memory::dims conv_bias_tz,
                             memory::dims conv_strides, memory::dims conv_dst_tz, memory::dims padding,
                             memory* conv_user_weights_memory, memory* conv_user_bias_memory) {

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

    /* reorder memory if necessary */
    auto tmp = conv_prim_desc.src_primitive_desc();
    auto conv_src_memory = make_conditional_reorder(net, last_output, tmp, memobjs);
    tmp = conv_prim_desc.weights_primitive_desc();
    auto conv_weights_memory = make_conditional_reorder(net_weights, conv_user_weights_memory, tmp, temporary_memobjs);
    tmp = conv_prim_desc.bias_primitive_desc();
    auto conv_bias_memory = make_conditional_reorder(net_weights, conv_user_bias_memory, tmp, temporary_memobjs);

    auto conv_dst_memory = new memory(conv_prim_desc.dst_primitive_desc());

    memobjs.push_back(conv_weights_memory);
    memobjs.push_back(conv_user_bias_memory);
    memobjs.push_back(conv_dst_memory);
    /* create convolution primitive and add it to net */
    net.push_back(convolution_forward(conv_prim_desc, *conv_src_memory,
                                      *conv_weights_memory, *conv_bias_memory,
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
    memobjs.push_back(pool_dst_memory);
    /* create pooling primitive an add it to net */
    net.push_back(
            pooling_forward(pool1_pd, *last_output, *pool_dst_memory));

    last_output = pool_dst_memory;
    last_output_shape = pool_dst_tz;
}

void FPNetwork::createFC(memory::dims fc_dst_tz, memory::dims fc_weights_tz, memory::dims fc_bias_tz,
                         memory* fc_user_weights_memory, memory* fc_user_bias_memory) {

    /* create memory descriptors for convolution data w/ no specified format
     */
    auto fc_bias_md = memory::desc({ fc_bias_tz }, memory::data_type::f32, memory::format::any);
    auto fc_weights_md = memory::desc({ fc_weights_tz }, memory::data_type::f32, memory::format::any);
    auto fc_dst_md = memory::desc({ fc_dst_tz }, memory::data_type::f32, memory::format::any);

    /* create a inner_product */
    auto fc_desc = inner_product_forward::desc(prop_kind::forward_inference, last_output->get_primitive_desc().desc(), fc_weights_md, fc_bias_md, fc_dst_md);
    auto fc_prim_desc = inner_product_forward::primitive_desc(fc_desc, cpu_engine);

    auto tmp = fc_prim_desc.weights_primitive_desc();
    auto fc_weights_memory = make_conditional_reorder(net_weights, fc_user_weights_memory, tmp, temporary_memobjs);
    tmp = fc_prim_desc.src_primitive_desc();
    auto fc_src_memory = make_conditional_reorder(net, last_output, tmp, memobjs);
    tmp = fc_prim_desc.bias_primitive_desc();
    auto fc_bias_memory = make_conditional_reorder(net_weights, fc_user_bias_memory, tmp, temporary_memobjs);

    auto fc_dst_memory = new memory(fc_prim_desc.dst_primitive_desc());
    memobjs.push_back(fc_dst_memory);
    memobjs.push_back(fc_weights_memory);
    memobjs.push_back(fc_user_bias_memory);
    /* create convolution primitive and add it to net */
    net.push_back(inner_product_forward(fc_prim_desc, *fc_src_memory,
                                        *fc_weights_memory, *fc_bias_memory, *fc_dst_memory));

    last_output = fc_dst_memory;
    last_output_shape = fc_dst_tz;
}
