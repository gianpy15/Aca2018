//
// Created by gianpaolo on 5/23/18.
//

#include <numeric>
#include <cmath>
#include <iostream>
#include "FPNetwork.h"
#include "../logging/logging.h"

FPNetwork::FPNetwork(const memory::dims &input_size) : AbsNet(input_size) {}

AbsNet * FPNetwork::createNet(const memory::dims &input_size) {
    return new FPNetwork(input_size);
}

void FPNetwork::createConv2D(const memory::dims& conv_src_tz, const memory::dims& conv_weights_tz, const memory::dims& conv_bias_tz,
                             const memory::dims& conv_strides, const memory::dims& conv_dst_tz, const memory::dims& padding,
                             membase* conv_user_weights_memory, membase* conv_user_bias_memory) {

    /* build up the description of the memory layout for the convolution */
    auto conv_src_md = memory::desc(
            { conv_src_tz }, memory::data_type::f32, memory::format::any);
    auto conv_bias_md = memory::desc(
            { conv_bias_tz }, memory::data_type::f32, memory::format::any);
    auto conv_weights_md = memory::desc(
            { conv_weights_tz }, memory::data_type::f32, memory::format::any);
    auto conv_dst_md = memory::desc(
            { conv_dst_tz }, memory::data_type::f32, memory::format::any);

    /*
    log("CONV");
    log(conv_src_tz, 4);
    log(conv_bias_tz, 1);
    log(conv_weights_tz, 4);
    log(conv_dst_tz, 4);
    */
    /* build up the description of the convolution */

    primitive_attr conv_attr;
    const float ops_scale = 1.f;
    const float ops_alpha = 0.f; // relu negative slope
    const float ops_beta = 0.f;
    post_ops ops;
    ops.append_eltwise(ops_scale, algorithm::eltwise_relu, ops_alpha, ops_beta);
    conv_attr.set_post_ops(ops);

    auto conv_desc = convolution_forward::desc(
            prop_kind::forward_inference, convolution_direct, conv_src_md,
            conv_weights_md, conv_bias_md, conv_dst_md, conv_strides,
            padding, padding, padding_kind::zero);
    auto conv_prim_desc
            = convolution_forward::primitive_desc(conv_desc, conv_attr, cpu_engine);

    /* allocate the necessary memory and issue reorders */
    auto tmp = conv_prim_desc.src_primitive_desc();
    auto conv_src_memory = dataPipelineManager->allocate_src(tmp);
    tmp = conv_prim_desc.weights_primitive_desc();
    auto conv_weights_memory = parametersManager->allocate_parameters(tmp, conv_user_weights_memory);
    tmp = conv_prim_desc.bias_primitive_desc();
    auto conv_bias_memory = parametersManager->allocate_parameters(tmp, conv_user_bias_memory);
    tmp = conv_prim_desc.dst_primitive_desc();
    auto conv_dst_memory = dataPipelineManager->allocate_dst(tmp);

    /* create convolution primitive and add it to inference_ops */
    inference_ops.push_back(std::move(convolution_forward(conv_prim_desc, *conv_src_memory->memref,
                                      *conv_weights_memory->memref, *conv_bias_memory->memref,
                                      *conv_dst_memory->memref)));

    /*
    const float negative2_slope = 1.0f;

    auto relu2_desc = eltwise_forward::desc(prop_kind::forward_inference,
                                            algorithm::eltwise_relu,
                                            conv_dst_memory->memref->get_primitive_desc().desc(), negative2_slope);
    auto relu2_prim_desc
            = eltwise_forward::primitive_desc(relu2_desc, cpu_engine);
    inference_ops.push_back(std::move(eltwise_forward(relu2_prim_desc, *conv_dst_memory->memref, *conv_dst_memory->memref)));
    */
    last_output = conv_dst_memory;
    last_output_shape = conv_dst_tz;
}

void FPNetwork::createPool2D(const memory::dims& pool_dst_tz, const memory::dims& pool_kernel, const memory::dims& pool_strides,
                             const memory::dims& pool_padding, algorithm pool_algorithm) {

    auto pool1_dst_md = memory::desc({ pool_dst_tz }, memory::data_type::f32, memory::format::any);

    /* create a pooling */
    auto pool1_desc = pooling_forward::desc(prop_kind::forward_inference,
                                            pool_algorithm, last_output->memref->get_primitive_desc().desc(),
                                            pool1_dst_md, pool_strides, pool_kernel, pool_padding,
                                            pool_padding, padding_kind::zero);
    auto pool1_pd = pooling_forward::primitive_desc(pool1_desc, cpu_engine);
    auto tmp = pool1_pd.src_primitive_desc();
    auto pool_src_memory = dataPipelineManager->allocate_src(tmp, last_output->scale);
    tmp = pool1_pd.dst_primitive_desc();
    auto pool_dst_memory = dataPipelineManager->allocate_dst(tmp, last_output->scale);


    inference_ops.push_back(
            std::move(pooling_forward(pool1_pd, *pool_src_memory->memref, *pool_dst_memory->memref)));

    last_output = pool_dst_memory;
    last_output_shape = pool_dst_tz;
}