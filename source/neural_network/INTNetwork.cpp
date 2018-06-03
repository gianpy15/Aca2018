//
// Created by luca on 24/05/18.
//

#include "INTNetwork.h"
#include <numeric>
#include <cmath>
#include <iostream>
#include <mkldnn_types.h>

INTNetwork::INTNetwork(const memory::dims &input_size) : AbsNet(input_size) {}

AbsNet * INTNetwork::createNet(const memory::dims &input_size) {
    return new INTNetwork(input_size);
}

void INTNetwork::createConv2D(const memory::dims& conv_src_tz, const memory::dims& conv_weights_tz, const memory::dims& conv_bias_tz,
                              const memory::dims& conv_strides, const memory::dims& conv_dst_tz, const memory::dims& padding,
                              membase* conv_user_weights_memory, membase* conv_user_bias_memory) {

    float src_scale = 2.f;
    float weights_scale = 2.f;
    float bias_scale = 4.f;
    float dst_scale = 4.f;

    /* create memory descriptors for convolution data w/ no specified format */
    auto conv_src_md = memory::desc(
            { conv_src_tz }, memory::data_type::u8, memory::format::any);
    auto conv_bias_md = memory::desc(
            { conv_bias_tz }, memory::data_type::s32, memory::format::any);
    auto conv_weights_md = memory::desc(
            { conv_weights_tz }, memory::data_type::s8, memory::format::any);
    auto conv_dst_md = memory::desc(
            { conv_dst_tz }, memory::data_type::u8, memory::format::any);

    auto conv_desc = convolution_forward::desc(prop_kind::forward,
                                               convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
                                               conv_dst_md, conv_strides, padding, padding,
                                               padding_kind::zero);


    primitive_attr conv_attr;
    const float ops_scale = 1.f;
    const float ops_alpha = 0.f; // relu negative slope
    const float ops_beta = 0.f;
    post_ops ops;
    ops.append_eltwise(ops_scale, algorithm::eltwise_relu, ops_alpha, ops_beta);
    conv_attr.set_post_ops(ops);

    auto conv_prim_desc = convolution_forward::primitive_desc(
            conv_desc, conv_attr, cpu_engine);

    memory::primitive_desc tmp = conv_prim_desc.src_primitive_desc();
    auto conv_src_memory = dataPipelineManager->allocate_src(tmp, src_scale);
    tmp = conv_prim_desc.dst_primitive_desc();
    auto conv_dst_memory = dataPipelineManager->allocate_dst(tmp, dst_scale);
    tmp = conv_prim_desc.weights_primitive_desc();
    auto conv_weights_memory = parametersManager->allocate_parameters(tmp, weights_scale, conv_user_weights_memory);
    tmp = conv_prim_desc.bias_primitive_desc();
    auto conv_bias_memory = parametersManager->allocate_parameters(tmp, bias_scale, conv_user_bias_memory);
    /* create convolution primitive and add it to inference_ops */
    inference_ops.push_back( convolution_forward(conv_prim_desc, *conv_src_memory->memref,
                                      *conv_weights_memory->memref, *conv_bias_memory->memref, *conv_dst_memory->memref));

    last_output_shape = conv_dst_tz;
    last_output = conv_dst_memory;
}

void INTNetwork::createPool2D(const memory::dims& pool_dst_tz, const memory::dims& pool_kernel, const memory::dims& pool_strides,
                             const memory::dims& pool_padding, algorithm pool_algorithm) {

    auto pool1_dst_md = memory::desc({ pool_dst_tz }, memory::data_type::u8, memory::format::any);

    /* create a pooling */
    auto pool1_desc = pooling_forward::desc(prop_kind::forward_inference,
                                            pool_algorithm, last_output->memref->get_primitive_desc().desc(),
                                            pool1_dst_md, pool_strides, pool_kernel, pool_padding,
                                            pool_padding, padding_kind::zero);
    auto pool1_pd = pooling_forward::primitive_desc(pool1_desc, cpu_engine);
    auto tmp = pool1_pd.dst_primitive_desc();
    auto pool_dst_memory = dataPipelineManager->allocate_dst(tmp, last_output->scale);

    inference_ops.push_back(
            pooling_forward(pool1_pd, *last_output->memref, *pool_dst_memory->memref));

    last_output = pool_dst_memory;
    last_output_shape = pool_dst_tz;
}

void INTNetwork::createFC(const memory::dims& fc_dst_tz, const memory::dims& fc_weights_tz, const memory::dims& fc_bias_tz,
                          const memory::dims& fc_src_tz, membase* user_weights, membase* user_bias) {

    auto fc_bias_md = memory::desc({ fc_bias_tz }, memory::data_type::f32, memory::format::any);
    auto fc_weights_md = memory::desc({ fc_weights_tz }, memory::data_type::f32, memory::format::any);
    auto fc_dst_md = memory::desc({ fc_dst_tz }, memory::data_type::f32, memory::format::any);
    auto fc_src_md = memory::desc({fc_src_tz}, memory::data_type::f32, memory::format::any);

    auto fc_desc = inner_product_forward::desc(prop_kind::forward_inference, fc_src_md, fc_weights_md, fc_bias_md, fc_dst_md);
    auto fc_prim_desc = inner_product_forward::primitive_desc(fc_desc, cpu_engine);
    auto tmp = fc_prim_desc.weights_primitive_desc();
    auto fc_weights_memory = parametersManager->allocate_parameters(tmp, user_weights);
    tmp = fc_prim_desc.src_primitive_desc();
    auto fc_src_memory = dataPipelineManager->allocate_src(tmp);
    tmp = fc_prim_desc.bias_primitive_desc();
    auto fc_bias_memory = parametersManager->allocate_parameters(tmp, user_bias);
    tmp = fc_prim_desc.dst_primitive_desc();
    auto fc_dst_memory = dataPipelineManager->allocate_dst(tmp);
    inference_ops.push_back(inner_product_forward(fc_prim_desc, *fc_src_memory->memref,
                                                  *fc_weights_memory->memref, *fc_bias_memory->memref, *fc_dst_memory->memref));
    last_output = fc_dst_memory;
    last_output_shape = fc_dst_tz;
}
