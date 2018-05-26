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
    /* Set Scaling mode for int8 quantizing */
    /*
    const std::vector<float> src_scales = { 2.0f }; // Qa = 255 / max(source)
    const std::vector<float> weight_scales = { 2.0f }; // Qw = 127 / max(abs(weights))
    const std::vector<float> bias_scales = { 4.0f }; // QaQw
    const std::vector<float> dst_scales = { 4.0f }; // will result in QaQw

    const int src_mask = 0;
    const int weight_mask = 0;
    const int bias_mask = 0;
    const int dst_mask = 0;

    temporary_memobjs.push_back(conv_user_bias_memory);
    temporary_memobjs.push_back(conv_user_weights_memory);
    */
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

    /* create a convolution */
    auto conv_desc = convolution_forward::desc(prop_kind::forward,
                                               convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
                                               conv_dst_md, conv_strides, padding, padding,
                                               padding_kind::zero);

    /* define the convolution attributes */
    primitive_attr conv_attr;
    /*
    const int conv_mask = 1;
    auto conv_scales = generate_vec({(int)conv_weights_tz[0]});
    std::fill(conv_scales->begin(), conv_scales->end(), 0.3f);
    conv_attr.set_int_output_round_mode(round_mode::round_nearest);
    conv_attr.set_output_scales(conv_mask, *conv_scales);
    */
    /* AlexNet: execute ReLU as PostOps */
    const float ops_scale = 1.f;
    const float ops_alpha = 0.f; // relu negative slope
    const float ops_beta = 0.f;
    post_ops ops;
    ops.append_eltwise(ops_scale, algorithm::eltwise_relu, ops_alpha, ops_beta);
    conv_attr.set_post_ops(ops);

    auto conv_prim_desc = convolution_forward::primitive_desc(
            conv_desc, conv_attr, cpu_engine);

    /* Next: create memory primitives for the convolution's input data
     * and use reorder to quantize the values into int8 */
    /*
    auto conv_src_memory = new memory(conv_prim_desc.src_primitive_desc());
    auto conv_weights_memory = new memory(conv_prim_desc.weights_primitive_desc());
    auto conv_bias_memory = new memory(conv_prim_desc.bias_primitive_desc());
    auto conv_dst_memory = new memory(conv_prim_desc.dst_primitive_desc());

    make_reorder(inference_ops, last_output, conv_src_memory, src_mask, src_scales);
    make_reorder(setup_ops, conv_user_weights_memory, conv_weights_memory, weight_mask, weight_scales);
    make_reorder(setup_ops, conv_user_bias_memory, conv_bias_memory, bias_mask, bias_scales);

    data_pipeline_memobjs.push_back(conv_src_memory);
    parameters_memobjs.push_back(conv_weights_memory);
    data_pipeline_memobjs.push_back(conv_dst_memory);
    parameters_memobjs.push_back(conv_bias_memory);
    */
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
    /*
    auto fpoutput = generate_vec(conv_dst_tz);

    auto fp_dst_memory = new memory(
            { { { conv_dst_tz }, memory::data_type::f32, memory::format::nchw },
              cpu_engine },
            fpoutput->data());

    data_pipeline_memobjs.push_back(fp_dst_memory);

    make_reorder(inference_ops, conv_dst_memory, fp_dst_memory, dst_mask, dst_scales);
    last_output = fp_dst_memory;
    */
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
    //auto pool_dst_memory = new memory(pool1_pd.dst_primitive_desc());
    //data_pipeline_memobjs.push_back(pool_dst_memory);
    /* create pooling primitive an add it to inference_ops */

    inference_ops.push_back(
            pooling_forward(pool1_pd, *last_output->memref, *pool_dst_memory->memref));

    last_output = pool_dst_memory;
    last_output_shape = pool_dst_tz;
}

void INTNetwork::createFC(const memory::dims& fc_dst_tz, const memory::dims& fc_weights_tz, const memory::dims& fc_bias_tz,
                          membase* user_weights, membase* user_bias) {
    std::cerr << "Function not implemented" << std::endl;
    exit(1);
}
