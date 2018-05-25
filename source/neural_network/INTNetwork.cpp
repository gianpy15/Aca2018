//
// Created by luca on 24/05/18.
//

#include "INTNetwork.h"
#include <numeric>
#include <cmath>
#include <iostream>

INTNetwork::INTNetwork(const memory::dims &input_size) : AbsNet(input_size) {
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

AbsNet * INTNetwork::createNet(const memory::dims &input_size) {
    return new INTNetwork(input_size);
}

void INTNetwork::createConv2D(memory::dims conv_src_tz, memory::dims conv_weights_tz, memory::dims conv_bias_tz,
                              memory::dims conv_strides, memory::dims conv_dst_tz, memory::dims padding) {
    /* Set Scaling mode for int8 quantizing */
    const std::vector<float> src_scales = { 2.0f }; // Qa = 255 / max(source)
    const std::vector<float> weight_scales = { 2.0f }; // Qw = 127 / max(abs(weights))
    const std::vector<float> bias_scales = { 4.0f }; // QaQw
    const std::vector<float> dst_scales = { 4.0f }; // will result in QaQw
    /* assign halves of vector with arbitrary values */
    auto conv_scales = generate_vec({(int)conv_weights_tz[0]});
    std::fill(conv_scales->begin(), conv_scales->end(), 0.3f);

    const int src_mask = 0;
    const int weight_mask = 0;
    const int bias_mask = 0;
    const int dst_mask = 0;
    const int conv_mask = 1;

    /* Allocate and fill buffers for weights and bias */
    auto conv_weights = generate_vec(conv_weights_tz);
    auto conv_bias = generate_vec(conv_bias_tz);

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

    temporary_memobjs.push_back(conv_user_bias_memory);
    temporary_memobjs.push_back(conv_user_weights_memory);


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
    conv_attr.set_int_output_round_mode(round_mode::round_nearest);
    conv_attr.set_output_scales(conv_mask, *conv_scales);

    /* AlexNet: execute ReLU as PostOps */
    const float ops_scale = 1.f;
    const float ops_alpha = 0.f; // relu negative slope
    const float ops_beta = 0.f;
    post_ops ops;
    ops.append_eltwise(ops_scale, algorithm::eltwise_relu, ops_alpha, ops_beta);
    conv_attr.set_post_ops(ops);

    /* check if int8 convolution is supported */
    try {
        auto conv_prim_desc = convolution_forward::primitive_desc(
                conv_desc, conv_attr, cpu_engine);
    } catch (error &e) {
        if (e.status == mkldnn_unimplemented) {
            std::cerr << "AVX512-BW support or Intel(R) MKL dependency is "
                         "required for int8 convolution" << std::endl;
        }
        throw;
    }

    auto conv_prim_desc = convolution_forward::primitive_desc(
            conv_desc, conv_attr, cpu_engine);

    /* Next: create memory primitives for the convolution's input data
     * and use reorder to quantize the values into int8 */
    auto conv_src_memory = new memory(conv_prim_desc.src_primitive_desc());
    primitive_attr src_attr;
    src_attr.set_int_output_round_mode(round_mode::round_nearest);
    src_attr.set_output_scales(src_mask, src_scales);
    auto src_reorder_pd
            = reorder::primitive_desc(last_output->get_primitive_desc(),
                                      conv_src_memory->get_primitive_desc(), src_attr);
    net.push_back(reorder(src_reorder_pd, *last_output, *conv_src_memory));

    auto conv_weights_memory = new memory(conv_prim_desc.weights_primitive_desc());
    primitive_attr weight_attr;
    weight_attr.set_int_output_round_mode(round_mode::round_nearest);
    weight_attr.set_output_scales(weight_mask, weight_scales);
    auto weight_reorder_pd
            = reorder::primitive_desc(conv_user_weights_memory->get_primitive_desc(),
                                      conv_weights_memory->get_primitive_desc(), weight_attr);
    net_weights.push_back(reorder(
            weight_reorder_pd, *conv_user_weights_memory, *conv_weights_memory));

    auto conv_bias_memory = new memory(conv_prim_desc.bias_primitive_desc());
    primitive_attr bias_attr;
    bias_attr.set_int_output_round_mode(round_mode::round_nearest);
    bias_attr.set_output_scales(bias_mask, bias_scales);
    auto bias_reorder_pd
            = reorder::primitive_desc(conv_user_bias_memory->get_primitive_desc(),
                                      conv_bias_memory->get_primitive_desc(), bias_attr);
    net_weights.push_back(reorder(bias_reorder_pd, *conv_user_bias_memory, *conv_bias_memory));

    auto conv_dst_memory = new memory(conv_prim_desc.dst_primitive_desc());

    memobjs.push_back(conv_src_memory);
    memobjs.push_back(conv_weights_memory);
    memobjs.push_back(conv_dst_memory);
    memobjs.push_back(conv_bias_memory);

    /* create convolution primitive and add it to net */
    net.push_back( convolution_forward(conv_prim_desc, *conv_src_memory,
                                      *conv_weights_memory, *conv_bias_memory, *conv_dst_memory));

    /* Convert data back into fp32 and compare values with u8.
     * Note: data is unsigned since there are no negative values
     * after ReLU */

    auto fpoutput = generate_vec(conv_dst_tz);

    /* Create a memory primitive for user data output */
    auto fp_dst_memory = new memory(
            { { { conv_dst_tz }, memory::data_type::f32, memory::format::nchw },
              cpu_engine },
            fpoutput->data());

    memobjs.push_back(fp_dst_memory);
    primitive_attr dst_attr;
    dst_attr.set_int_output_round_mode(round_mode::round_nearest);
    dst_attr.set_output_scales(dst_mask, dst_scales);
    auto dst_reorder_pd
            = reorder::primitive_desc(conv_dst_memory->get_primitive_desc(),
                                      fp_dst_memory->get_primitive_desc(), dst_attr);

    /* Convert the destination memory from convolution into user
     * data format */
    net.push_back(reorder(dst_reorder_pd, *conv_dst_memory, *fp_dst_memory));

    last_output = fp_dst_memory;
    last_output_shape = conv_dst_tz;
}

void INTNetwork::createPool2D(memory::dims pool_dst_tz, memory::dims pool_kernel, memory::dims pool_strides,
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