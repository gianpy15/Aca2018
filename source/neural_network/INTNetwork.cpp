//
// Created by luca on 24/05/18.
//

#include "INTNetwork.h"
#include <numeric>
#include <cmath>
#include <iostream>

INTNetwork::INTNetwork(const memory::dims &input_size) : AbsNet(input_size) {
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

AbsNet * INTNetwork::addConv2D(int channels_out, const int *kernel_size, const int *strides, Padding padding) {

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

void INTNetwork::createConv2D(memory::dims conv_src_tz,
                             memory::dims conv_weights_tz,
                             memory::dims conv_bias_tz,
                             memory::dims conv_strides,
                             memory::dims conv_dst_tz,
                             memory::dims padding) {
    /* Set Scaling mode for int8 quantizing */
    const std::vector<float> src_scales = { 1.8f };
    const std::vector<float> weight_scales = { 2.0f };
    const std::vector<float> bias_scales = { 1.0f };
    const std::vector<float> dst_scales = { 0.55f };
    /* assign halves of vector with arbitrary values */
    std::vector<float> conv_scales(384);
    const int scales_half = 384 / 2;
    std::fill(conv_scales.begin(), conv_scales.begin() + scales_half, 0.3f);
    std::fill(conv_scales.begin() + scales_half + 1, conv_scales.end(), 0.8f);

    const int src_mask = 0;
    const int weight_mask = 0;
    const int bias_mask = 0;
    const int dst_mask = 0;
    const int conv_mask = 2; // 1 << output_channel_dim

    /* Allocate and fill buffers for weights and bias */
    std::vector<float> conv_weights(std::accumulate(conv_weights_tz.begin(),
                                                    conv_weights_tz.end(), 1, std::multiplies<uint32_t>()));
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

    /* create memory descriptors for convolution data w/ no specified format */
    auto conv_src_md = memory::desc(
            { conv_src_tz }, memory::data_type::u8, memory::format::any);
    auto conv_bias_md = memory::desc(
            { conv_bias_tz }, memory::data_type::s8, memory::format::any);
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
    conv_attr.set_output_scales(conv_mask, conv_scales);

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

    /* create convolution primitive and add it to net */
    net.push_back(convolution_forward(conv_prim_desc, *conv_src_memory,
                                      *conv_weights_memory, *conv_bias_memory, *conv_dst_memory));

    /* Convert data back into fp32 and compare values with u8.
     * Note: data is unsigned since there are no negative values
     * after ReLU */

    std::vector<float> fpoutput(std::accumulate(
            conv_dst_tz.begin(), conv_dst_tz.end(), 1,
            std::multiplies<uint32_t>()));

    /* Create a memory primitive for user data output */
    auto fp_dst_memory = new memory(
            { { { conv_dst_tz }, memory::data_type::f32, memory::format::nchw },
              cpu_engine },
            fpoutput.data());

    primitive_attr dst_attr;
    dst_attr.set_int_output_round_mode(round_mode::round_nearest);
    dst_attr.set_output_scales(dst_mask, dst_scales);
    auto dst_reorder_pd
            = reorder::primitive_desc(conv_dst_memory->get_primitive_desc(),
                                      fp_dst_memory->get_primitive_desc(), dst_attr);

    /* Convert the destination memory from convolution into user
     * data format if necessary */
    if (conv_dst_memory != fp_dst_memory) {
        net.push_back(
                reorder(dst_reorder_pd, *conv_dst_memory, *fp_dst_memory));
    }

    last_output = fp_dst_memory;
    last_output_shape = conv_dst_tz;
}

AbsNet *INTNetwork::addPool2D(const int *kernel_size, const int *strides, Pooling pooling_algorithm, Padding padding) {
    return nullptr;
}

void INTNetwork::createPool2D(memory::dims pool_out_shape, memory::dims pool_kernel, memory::dims pool_strides,
                              memory::dims pool_padding, algorithm pool_algorithm) {

}