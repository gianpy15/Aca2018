//
// Created by gianpaolo on 5/23/18.
//
#include <mkldnn_types.h>
#include "AbsNet.h"
#include "../io/h5io.h"

void AbsNet::run_net(int times) {
    if (!inference_ops.empty()) {
        try {
            auto start_time = std::chrono::high_resolution_clock::now();
            for (int i=0; i<times; i++) {
                stream(stream::kind::eager).submit(inference_ops).wait();
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
    if (!setup_ops.empty()) {
        try {
            stream(stream::kind::eager).submit(setup_ops).wait();
        } catch (error &e) {
            std::cerr << "status: " << e.status << std::endl;
            std::cerr << "message: " << e.message << std::endl;
            throw;
        }
    }

    parametersManager->setup_done();

}

AbsNet::AbsNet(const std::string filename){
    dataPipelineManager = new DataPipelineManager(inference_ops);
    parametersManager = new ParametersManager(setup_ops);

    H5io parser(filename);

    memory::dims tmp_dims {0, 0, 0, 0};
    memory::dims tmp_wdims {0, 0, 0, 0};
    memory::dims tmp_bdims {0, 0, 0, 0};

    LayerDescriptor* tmp_layer;
    int layers_num = 0;
    int i, j;
    int ksize[] {0, 0};
    int default_strides[] {1, 1};
    memory::primitive_desc tmp_memdesc;
    membase * tmp_weights;
    membase * tmp_biases;

    while(parser.has_next()){
        tmp_layer = parser.get_next();
        switch(tmp_layer->layerType){
            case LayerType::INPUT:
                if (layers_num){
                    std::cerr << "PARSING ERROR: INPUT SPECIFICATION NOT FIRST" << std::endl;
                    exit(-1);
                }
                for (i=0; i<4; i++)
                    tmp_dims[i] = (int)tmp_layer->weightsDimensions[i];
                last_output_shape = input_tz;
                tmp_memdesc ={ { { tmp_dims }, memory::data_type::f32, memory::format::nchw }, cpu_engine};
                last_output = dataPipelineManager->allocate_src(tmp_memdesc);
                layers_num++;
                break;
            case LayerType::CONV:
                for (i=0; i<4; i++)
                    tmp_wdims[i] = (int)tmp_layer->weightsDimensions[i];
                for (i=0; i<1; i++)
                    tmp_bdims[i] = (int)tmp_layer->biasesDimensions[i];

                tmp_weights = new membase(tmp_wdims, memory::format::hwio, memory::data_type::f32, tmp_layer->weights);
                tmp_biases = new membase(tmp_wdims, memory::format::x, memory::data_type::f32, tmp_layer->biases);

                ksize[0] = tmp_wdims[0];
                ksize[1] = tmp_wdims[1];

                addConv2D(tmp_wdims[3], ksize, default_strides, Padding::SAME, tmp_weights, tmp_biases);
                layers_num++;
                break;
            case LayerType::DENSE:
                for (i=0; i<4; i++)
                    tmp_wdims[i] = (int)tmp_layer->weightsDimensions[i];
                for (i=0; i<1; i++)
                    tmp_bdims[i] = (int)tmp_layer->biasesDimensions[i];

                tmp_weights = new membase(tmp_wdims, memory::format::nc, memory::data_type::f32, tmp_layer->weights);
                tmp_biases = new membase(tmp_wdims, memory::format::x, memory::data_type::f32, tmp_layer->biases);

                // JUST A GUESS: NOT SURE WHETHER FORMAT IS (INPUT, OUTPUT) OR (OUTPUT, INPUT). GUESSED THE SECOND ONE.
                addFC(tmp_wdims[0], tmp_weights, tmp_biases); // TODO: CHECK CORRECT ORDER OF DIMS IN PARSING
                layers_num++;
                break;
            case LayerType::FLATTEN:
                // should be automatic in the reorder operations from conv outputs to dense inputs
                layers_num++;
                break;
            case LayerType::POOL:
                ksize[0] = ksize[1] = 2;
                addPool2D(ksize, Pooling::MAX, Padding::SAME);
                layers_num++;
                break;
        }
    }
}

AbsNet::AbsNet(const memory::dims &input_size): input_tz(input_size) {
    last_output_shape = input_tz;
    dataPipelineManager = new DataPipelineManager(inference_ops);
    parametersManager = new ParametersManager(setup_ops);
    memory::primitive_desc memdesc ={ { { input_size }, memory::data_type::f32, memory::format::nchw }, cpu_engine};
    last_output = dataPipelineManager->allocate_src(memdesc);
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

AbsNet *AbsNet::addConv2D(int channels_out, const int *kernel_size, const int *strides, Padding padding, membase * weights, membase * bias) {
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
        createConv2D(in_shape, conv_weights_tz, conv_bias_tz, conv_strides, out_shape, padding_tz, weights, bias);
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

    acc += parametersManager->memory_usage();
    acc += dataPipelineManager->memory_usage();
    return acc;
}

size_t AbsNet::parameters_memory_usage() {
    return parametersManager->memory_usage();
}

AbsNet::~AbsNet() {
    inference_ops.clear();
    setup_ops.clear();
    delete parametersManager;
    delete dataPipelineManager;
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

AbsNet *AbsNet::addFC(int outputs, membase * weights, membase * bias) {
    int inputs = last_output_shape[1];
    int batch_size = last_output_shape[0];
    memory::dims weights_shape = { inputs, outputs };
    memory::dims biases_shape = { outputs };
    memory::dims output_shape = { batch_size, outputs };

    std::cout << "initialized fc dimensions" << std::endl;

    try {
        createFC(output_shape, weights_shape, biases_shape, weights, bias);
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

    auto conv_weights_memory = new membase(conv_weights_tz, memory::format::oihw, memory::data_type::f32, nullptr, 1.f);
    auto conv_bias_memory = new membase(conv_bias_tz, memory::format::x, memory::data_type::f32, nullptr, 1.f);
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

    auto weights_memory = new membase(fc_weights_tz, memory::format::nc, memory::data_type::f32, nullptr, 1.f);
    auto bias_memory = new membase(fc_bias_tz, memory::format::x, memory::data_type::f32, nullptr, 1.f);
    createFC(fc_dst_tz, fc_weights_tz, fc_bias_tz, weights_memory, bias_memory);
}