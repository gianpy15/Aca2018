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
    memory::dims tmp_bdims {0};

    LayerDescriptor* tmp_layer;
    int layers_num = 0;
    int i, j;
    int ksize[] {0, 0};
    int default_strides[] {1, 1};
    memory::primitive_desc tmp_memdesc;
    membase * tmp_weights;
    membase * tmp_biases;

    while(parser.has_next()){
        cout << "Layer count pre: " << layers_num << endl;
        tmp_layer = parser.get_next();
        cout << "Layer count post: " << layers_num << endl;
        cout << "Layer type: " << tmp_layer->layerType << endl;
        cout << "Input layer type: " << LayerType::INPUT << endl;

        switch(tmp_layer->layerType){
            case LayerType::INPUT:
                cout << "Input" << endl;
                cout << "Layer count in: " << layers_num << endl;
                if (layers_num){
                    std::cerr << "PARSING ERROR: INPUT SPECIFICATION NOT FIRST" << std::endl;
                    exit(-1);
                }
                /*
                for (i=0; i<4; i++)
                    tmp_dims[i] = (int)tmp_layer->weightsDimensions[i];
                    */
                tmp_dims = {100, 224, 224, 3};
                input_tz = {100, 224, 224, 3};
                delete last_output_shape;
                last_output_shape = new memory::dims(input_tz);
                tmp_memdesc ={ { { tmp_dims }, memory::data_type::f32, memory::format::nhwc }, cpu_engine};
                last_output = dataPipelineManager->allocate_src(tmp_memdesc);
                layers_num++;
                break;
            case LayerType::CONV:
                cout << "conv" << endl;
                /*
                for (i=0; i<4; i++)
                    tmp_wdims[i] = (int) tmp_layer->weightsDimensions[i];
                for (i=0; i<1; i++) {
                    tmp_bdims[i] = (int)tmp_layer->biasesDimensions[i];
                }

                tmp_weights = new membase(tmp_wdims, memory::format::hwio, memory::data_type::f32, tmp_layer->weights);
                tmp_biases = new membase(tmp_bdims, memory::format::x, memory::data_type::f32, tmp_layer->biases);

                ksize[0] = tmp_wdims[0];
                ksize[1] = tmp_wdims[1];
                cout << "Layers count: " << layers_num << endl;

                addConv2D(tmp_wdims[3], ksize, default_strides, Padding::SAME, tmp_weights, tmp_biases);
                 */
                layers_num++;
                break;
            case LayerType::DENSE:
                cout << "Dense" << endl;
                for (i=0; i<4; i++)
                    tmp_wdims[i] = (int)tmp_layer->weightsDimensions[i];
                for (i=0; i<1; i++)
                    tmp_bdims[i] = (int)tmp_layer->biasesDimensions[i];

                tmp_weights = new membase(tmp_wdims, memory::format::nc, memory::data_type::f32, tmp_layer->weights);
                tmp_biases = new membase(tmp_wdims, memory::format::x, memory::data_type::f32, tmp_layer->biases);

                // JUST A GUESS: NOT SURE WHETHER FORMAT IS (INPUT, OUTPUT) OR (OUTPUT, INPUT). GUESSED THE LAST ONE.
                addFC(tmp_wdims[0], tmp_weights, tmp_biases); // TODO: CHECK CORRECT ORDER OF DIMS IN PARSING
                layers_num++;
                break;
            case LayerType::FLATTEN:
                cout << "Flatten" << endl;
                flatten();
                layers_num++;
                break;
            case LayerType::POOL:
                cout << "Pool" << endl;
                ksize[0] = 2;
                ksize[1] = 2;
                addPool2D(ksize, Pooling::MAX, Padding::SAME);
                layers_num++;
                break;
        }
    }
}

AbsNet::AbsNet(const memory::dims &input_size): input_tz(input_size) {
    delete last_output_shape;
    last_output_shape = new memory::dims(input_tz);
    dataPipelineManager = new DataPipelineManager(inference_ops);
    parametersManager = new ParametersManager(setup_ops);
    memory::primitive_desc memdesc ={ { { input_size }, memory::data_type::f32, memory::format::nchw }, cpu_engine};
    last_output = dataPipelineManager->allocate_src(memdesc);
}

AbsNet *AbsNet::addConv2D(int channels_out, const int *kernel_size, const int *strides, Padding padding) {
    memory::dims in_shape = *last_output_shape;

    /**
     * Channels_out: the number of channels outputs by the convolution
     * in_shape[3]: should be the number of channels of the input tensor for channel last configuration
     * kernel_size[0:1]: is the effective dimension of the kernel function
     */

    cout << in_shape[3];

    memory::dims conv_weights_tz = {channels_out, in_shape[3], kernel_size[0], kernel_size[1] };
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
    memory::dims in_shape = *last_output_shape;

    /**
     * Channels_out: the number of channels outputs by the convolution
     * in_shape[3]: should be the number of channels of the input tensor for channel last configuration
     * kernel_size[0:1]: is the effective dimension of the kernel function
     */

    memory::dims conv_weights_tz = {channels_out, in_shape[3], kernel_size[0], kernel_size[1] };
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
    memory::dims in_shape = *last_output_shape;
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
        this->createPool2D(pool_out_shape, pool_kernel, pool_strides, pool_padding, pool_alg);
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
    if (last_output_shape->size() == 4){
        flatten();
    }

    memory::dims biases_shape = { outputs };
    memory::dims output_shape = { (*last_output_shape)[0], outputs };
    memory::dims weights_shape = {outputs, (*last_output_shape)[1]};
    memory::dims source_shape = *last_output_shape;


    try {
        createFC(output_shape, weights_shape, biases_shape, source_shape);
    } catch (error &e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
        throw;
    }

    return this;
}

AbsNet *AbsNet::addFC(int outputs, membase * weights, membase * bias) {
    if (last_output_shape->size() == 4){
        flatten();
    }

    memory::dims biases_shape = { outputs };
    memory::dims output_shape = { (*last_output_shape)[0], outputs };
    memory::dims weights_shape = {outputs, (*last_output_shape)[1]};
    memory::dims source_shape = *last_output_shape;

    try {
        createFC(output_shape, weights_shape, biases_shape, source_shape, weights, bias);
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

void AbsNet::createFC(const memory::dims& fc_dst_tz, const memory::dims& fc_weights_tz, const memory::dims& fc_bias_tz,
                      const memory::dims& fc_src_tz) {

    membase * weights_memory;
    if (fc_weights_tz.size() == 4)
        weights_memory = new membase(fc_weights_tz, memory::format::oihw, memory::data_type::f32, nullptr, 1.f);
    else
        weights_memory = new membase(fc_weights_tz, memory::format::oi, memory::data_type::f32, nullptr, 1.f);
    auto bias_memory = new membase(fc_bias_tz, memory::format::x, memory::data_type::f32, nullptr, 1.f);
    createFC(fc_dst_tz, fc_weights_tz, fc_bias_tz, fc_src_tz, weights_memory, bias_memory);
}

void AbsNet::flatten(){
    int channels = 1;
    for (int i = 1; i<last_output_shape->size(); i++)
        channels *= (*last_output_shape)[i];
    int batch_size = (*last_output_shape)[0];

    last_output = new membase({batch_size, channels}, memory::format::oi,
                              last_output->dtype(), last_output->memref->get_data_handle(), last_output->scale);
    last_output_shape = new memory::dims{batch_size, channels};
    // need just a little hack now...
    dataPipelineManager->last_output = last_output;
}