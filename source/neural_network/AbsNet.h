//
// Created by gianpaolo on 5/23/18.
//

#ifndef ACA2018_ABSNET_H
#define ACA2018_ABSNET_H

#include <string>
#include <iostream>
#include <chrono>
#include <cmath>

#include "mkldnn.hpp"
#include "../mem_management/mem_management.h"
#include "../mem_management/mem_base.h"

using namespace mkldnn;

/**
 * enumeration used for identifying different types of padding
 * for convolutions (one of valid and same).
 * SAME: will apply padding in order to keep the same dimension of the output
 *       in case of stride [1, 1]
 * VALID: will not apply any zero padding to the input
 */
enum Padding { SAME = 's', VALID = 'v' };
enum Pooling { MAX, AVG };


class AbsNet {
public:
    explicit AbsNet(const memory::dims &input_size);

    AbsNet * addConv2D(int channels_out, const int *kernel_size, const int *strides, Padding padding,
            membase * weights, membase*bias);
    AbsNet * addConv2D(int channels_out, const int *kernel_size, const int *strides, Padding padding);
    AbsNet * addPool2D(const int *kernel_size, Pooling pooling_algorithm, Padding padding);
    AbsNet * addFC(int outputs);
    AbsNet * addFC(int outputs, membase * weights, membase * bias);
    void flatten();
    // virtual static AbsNet *createNet(const memory::dims &input_size)=0;
    void run_net();
    void run_net(int times);
    void setup_net();
    void fromFile(const std::string filename);
    size_t total_memory_usage();
    size_t parameters_memory_usage();
    bool fold_memory = true;
    ~AbsNet();
protected:
    ParametersManager *parametersManager;
    DataPipelineManager *dataPipelineManager;
    memory::dims input_tz;
    std::vector<primitive> inference_ops;
    std::vector<primitive> setup_ops;
    membase * last_output;
    /// Format: { batch, channels, width, height }
    memory::dims last_output_shape;
    engine cpu_engine = get_glob_engine();

    /**
     * This method will be called from addConv2D and adds a convolution layer to the network
     * The bias and output dims are not taken from input, they are inferred from the
     * other parameter
     * @param conv_src_tz is the memory descriptor for the input tensor
     * @param conv_weights_tz  is the memory descriptor for the weights tensor
     * @param conv_bias_tz is the memory descriptor for the bias tensor
     * @param conv_strides is the memory descriptor for the strider
     * @param conv_dst_tz is the memory descriptor for the output tensor
     * @param padding is the padding desired, one of SAME or VALID
     */
    virtual void createConv2D(const memory::dims& conv_src_tz,
                              const memory::dims& conv_weights_tz,
                              const memory::dims& conv_bias_tz,
                              const memory::dims& conv_strides,
                              const memory::dims& conv_dst_tz,
                              const memory::dims& padding,
                              membase* weights,
                              membase* bias)= 0;
    void createConv2D(const memory::dims& conv_src_tz,
                      const memory::dims& conv_weights_tz,
                      const memory::dims& conv_bias_tz,
                      const memory::dims& conv_strides,
                      const memory::dims& conv_dst_tz,
                      const memory::dims& padding);

    virtual void createPool2D(const memory::dims& pool_out_shape,
                              const memory::dims& pool_kernel,
                              const memory::dims& pool_strides,
                              const memory::dims& pool_padding,
                              algorithm pool_algorithm)= 0;
    virtual void createFC(const memory::dims& fc_dst_tz, const memory::dims& fc_weights_tz, const memory::dims& fc_bias_tz,
                          const memory::dims& fc_src_tz, membase * weights, membase * bias)=0;

    void createFC(const memory::dims& fc_dst_tz,
                  const memory::dims& fc_weights_tz,
                  const memory::dims& fc_bias_tz,
                  const memory::dims& fc_src_tz);
};


#endif //ACA2018_ABSNET_H
