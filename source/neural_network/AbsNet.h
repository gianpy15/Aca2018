//
// Created by gianpaolo on 5/23/18.
//

#ifndef ACA2018_ABSNET_H
#define ACA2018_ABSNET_H

#include <string>
#include "mkldnn.hpp"

using namespace mkldnn;

/**
 * enumeration used for identifying different types of padding
 * for convolutions (one of valid and same).
 * SAME: will apply padding in order to keep the same dimension of the output
 *       in case of stride [1, 1]
 * VALID: will not apply any zero padding to the input
 */
enum Padding { SAME = 's', VALID = 'v' };


class AbsNet {
public:
    AbsNet(const memory::dims &input_size);

    virtual AbsNet * addConv2D(int channels_out, const int *kernel_size, const int *strides, Padding padding)= 0;
    void run_net();
    void run_net(int times);
    void setup_net();

protected:
    memory::dims input_tz;
    std::vector<primitive> net;
    std::vector<primitive> net_weights;
    memory * last_output = new memory(mkldnn::primitive());
    memory::dims last_output_shape;
    engine cpu_engine = engine(engine::cpu, 0);

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
    virtual void createConv2D(memory::dims conv_src_tz,
                              memory::dims conv_weights_tz,
                              memory::dims conv_bias_tz,
                              memory::dims conv_strides,
                              memory::dims conv_dst_tz,
                              memory::dims padding)= 0;
};


#endif //ACA2018_ABSNET_H
