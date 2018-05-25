//
// Created by luca on 24/05/18.
//

#ifndef ACA2018_INTNETWORK_H
#define ACA2018_INTNETWORK_H

#include "AbsNet.h"


class INTNetwork: public AbsNet {
public:
    INTNetwork(const memory::dims &input_size);
    AbsNet * addConv2D(int channels_out, const int *kernel_size, const int *strides, Padding padding) override;

    AbsNet *addPool2D(const int *kernel_size, Pooling pooling_algorithm, Padding padding) override;
    void setup_net();
protected:
    void createPool2D(memory::dims pool_out_shape, memory::dims pool_kernel, memory::dims pool_strides,
                      memory::dims pool_padding, algorithm pool_algorithm) override;

private:
    std::vector<memory*> temporary_memories;
    void createConv2D(memory::dims conv_src_tz,
                      memory::dims conv_weights_tz,
                      memory::dims conv_bias_tz,
                      memory::dims conv_strides,
                      memory::dims conv_dst_tz,
                      memory::dims padding) override;
};


#endif //ACA2018_INTNETWORK_H
