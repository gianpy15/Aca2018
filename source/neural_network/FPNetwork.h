//
// Created by gianpaolo on 5/23/18.
//

#ifndef ACA2018_FPNETWORK_H
#define ACA2018_FPNETWORK_H

#include "AbsNet.h"


class FPNetwork: public AbsNet {
public:
    explicit FPNetwork(const memory::dims &input_size);
    AbsNet *addPool2D(const int *kernel_size, Pooling pooling_algorithm, Padding padding);
    static AbsNet *createNet(const memory::dims &input_size);
    void setup_net();
private:
    void createConv2D(memory::dims conv_src_tz,
                      memory::dims conv_weights_tz,
                      memory::dims conv_bias_tz,
                      memory::dims conv_strides,
                      memory::dims conv_dst_tz,
                      memory::dims padding) override;

    void createPool2D(memory::dims pool_dst_tz, memory::dims pool_kernel, memory::dims pool_strides,
                      memory::dims pool_padding, algorithm pool_algorithm) override;

    void createFC(memory::dims fc_dst_tz, memory::dims fc_weights_tz, memory::dims fc_bias_tz) override;


};


#endif //ACA2018_FPNETWORK_H
