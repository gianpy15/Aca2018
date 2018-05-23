//
// Created by gianpaolo on 5/23/18.
//

#ifndef ACA2018_FPNETWORK_H
#define ACA2018_FPNETWORK_H

#include "AbsNet.h"


class FPNetwork: AbsNet {
public:
    FPNetwork(const memory::dims &input_size);
    AbsNet * addConv2D(int channels_out, const int *kernel_size, const int *strides, Padding padding) override;
private:
    void createConv2D(memory::dims conv_src_tz,
                      memory::dims conv_weights_tz,
                      memory::dims conv_bias_tz,
                      memory::dims conv_strides,
                      memory::dims conv_dst_tz,
                      memory::dims padding) override;

};


#endif //ACA2018_FPNETWORK_H
