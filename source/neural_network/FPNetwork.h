//
// Created by gianpaolo on 5/23/18.
//

#ifndef ACA2018_FPNETWORK_H
#define ACA2018_FPNETWORK_H

#include "AbsNet.h"


class FPNetwork: public AbsNet {
public:
    explicit FPNetwork(const memory::dims &input_size);
    static AbsNet *createNet(const memory::dims &input_size);
private:
    void createConv2D(memory::dims conv_src_tz,
                      memory::dims conv_weights_tz,
                      memory::dims conv_bias_tz,
                      memory::dims conv_strides,
                      memory::dims conv_dst_tz,
                      memory::dims padding,
                      memory* conv_user_weights_memory,
                      memory* conv_user_bias_memory) override;

    void createPool2D(memory::dims pool_dst_tz, memory::dims pool_kernel, memory::dims pool_strides,
                      memory::dims pool_padding, algorithm pool_algorithm) override;

    void createFC(memory::dims fc_dst_tz, memory::dims fc_weights_tz, memory::dims fc_bias_tz,
                  memory* user_weights, memory* user_bias) override;


};


#endif //ACA2018_FPNETWORK_H
