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

private:
    void createConv2D(memory::dims conv_src_tz,
                      memory::dims conv_weights_tz,
                      memory::dims conv_bias_tz,
                      memory::dims conv_strides,
                      memory::dims conv_dst_tz,
                      memory::dims padding) override;
};


#endif //ACA2018_INTNETWORK_H
