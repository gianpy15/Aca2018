//
// Created by luca on 24/05/18.
//

#ifndef ACA2018_INTNETWORK_H
#define ACA2018_INTNETWORK_H

#include "AbsNet.h"


class INTNetwork: public AbsNet {
public:
    explicit INTNetwork(const memory::dims &input_size);
    AbsNet *addPool2D(const int *kernel_size, Pooling pooling_algorithm, Padding padding);
    static AbsNet *createNet(const memory::dims &input_size);

protected:
    void createPool2D(memory::dims pool_out_shape, memory::dims pool_kernel, memory::dims pool_strides,
                      memory::dims pool_padding, algorithm pool_algorithm) override;

private:
    void createConv2D(memory::dims conv_src_tz,
                      memory::dims conv_weights_tz,
                      memory::dims conv_bias_tz,
                      memory::dims conv_strides,
                      memory::dims conv_dst_tz,
                      memory::dims padding) override;

protected:
    void createFC(memory::dims fc_dst_tz, memory::dims fc_weights_tz, memory::dims fc_bias_tz) override;
};


#endif //ACA2018_INTNETWORK_H
