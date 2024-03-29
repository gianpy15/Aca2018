//
// Created by luca on 24/05/18.
//

#ifndef ACA2018_INTNETWORK_H
#define ACA2018_INTNETWORK_H

#include "AbsNet.h"


class INTNetwork: public AbsNet {
public:
    explicit INTNetwork(const memory::dims &input_size);
    static AbsNet *createNet(const memory::dims &input_size);

protected:
    void createPool2D(const memory::dims& pool_out_shape, const memory::dims& pool_kernel, const memory::dims& pool_strides,
                      const memory::dims& pool_padding, algorithm pool_algorithm) override;

private:
    void createConv2D(const memory::dims& conv_src_tz,
                      const memory::dims& conv_weights_tz,
                      const memory::dims& conv_bias_tz,
                      const memory::dims& conv_strides,
                      const memory::dims& conv_dst_tz,
                      const memory::dims& padding,
                      membase* conv_user_weights_memory,
                      membase* conv_user_bias_memory) override;
};


#endif //ACA2018_INTNETWORK_H
