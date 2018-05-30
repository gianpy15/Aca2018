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
    void createConv2D(const memory::dims& conv_src_tz,
                      const memory::dims& conv_weights_tz,
                      const memory::dims& conv_bias_tz,
                      const memory::dims& conv_strides,
                      const memory::dims& conv_dst_tz,
                      const memory::dims& padding,
                      membase* conv_user_weights_memory,
                      membase* conv_user_bias_memory) override;

    void createPool2D(const memory::dims& pool_dst_tz, const memory::dims& pool_kernel, const memory::dims& pool_strides,
                      const memory::dims& pool_padding, algorithm pool_algorithm) override;

    void createFC(const memory::dims& fc_dst_tz, const memory::dims& fc_weights_tz, const memory::dims& fc_bias_tz,
                  const memory::dims& fc_src_tz, membase* user_weights, membase* user_bias) override;


};


#endif //ACA2018_FPNETWORK_H
