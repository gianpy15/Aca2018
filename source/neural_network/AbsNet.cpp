//
// Created by gianpaolo on 5/23/18.
//

#include "AbsNet.h"

void AbsNet::run_net() {
    stream(stream::kind::eager).submit(net).wait();
}

void AbsNet::setup_net() {
    stream(stream::kind::eager).submit(net_weights).wait();
}

AbsNet::AbsNet(const memory::dims &input_size): input_tz(input_size) {
    input_tz = input_size;
}