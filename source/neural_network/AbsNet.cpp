//
// Created by gianpaolo on 5/23/18.
//

#include "AbsNet.h"
#include <iostream>

void AbsNet::run_net() {
    if (!net.empty()) {
        try {
            stream(stream::kind::eager).submit(net).wait();
        } catch (error &e) {
            std::cerr << "status: " << e.status << std::endl;
            std::cerr << "message: " << e.message << std::endl;
            throw;
        }
    }
}

void AbsNet::setup_net() {
    if (!net_weights.empty()) {
        try {
            stream(stream::kind::eager).submit(net_weights).wait();
        } catch (error &e) {
            std::cerr << "status: " << e.status << std::endl;
            std::cerr << "message: " << e.message << std::endl;
            throw;
        }
    }
}

AbsNet::AbsNet(const memory::dims &input_size): input_tz(input_size) {

}