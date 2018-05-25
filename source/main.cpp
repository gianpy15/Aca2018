#include <iostream>
#include "neural_network/FPNetwork.h"
#include "neural_network/INTNetwork.h"

const int kernel_5x5[] = {5, 5};
const int kernel_3x3[] = {3, 3};
const int kernel_2x2[] = {2, 2};
const int kernel_11x11[] = {11, 11};
const int no_stride[] = {1, 1};
const int four_stride[] = {4, 4};

AbsNet * test_fpNet(int repetitions) {
    auto *net = new FPNetwork({1, 3, 227, 227});
    std::cout << "Network initialized" << std::endl;
    net->addConv2D(64, kernel_11x11, four_stride, Padding::VALID);
    std::cout << "Added first convolution" << std::endl;
    net->addConv2D(64, kernel_3x3, no_stride, Padding::VALID);
    std::cout << "Added second convolution" << std::endl;
    net->setup_net();
    std::cout << "Network setup successful!" << std::endl;
    net->run_net(repetitions);

    return net;
}

AbsNet * test_intNet(int repetitions) {
    AbsNet *net = new INTNetwork({1, 3, 227, 227});
    std::cout << "Network initialized" << std::endl;
    net->addConv2D(64, kernel_11x11, no_stride, Padding::VALID);
    std::cout << "Added first convolution" << std::endl;
    net->addConv2D(64, kernel_3x3, no_stride, Padding::VALID);
    std::cout << "Added second convolution" << std::endl;
    net->setup_net();
    std::cout << "Network setup successful!" << std::endl;
    net->run_net(repetitions);

    return net;
}

int main() {
    //AbsNet *net = test_fpNet(1);
    AbsNet *net2 = test_intNet(1);
}