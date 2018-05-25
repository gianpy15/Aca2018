#include <iostream>
#include "neural_network/FPNetwork.h"
#include "neural_network/INTNetwork.h"

const int kernel_5x5[] = {5, 5};
const int kernel_3x3[] = {3, 3};
const int kernel_2x2[] = {2, 2};
const int kernel_11x11[] = {11, 11};
const int no_stride[] = {1, 1};
const int four_stride[] = {4, 4};


AbsNet * setup_test_net(AbsNet* baseNet){
    // std::cout << "Network initialized" << std::endl;
    baseNet->addConv2D(128, kernel_3x3, no_stride, Padding::VALID);
    // std::cout << "Added first convolution" << std::endl;
    baseNet->addPool2D(kernel_3x3, Pooling::MAX, Padding::VALID);
    // std::cout << "Added first pooling" << std::endl;
    baseNet->addConv2D(256, kernel_5x5, no_stride, Padding::SAME);
    for (int i=0; i<10; i++)
        baseNet->addConv2D(256, kernel_3x3, no_stride, Padding::SAME);
    // std::cout << "Added second convolution" << std::endl;
    baseNet->addPool2D(kernel_2x2, Pooling::MAX, Padding::SAME);
    // std::cout << "Added second pooling" << std::endl;
    baseNet->setup_net();
    // std::cout << "Network setup successful!" << std::endl;
    return baseNet;
}


AbsNet * test_net(AbsNet* (*initializer)(const memory::dims&), int repetitions) {
    AbsNet *net = initializer({1, 3, 227, 227});
    net = setup_test_net(net);
    net->run_net(repetitions);
    return net;
}

AbsNet * test_fpNet(int repetitions) {
    return test_net(&FPNetwork::createNet, repetitions);
}

AbsNet * test_intNet(int repetitions) {
    return test_net(&INTNetwork::createNet, repetitions);
}

int main() {
    std::cout << "Testing Int network" << std::endl;
    test_net(&INTNetwork::createNet, 1);
    std::cout << "Testing FP network" << std::endl;
    test_net(&FPNetwork::createNet, 1);
    //AbsNet *net = test_fpNet(1);
   // AbsNet *net2 = test_intNet(1);
}