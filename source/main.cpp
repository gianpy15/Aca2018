#include <iostream>
#include "neural_network/FPNetwork.h"
#include "neural_network/INTNetwork.h"

int main() {
    std::cout << "Hello, World!" << std::endl;
    AbsNet *net = new FPNetwork({1, 3, 227, 227});
    std::cout << "Network initialized" << std::endl;
    int kernel_5x5[] = {5, 5};
    int kernel_3x3[] = {3, 3};
    int kernel_11x11[] = {11, 11};
    int no_stride[] = {1, 1};
    int four_stride[] = {4, 4};
    net->addConv2D(96, kernel_11x11, four_stride, Padding::VALID);
    std::cout << "Added first convolution" << std::endl;
    net->addConv2D(50, kernel_3x3, no_stride, Padding::SAME);
    std::cout << "Added second convolution" << std::endl;
    net->setup_net();
    std::cout << "Network setup successful!" << std::endl;
    net->run_net();
    std::cout << "FP Network run successful! Horay!" << std::endl;

    net = new INTNetwork({1, 3, 227, 227});
    std::cout << "Network initialized" << std::endl;
    net->addConv2D(96, kernel_11x11, four_stride, Padding::VALID);
    std::cout << "Added first convolution" << std::endl;
    net->addConv2D(50, kernel_3x3, no_stride, Padding::SAME);
    std::cout << "Added second convolution" << std::endl;
    net->setup_net();
    std::cout << "Network setup successful!" << std::endl;
    net->run_net();
    std::cout << "INT Network run successful! Horay!" << std::endl;
    return 0;
}