#include <iostream>
#include "neural_network/FPNetwork.h"

int main() {
    std::cout << "Hello, World!" << std::endl;
    FPNetwork *net = new FPNetwork({1, 3, 220, 220});
    int kernel_5x5[] = {5, 5};
    int kernel_3x3[] = {3, 3};
    int no_stride[] = {1, 1};
    net->addConv2D(20, kernel_5x5, no_stride, Padding::SAME);
    net->addConv2D(50, kernel_3x3, no_stride, Padding::SAME);
    net->setup_net();
    net->run_net();
    return 0;
}