#include <iostream>
#include "neural_network/FPNetwork.h"
#include "neural_network/INTNetwork.h"
#include "monitoring/mem_monitoring.h"
#include "logging/logging.h"
#include "io/DataSetIO.h"
#include "measurements.h"

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
    for (int i=0; i<5; i++)
        baseNet->addConv2D(512, kernel_3x3, no_stride, Padding::SAME);
    // std::cout << "Added second convolution" << std::endl;
    baseNet->addPool2D(kernel_2x2, Pooling::MAX, Padding::SAME);
    // std::cout << "Added second pooling" << std::endl;
    std::cout << "Peak memory usage: " << baseNet->total_memory_usage()/1000000 << "MB" << std::endl;
    std::cout << "Memory debug msg: " << getCurrentMemUsage()/1000 << "MB" << std::endl;
    baseNet->setup_net();
    // std::cout << "Network setup successful!" << std::endl;
    return baseNet;
}


AbsNet * test_net(AbsNet* (*initializer)(const memory::dims&), int repetitions) {
    AbsNet *net = initializer({1, 3, 256, 256});
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
    /*
    DataSetIO dataset("images");
    Logger logger("log");

    auto input_images = dataset.get_images();
    auto output_labels = dataset.get_labels();
    auto out_shape = dataset.get_labels_shape();
    auto in_shape = dataset.get_images_shape();

    auto net_in_shape = {(int)in_shape[0], (int)in_shape[3], (int)in_shape[1], (int)in_shape[2]};

    FPNetwork vgg16(net_in_shape);
    vgg16.fromFile("vgg");
    vgg16.set_input_data(input_images);
    measureAndLog(logger, &vgg16);
    auto output = vgg16.top_n_output(10);

    std::cout << "Results:" << std::endl;
    int i=0;
    for (auto im: output) {
        std::cout << "Im " << i << ": ";
        for (auto pred: im)
            std::cout << pred << " ";
        std::cout << std::endl;
        i++;
    }

    std::cerr << "Images";
    for (i=0; i<5; i++){
        std::cerr << std::endl;
        for(int j=0; j< 20; j++)
            std::cerr << input_images[i*230*230*3 + j] << " ";
    }
    */
    benchMachine(5, 0, 1, 1);
}