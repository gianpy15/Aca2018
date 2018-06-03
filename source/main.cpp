#include <iostream>
#include "neural_network/FPNetwork.h"
#include "neural_network/INTNetwork.h"
#include "monitoring/mem_monitoring.h"
#include "logging/logging.h"

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
    size_t init_mem = getCurrentMemUsage();
    std::cout << "Memory debug msg: " << getCurrentMemUsage()/1000 << "MB" << std::endl;
    std::cout << "Testing FP network" << std::endl;
    auto fpnet = test_net(&FPNetwork::createNet, 1);
    size_t after_fpnet = getCurrentMemUsage();
    std::cout << "Memory for fp32 net (tot): " << fpnet->total_memory_usage()/1000000 << "MB" << std::endl;
    std::cout << "Memory for fp32 net (par): " << fpnet->parameters_memory_usage()/1000 << "KB" << std::endl;
    std::cout << "Memory for fp32 net (pro): " << (after_fpnet-init_mem)/1000 << "MB" << std::endl;
    std::cout << "Memory debug msg: " << getCurrentMemUsage()/1000 << "MB" << std::endl;
    delete fpnet;
    std::cout << "Memory debug msg: " << getCurrentMemUsage()/1000 << "MB" << std::endl;
    after_fpnet = getCurrentMemUsage();

    std::cout << "Testing Int network" << std::endl;
    auto intnet = test_net(&INTNetwork::createNet, 1);
    size_t after_intnet = getCurrentMemUsage();
    std::cout << "Memory debug msg: " << getCurrentMemUsage()/1000 << "MB" << std::endl;

    std::cout << "Memory for int8 net (tot): " << intnet->total_memory_usage()/1000000 << "MB" << std::endl;
    std::cout << "Memory for int8 net (par): " << intnet->parameters_memory_usage()/1000 << "KB" << std::endl;
    std::cout << "Memory for int8 net (pro): " << (after_intnet-after_fpnet)/1000 << "MB" << std::endl;
    std::cout << "Memory debug msg: " << getCurrentMemUsage()/1000 << "MB" << std::endl;

    //AbsNet *net = test_fpNet(1);
   // AbsNet *net2 = test_intNet(1);


    memory::dims input = {1, 3, 227, 227};
    auto net = new FPNetwork(input);
    net->addConv2D(32, kernel_3x3, no_stride, Padding::VALID);
    net->addFC(64);
    net->addFC(16);
    net->setup_net();
    net->run_net();
    */

    AbsNet *n = new INTNetwork({1, 3, 230, 230});
    size_t init_mem = getCurrentMemUsage();
    std::cout << "Net created..." << std::endl;
    n->fromFile("vgg");
    std::cout << "Net loaded..." << std::endl;
    size_t net_loaded_mem = getCurrentMemUsage();
    log("Memory consumption (KB): ", n->total_memory_usage()/1000);
    log("Param memory (KB): ", n->parameters_memory_usage()/1000);
    log("Memory allocation (KB): ", net_loaded_mem-init_mem);
    n->setup_net();
    std::cout << "Net ready..." << std::endl;
    size_t net_setup_mem = getCurrentMemUsage();
    log("Memory consumption (KB): ", n->total_memory_usage()/1000);
    log("Param memory (KB): ", n->parameters_memory_usage()/1000);
    log("Memory allocation (KB): ", net_setup_mem-init_mem);
    n->run_net();
    std::cout << "Net run successfully!" << std::endl;
    size_t net_run_mem = getCurrentMemUsage();
    log("Memory consumption (KB): ", n->total_memory_usage()/1000);
    log("Param memory (KB): ", n->parameters_memory_usage()/1000);
    log("Memory allocation (KB): ", net_run_mem-init_mem);
}