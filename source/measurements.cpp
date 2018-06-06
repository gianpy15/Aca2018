//
// Created by luca on 05/06/18.
//

#include "measurements.h"
#include "monitoring/mem_monitoring.h"
#include "monitoring/cpu_monitoring.h"
#include "neural_network/FPNetwork.h"
#include "neural_network/INTNetwork.h"
#include "neural_network/AbsNet.h"
#include "io/h5io.h"
#include "io/DataSetIO.h"
#include <ctime>


void benchMachine(int maxconv, int maxdense, int maxbsize, int bsizestep, int maxchannels, int chstep) {
    std::cout << "CPU: " << cpu_type() << std::endl;
    std::cout << "cores: " << cpu_cores() << std::endl;

    auto cpuname = cpu_type();
    int cores = cpu_cores();

    int kernel_3x3[] = {3, 3};
    int nostrides[] = {1, 1};
    int pool_kernel[] = {2, 2};

    int convnum, densenum, batchsize, chnum, i;
    Logger logger("log");

    for (convnum = 1; convnum <= maxconv; convnum++){
        for (chnum = chstep; chnum <= maxchannels; chnum += chstep)
            for (densenum = 0; densenum <= maxdense; densenum++) {
                for (batchsize = bsizestep; batchsize <= maxbsize; batchsize += bsizestep){
                    if (chnum == 12 and convnum > 1)
                        break;
                    std::cout << "Running test of quantized net with "
                              << convnum << " convs, "
                              << densenum << " fcs, "
                              << batchsize << " batchsize" << std::endl;
                    auto net = new INTNetwork({batchsize, 3, 230, 230});
                    for (i = 0; i<convnum; i++){
                        //if (i % 3 == 2)
                        //    net->addPool2D(pool_kernel, Pooling::MAX, Padding::SAME);
                        net->addConv2D(chnum, kernel_3x3, nostrides, Padding::SAME);
                    }
                    if (densenum > 0) {
                        //net->addPool2D(pool_kernel, Pooling::MAX, Padding::SAME);
                        net->flatten();
                    }
                    for (i = 0; i<densenum; i++){
                        net->addFC(32);
                        net->addRelu();
                    }

                    logger.logValue(cpuname);
                    logger.logValue(cores);
                    logger.logValue("int8");
                    logger.logValue(convnum);
                    logger.logValue(chnum);
                    logger.logValue(batchsize);
                    measureAndLog(logger, net);
                    delete net;
                }
            }
    }

    for (convnum = 1; convnum <= maxconv; convnum++){
        for (chnum = chstep; chnum <= maxchannels; chnum += chstep)
            for (densenum = 0; densenum <= maxdense; densenum++) {
                for (batchsize = bsizestep; batchsize <= maxbsize; batchsize += bsizestep){
                    std::cout << "Running test of fp32 net with "
                              << convnum << " convs, "
                              << densenum << " fcs, "
                              << batchsize << " batchsize" << std::endl;
                    auto net = new FPNetwork({batchsize, 3, 230, 230});
                    for (i = 0; i<convnum; i++){
                        //if (i % 3 == 2)
                        //    net->addPool2D(pool_kernel, Pooling::MAX, Padding::SAME);
                        net->addConv2D(chnum, kernel_3x3, nostrides, Padding::SAME);
                    }
                    if (densenum > 0) {
                        //net->addPool2D(pool_kernel, Pooling::MAX, Padding::SAME);
                        net->flatten();
                    }
                    for (i = 0; i<densenum; i++){
                        net->addFC(32);
                        net->addRelu();
                    }

                    logger.logValue(cpuname);
                    logger.logValue(cores);
                    logger.logValue("fp32");
                    logger.logValue(convnum);
                    logger.logValue(chnum);
                    logger.logValue(batchsize);
                    measureAndLog(logger, net);
                    delete net;
                }
            }
    }
}

void measureAndLog(Logger& logger, AbsNet* net){
    size_t initialParamsMem = net->parameters_memory_usage() / 1000;
    size_t initialNetMem = net->total_memory_usage()/ 1000;
    double timeToSetup = net->setup_net();
    size_t prerunParamsMem = net->parameters_memory_usage()/ 1000;
    size_t prerunNetMem = net->total_memory_usage()/ 1000;
    double timeToRun = net->run_net();
    size_t postrunParamsMem = net->parameters_memory_usage()/ 1000;
    size_t postrunNetMem = net->total_memory_usage()/ 1000;

    logger.logValue(timeToRun);
    logger.logValue(timeToSetup);
    logger.logValue(prerunParamsMem);
    logger.logValue(prerunNetMem - prerunParamsMem);
    logger.endLine();
}

void runVGG16s(){
    DataSetIO dataset("images");

    auto input_images = dataset.get_images();
    auto in_shape = dataset.get_images_shape();

    auto net_in_shape = {(int)in_shape[0], (int)in_shape[3], (int)in_shape[1], (int)in_shape[2]};

    auto separator = " ######################## ";
    std::cout << separator << "RUNNING VGG16 FP32 VERSION" << separator << std::endl;
    FPNetwork *vgg16f32 = new FPNetwork(net_in_shape);
    vgg16f32->fromFile("vgg");
    std::cout << separator << "VGG16 FP32 VERSION NETWORK LOEADED" << separator << std::endl;
    vgg16f32->set_input_data(input_images);
    std::cout << separator << "VGG16 FP32 VERSION INPUT LOADED" << separator << std::endl;
    vgg16f32->setup_net();
    std::cout << separator << "VGG16 FP32 VERSION SETUP DONE" << separator << std::endl;
    vgg16f32->run_net();
    std::cout << separator << "VGG16 FP32 VERSION RUN COMPLETED" << separator << std::endl;

    delete vgg16f32;

    std::cout << separator << "RUNNING VGG16 QUANTIZED VERSION" << separator << std::endl;
    INTNetwork *vgg16int = new INTNetwork(net_in_shape);
    vgg16int->fromFile("vgg");
    std::cout << separator << "VGG16 QUANTIZED VERSION NETWORK LOEADED" << separator << std::endl;
    vgg16int->set_input_data(input_images);
    std::cout << separator << "VGG16 QUANTIZED VERSION INPUT LOADED" << separator << std::endl;
    vgg16int->setup_net();
    std::cout << separator << "VGG16 QUANTIZED VERSION SETUP DONE" << separator << std::endl;
    vgg16int->run_net();
    std::cout << separator << "VGG16 QUANTIZED VERSION RUN COMPLETED" << separator << std::endl;
}