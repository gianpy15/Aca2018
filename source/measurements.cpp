//
// Created by luca on 05/06/18.
//

#include "measurements.h"
#include "monitoring/mem_monitoring.h"
#include "monitoring/cpu_monitoring.h"
#include "neural_network/FPNetwork.h"
#include "neural_network/INTNetwork.h"
#include "neural_network/AbsNet.h"
#include <ctime>


void benchMachine(int maxconv, int maxdense, int maxbsize, int bsizestep){
    std::cout << "CPU: " << cpu_type() << std::endl;
    std::cout << "cores: " << cpu_cores() << std::endl;

    auto cpuname = cpu_type();
    int cores = cpu_cores();

    int kernel_3x3[] = {3, 3};
    int nostrides[] = {1, 1};
    int pool_kernel[] = {2, 2};

    int convnum, densenum, batchsize, i;
    Logger logger("log");

    for (convnum = 1; convnum <= maxconv; convnum++){
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
                    net->addConv2D(64, kernel_3x3, nostrides, Padding::SAME);
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
                logger.logValue(densenum);
                logger.logValue(batchsize);
                measureAndLog(logger, net);
                delete net;
            }
        }
    }

    for (convnum = 1; convnum <= maxconv; convnum++){
        for (densenum = 0; densenum <= maxdense; densenum++) {
            for (batchsize = bsizestep; batchsize <= maxbsize; batchsize += bsizestep){
                std::cout << "Running test of quantized net with "
                          << convnum << " convs, "
                          << densenum << " fcs, "
                          << batchsize << " batchsize" << std::endl;
                auto net = new INTNetwork({batchsize, 3, 230, 230});
                for (i = 0; i<convnum; i++){
                    //if (i % 3 == 2)
                    //    net->addPool2D(pool_kernel, Pooling::MAX, Padding::SAME);
                    net->addConv2D(64, kernel_3x3, nostrides, Padding::SAME);
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
                logger.logValue(densenum);
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