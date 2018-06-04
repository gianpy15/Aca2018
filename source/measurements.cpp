//
// Created by luca on 05/06/18.
//

#include "measurements.h"
#include "monitoring/mem_monitoring.h"
#include <ctime>

void measureAndLog(Logger& logger, AbsNet* net){
    size_t initialProcMem = getCurrentMemUsage() / 1000;
    size_t initialParamsMem = net->parameters_memory_usage() / 1000000;
    size_t initialNetMem = net->total_memory_usage()/ 1000000;
    double timeToSetup = net->setup_net();
    size_t prerunProcMem = getCurrentMemUsage() / 1000;
    size_t prerunParamsMem = net->parameters_memory_usage()/ 1000000;
    size_t prerunNetMem = net->total_memory_usage()/ 1000000;
    double timeToRun = net->run_net();
    size_t postrunProcMem = getCurrentMemUsage() / 1000;
    size_t postrunParamsMem = net->parameters_memory_usage()/ 1000000;
    size_t postrunNetMem = net->total_memory_usage()/ 1000000;

    std::cout << "Initial Mem" << std::endl;
    std::cout << "Net\tParams\tProc" << std::endl;
    std::cout << initialNetMem << "\t" << initialParamsMem << "\t" << initialProcMem << std::endl;
    std::cout << "Time to setup" << std::endl;
    std::cout << timeToSetup << std::endl;
    std::cout << "Prerun Mem" << std::endl;
    std::cout << "Net\tParams\tProc" << std::endl;
    std::cout << prerunNetMem << "\t" << prerunParamsMem << "\t" << prerunProcMem << std::endl;
    std::cout << "Time to run" << std::endl;
    std::cout << timeToRun << std::endl;
    std::cout << "Postrun Mem" << std::endl;
    std::cout << "Net\tParams\tProc" << std::endl;
    std::cout << postrunNetMem << "\t" << postrunParamsMem << "\t" << postrunProcMem << std::endl;

    logger.logValue(timeToRun);
    logger.logValue(timeToSetup);
    logger.logValue(prerunParamsMem);
    logger.logValue(prerunNetMem - prerunParamsMem);
    logger.logValue(prerunProcMem - prerunNetMem);
    logger.endLine();
}