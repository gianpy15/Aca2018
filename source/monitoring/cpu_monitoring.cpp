//
// Created by luca on 05/06/18.
//

#include "cpu_monitoring.h"
#include <iostream>
#include <fstream>
#include <cstring>

int cpu_cores(){
    size_t result = 0;
    std::ifstream cpuinfo("/proc/cpuinfo");
    char line[512];

    int idx;
    int maxidx = -1;
    if (cpuinfo.is_open()) {
        while (cpuinfo.getline(line, 512)){
            if (std::strncmp(line, "processor", 9) == 0) {
                idx = atoi(line+11);
                if (idx > maxidx)
                    maxidx = idx;
            }
        }
        cpuinfo.close();
    } else {
        std::cerr << "Unable to open /proc/cpuinfo" << std::endl;
        exit(-1);
    }
    return maxidx + 1;
}

std::string cpu_type(){
    size_t result = 0;
    std::ifstream cpuinfo("/proc/cpuinfo");
    char line[512];

    if (cpuinfo.is_open()) {
        while (cpuinfo.getline(line, 512)){
            if (std::strncmp(line, "model name", 10) == 0) {
                return std::string(line+13);
            }
        }
        cpuinfo.close();
    } else {
        std::cerr << "Unable to open /proc/cpuinfo" << std::endl;
        exit(-1);
    }
    return nullptr;
}