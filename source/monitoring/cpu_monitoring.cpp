//
// Created by luca on 05/06/18.
//

#include "cpu_monitoring.h"
#include <iostream>
#include <fstream>
#include <cstring>

int cpu_cores(){
    char line[16];
    if(system("nproc >/tmp/nprocout") != 0)
        return -1;
    std::ifstream nprocout("/tmp/nprocout");
    nprocout >> line;
    return atoi(line);
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