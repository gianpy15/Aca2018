//
// Created by luca on 01/06/18.
//

#include "logging.h"
#include <iostream>

bool shallLog = true;

void log(std::string str){
    if (!shallLog)
        return;
    std::cout << str << std::endl;
}

void log(mkldnn::memory::dims d, int size){
    if (!shallLog)
        return;
    std::cout << "mkldnn::memory::dims{";
    for (int i=0; i<size-1; i++){
        std::cout << d[i] << ", ";
    }
    std::cout << d[size-1] << "}" << std::endl;
}