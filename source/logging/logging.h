//
// Created by luca on 01/06/18.
//

#ifndef ACA2018_LOGGING_H
#define ACA2018_LOGGING_H

#include "mkldnn.hpp"
#include "../mem_management/mem_base.h"

void log(std::string str);
void log(mkldnn::memory::dims d, int size);
void log(int n);
void log(std::string label, int n);
void log(membase* mb);
void log(std::vector<int> v);

#endif //ACA2018_LOGGING_H
