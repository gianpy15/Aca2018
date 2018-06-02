//
// Created by luca on 01/06/18.
//

#ifndef ACA2018_LOGGING_H
#define ACA2018_LOGGING_H

#include "mkldnn.hpp"
#include "../mem_management/mem_base.h"

void log(std::string str);
void log(mkldnn::memory::dims d, int size);
void log(long long n);
void log(std::string label, long long n);
void log(membase* mb);
void log(std::vector<int> v);

std::string error_message(int status);

#endif //ACA2018_LOGGING_H
