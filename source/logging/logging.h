//
// Created by luca on 01/06/18.
//

#ifndef ACA2018_LOGGING_H
#define ACA2018_LOGGING_H

#include "mkldnn.hpp"

void log(std::string str);
void log(mkldnn::memory::dims d, int size);

#endif //ACA2018_LOGGING_H
