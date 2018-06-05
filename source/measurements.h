//
// Created by luca on 05/06/18.
//

#ifndef ACA2018_MEASUREMENTS_H
#define ACA2018_MEASUREMENTS_H

#include "logging/logging.h"
#include "neural_network/AbsNet.h"

void benchMachine(int maxconv, int maxdense, int maxbsize, int bsizestep);
void measureAndLog(Logger& logger, AbsNet* net);
#endif //ACA2018_MEASUREMENTS_H
