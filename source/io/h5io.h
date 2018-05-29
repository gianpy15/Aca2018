//
// Created by gianpaolo on 5/26/18.
//

#ifndef ACA2018_H5IO_H
#define ACA2018_H5IO_H

#endif //ACA2018_H5IO_H

#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <vector>
#include "H5Cpp.h"

using namespace H5;
using namespace std;

enum LayerType {
    INPUT,
    CONV,
    POOL,
    DENSE,
    FLATTEN
};



struct LayerDescriptor {
    LayerType layerType;
    float *biases = nullptr;
    float *weights = nullptr;
    unsigned long long *weightsDimensions = nullptr;
    unsigned long long *biasesDimensions = nullptr;
    unsigned long long *poolSize = nullptr;

    virtual ~LayerDescriptor();
};

class H5io {
private:
    H5std_string H5_NAME;
    H5std_string ROOT;
    string name;
    string TXT_NAME;
    H5File *net;
    vector<string> layers;
    vector<string>::iterator layers_iter;

    LayerDescriptor *get_layer(string layer_name, LayerType layer_type);

public:
    explicit H5io(string file_name);
    virtual ~H5io();
    LayerDescriptor *get_next();
    bool has_next();

};
