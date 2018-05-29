//
// Created by gianpaolo on 5/26/18.
//

#include "h5io.h"

H5io::H5io(string file_name) : H5_NAME(file_name) {
    string path = "../resources/" + file_name + ".h5";
    this->name = file_name;
    this->TXT_NAME = "../resources/" + file_name + ".txt";
    this->H5_NAME = H5std_string(path);
    this->ROOT = H5std_string("root");
    this->net = new H5File(this->H5_NAME, H5F_ACC_RDONLY);

    /// Creating array with all layers name
    string line;
    ifstream layersFile(this->TXT_NAME);
    if (layersFile.is_open()) {
        while (getline(layersFile, line))
            this->layers.push_back(line);
        layersFile.close();
    }

    this->layers_iter = this->layers.begin();
}

LayerInfo *H5io::get_layer(string layer_name, LayerType layer_type) {
    const H5std_string &layerName(layer_name);
    const H5std_string weightsStr("weights");
    const H5std_string biasesStr("biases");

    auto *layerDescriptor = new LayerInfo;

    switch (layer_type) {
        case INPUT:
        case FLATTEN:
        case POOL:
            layerDescriptor->layerType = layer_type;
            return layerDescriptor;
        default:
            break;
    }

    Group root(net->openGroup(this->ROOT));
    Group layer(root.openGroup(layerName));

    DataSet weights(layer.openDataSet(weightsStr));
    DataSet biases(layer.openDataSet(biasesStr));

    DataSpace weightsDataSpace(weights.getSpace());
    DataSpace biasesDataSpace(biases.getSpace());

    int weightsRank = weightsDataSpace.getSimpleExtentNdims();
    int biasesRank = biasesDataSpace.getSimpleExtentNdims();

    auto weightsDims = new hsize_t[weightsRank];
    auto biasesDims = new hsize_t[biasesRank];

    weightsRank = weightsDataSpace.getSimpleExtentDims(weightsDims);
    biasesRank = biasesDataSpace.getSimpleExtentDims(biasesDims);

    int weightsFullDimension = 1;
    int biasesFullDimension = 1;

    for (int i = 0; i < weightsRank; i++)
        weightsFullDimension *= weightsDims[i];

    for (int i = 0; i < biasesRank; i++)
        biasesFullDimension *= biasesDims[i];

    auto weightsVector = new float[weightsFullDimension];
    auto biasesVector = new float[biasesFullDimension];

    DataSpace weightsMemory(weightsRank, weightsDims);
    DataSpace biasesMemory(biasesRank, biasesDims);

    weights.read(weightsVector, PredType::NATIVE_FLOAT, weightsMemory, weightsDataSpace);
    biases.read(biasesVector, PredType::NATIVE_FLOAT, biasesMemory, biasesDataSpace);

    layerDescriptor->layerType = layer_type;
    layerDescriptor->weights = weightsVector;
    layerDescriptor->biases = biasesVector;
    layerDescriptor->weightsDimensions = weightsDims;
    layerDescriptor->biasesDimensions = biasesDims;

    return layerDescriptor;
}

LayerInfo *H5io::get_next() {
    string layer_name;
    LayerInfo *layerDescriptor = nullptr;
    if (this->layers_iter < this->layers.end()) {
        layer_name = *(this->layers_iter++);
        unsigned long long *pool_size = nullptr;

        /// Identification of the layer type
        LayerType type;
        if (layer_name.find("input") != std::string::npos)
            type = LayerType::INPUT;
        else if (layer_name.find("conv") != std::string::npos)
            type = LayerType::CONV;
        else if (layer_name.find("pool") != std::string::npos) {
            type = LayerType::POOL;
            pool_size = new unsigned long long[2];
            string size = layer_name.substr(layer_name.find_first_of(".") + 1);
            pool_size[0] = static_cast<unsigned long long int>(stoi(size.substr(0, size.find("x"))));
            pool_size[1] = static_cast<unsigned long long int>(stoi(size.substr(size.find_last_of('x') + 1)));
        }
        else if (layer_name.find("flatten") != std::string::npos)
            type = LayerType::FLATTEN;
        else if (layer_name.find("fc") != std::string::npos)
            type = LayerType::DENSE;
        else if (layer_name.find("predictions") != std::string::npos)
            type = LayerType::DENSE;
        else type = LayerType::INPUT;

        layerDescriptor = this->get_layer(layer_name, type);
        if (pool_size != nullptr)
            layerDescriptor->poolSize = pool_size;
    }

    return layerDescriptor;
}

bool H5io::has_next() {
    return this->layers_iter < this->layers.end();
}

H5io::~H5io() {
    delete this->net;
}

int main() {
    auto test = new H5io("vgg");
    for (int i = 0; i < 40; i++) {
        auto a = test->get_next();
        if (a && a->layerType == POOL)
            cout << a->poolSize[0] << " x " << a->poolSize[1] << endl;
    }
    delete test;
}
