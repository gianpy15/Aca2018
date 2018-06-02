//
// Created by gianpaolo on 02/06/18.
//

#include "DataSetIO.h"
#include <utility>

DataSetIO::DataSetIO(string file_name) {
    string path = "../resources/" + file_name + ".h5";
    this->H5_NAME = H5std_string(path);
    try {
        this->dataSet = new H5File(this->H5_NAME, H5F_ACC_RDONLY);
    } catch (H5::FileIException) {
        cerr << "unable to open file at " << path << endl;
        exit(-1);
    };
}

int *DataSetIO::get_labels() {
    Group root(dataSet->openGroup(this->ROOT));
    DataSet labels(root.openDataSet(this->LABELS));

    DataSpace labelsDataSpace(labels.getSpace());
    auto rank = labelsDataSpace.getSimpleExtentNdims();
    auto dims = new hsize_t[rank];
    rank = labelsDataSpace.getSimpleExtentDims(dims);
    auto fullDimension = 1;

    for (int i = 0; i < rank; i++)
        fullDimension *= dims[i];

    auto labelsVector = new int[fullDimension];
    DataSpace weightsMemory(rank, dims);
    labels.read(labelsVector, PredType::NATIVE_FLOAT, weightsMemory, labelsDataSpace);
    this->labels_shape = (int *)dims;

    return labelsVector;
}

float *DataSetIO::get_images() {
    Group root(dataSet->openGroup(this->ROOT));
    DataSet images(root.openDataSet(this->IMAGES));

    DataSpace imagesDataSpace(images.getSpace());
    auto rank = imagesDataSpace.getSimpleExtentNdims();
    auto dims = new hsize_t[rank];
    rank = imagesDataSpace.getSimpleExtentDims(dims);
    auto fullDimension = 1;

    for (int i = 0; i < rank; i++)
        fullDimension *= dims[i];

    auto imagesVector = new float[fullDimension];
    DataSpace weightsMemory(rank, dims);
    images.read(imagesVector, PredType::NATIVE_FLOAT, weightsMemory, imagesDataSpace);
    this->images_shape = (int *)dims;

    return imagesVector;
}

int *DataSetIO::get_images_shape() {
    return this->images_shape;
}

int *DataSetIO::get_labels_shape() {
    return this->labels_shape;
}

DataSetIO::~DataSetIO() {
    delete this->labels_shape;
    delete this->images_shape;
    delete this->dataSet;
}