//
// Created by gianpaolo on 02/06/18.
//

#ifndef ACA2018_DATASETIO_H
#define ACA2018_DATASETIO_H

#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <vector>
#include "H5Cpp.h"

using namespace H5;
using namespace std;


class DataSetIO {
private:
    H5std_string H5_NAME;
    const H5std_string ROOT = H5std_string("images");
    const H5std_string IMAGES = H5std_string("images");
    const H5std_string LABELS = H5std_string("labels");
    int *images_shape = nullptr;
    int *labels_shape = nullptr;
    H5File *dataSet = nullptr;

public:
    /**
     * @return the images into a float array dynamically allocated
     */
    float *get_images();
    /**
     * @return the labels into an integer array dynamically allocated
     */
    int *get_labels();
    /**
     * @return the images's shape with format (batch, width, height, channels)
     *         if the method get_images() has not been called yet, this will return nullptr
     */
    int *get_images_shape();
    /**
     * @return the labels's shape with format (batch)
     *         if the method get_labels() has not been called yet, this will return nullptr
     */
    int *get_labels_shape();
    /**
     * Create a new instance for this class
     * @param file_name is the name of the file with the data set in .h5 format, is not necessary
     *                  to specify the extension neither the complete path
     */
    explicit DataSetIO(string file_name);
    virtual ~DataSetIO();
};


#endif //ACA2018_DATASETIO_H
