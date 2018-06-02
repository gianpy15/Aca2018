//
// Created by gianpaolo on 29/05/18.
//

#include "gtest/gtest.h"
#include "h5io.h"
#include "DataSetIO.h"

namespace IOTest{

    TEST(H5ioTest, opening_files) {
        H5io *test = nullptr;
        ASSERT_NO_FATAL_FAILURE(test = new H5io("vgg"));
        EXPECT_TRUE(test != nullptr);
        ASSERT_NO_FATAL_FAILURE(delete test);
    }

    TEST(H5ioTest, reading_layers_correctly) {
        H5io *net = new H5io("vgg");
        while(net->has_next())
            EXPECT_TRUE(net->get_next() != nullptr);
        delete net;
    }

    TEST(H5ioTest, check_type_consistency) {
        H5io *net = new H5io("vgg");
        while(net->has_next()) {
            LayerDescriptor *layerDescriptor = net->get_next();
            EXPECT_TRUE(layerDescriptor != nullptr);
            switch (layerDescriptor->layerType) {
                case CONV:
                case DENSE:
                    EXPECT_TRUE(layerDescriptor->weightsDimensions);
                    EXPECT_TRUE(layerDescriptor->biasesDimensions);
                    EXPECT_TRUE(layerDescriptor->weights);
                    EXPECT_TRUE(layerDescriptor->biases);
                    EXPECT_TRUE(layerDescriptor->poolSize == nullptr);
                    break;
                case POOL:
                    EXPECT_TRUE(layerDescriptor->weightsDimensions == nullptr);
                    EXPECT_TRUE(layerDescriptor->biasesDimensions == nullptr);
                    EXPECT_TRUE(layerDescriptor->weights == nullptr);
                    EXPECT_TRUE(layerDescriptor->biases == nullptr);
                    EXPECT_TRUE(layerDescriptor->poolSize);
                    break;
                case INPUT:
                case FLATTEN:
                    EXPECT_TRUE(layerDescriptor->weightsDimensions == nullptr);
                    EXPECT_TRUE(layerDescriptor->biasesDimensions == nullptr);
                    EXPECT_TRUE(layerDescriptor->weights == nullptr);
                    EXPECT_TRUE(layerDescriptor->biases == nullptr);
                    EXPECT_TRUE(layerDescriptor->poolSize == nullptr);
                    break;
            }
            delete layerDescriptor;
        }
    }

    TEST(DataSetIO, file_opening) {
        ASSERT_NO_THROW(DataSetIO dataSet("images"));
    }

    TEST(DataSetIO, file_exploring) {
        auto dataSet = new DataSetIO("images");
        int *labels;
        int *labels_shape;
        float *images;
        int *images_shape;
        images = dataSet->get_images();
        EXPECT_TRUE(images != nullptr);
        images_shape = dataSet->get_images_shape();
        EXPECT_TRUE(images_shape != nullptr);

        int dim = 1;
        for(int i = 0; i < 4; i++)
            dim *= images_shape[i];
        for(int i = 0; i < dim; i++)
            EXPECT_GE(images[i], 0);

        labels = dataSet->get_labels();
        EXPECT_TRUE(labels != nullptr);
        labels_shape = dataSet->get_labels_shape();
        EXPECT_TRUE(labels_shape != nullptr);

        dim = 1;
        for(int i = 0; i < 1; i++)
            dim *= labels_shape[i];
        for(int i = 0; i < dim; i++)
            EXPECT_GT(labels[i], 0);

        delete dataSet;
    }

}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}