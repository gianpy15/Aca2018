//
// Created by gianpaolo on 29/05/18.
//

#include "gtest/gtest.h"
#include "h5io.h"

namespace IOTest{

    TEST(IOTest, opening_files) {
        H5io *test = nullptr;
        ASSERT_NO_FATAL_FAILURE(test = new H5io("vgg"));
        EXPECT_TRUE(test != nullptr);
        ASSERT_NO_FATAL_FAILURE(delete test);
    }

    TEST(IOTest, reading_layers_correctly) {
        H5io *net = new H5io("vgg");
        while(net->has_next())
            EXPECT_TRUE(net->get_next() != nullptr);
        delete net;
    }

    TEST(IOTest, check_type_consistency) {
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

}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}