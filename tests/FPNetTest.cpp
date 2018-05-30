//
// Created by gianpaolo on 5/25/18.
//

#include "gtest/gtest.h"
#include "AbsNet.h"
#include "FPNetwork.h"

namespace FPTests{
    const memory::dims net_input = {10, 3, 227, 227};
    const int kernel_5x5[] = {5, 5};
    const int kernel_3x3[] = {3, 3};
    const int kernel_2x2[] = {2, 2};
    const int no_stride[] = {1, 1};
    const int two_stride[] = {2, 2};
    const int four_stride[] = {2, 2};

    TEST(FPNetwork, setup) {
        AbsNet *net = new FPNetwork(net_input);
    }

    TEST(FPNetwork_conv, different_kernels) {
        AbsNet *net = new FPNetwork(net_input);
        ASSERT_NO_THROW(net->addConv2D(16, kernel_3x3, no_stride, Padding::SAME));
        ASSERT_NO_THROW(net->addConv2D(16, kernel_5x5, no_stride, Padding::SAME));
    }

    TEST(FPNetwork_conv, different_strides) {
        AbsNet *net = new FPNetwork(net_input);
        ASSERT_NO_THROW(net->addConv2D(16, kernel_3x3, no_stride, Padding::SAME));
        ASSERT_NO_THROW(net->addConv2D(16, kernel_5x5, two_stride, Padding::SAME));
        ASSERT_NO_THROW(net->addConv2D(16, kernel_5x5, four_stride, Padding::SAME));
    }
}

int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}