#include "../../utils/Tensor.h"

#pragma once 

class Conv2d // public Layer
{
public:
    int input_channels;
    int output_channels;
    int filter_size;
    Tensor<float, 4> weights;
    Tensor<float, 4> input; // input should always be sitting on device as its only use in backprop
    Tensor<float, 4> output;
    Tensor<float, 1> bias;

    int padding;
    int stride;
    bool use_bias;
    bool use_relu;

    Conv2d(int input_channels, int output_channels, int filter_size, int padding=0, int stride=1, bool use_bias=false, bool use_relu=false);

    Tensor<float, 4> forward(Tensor<float,4> &input);
    Tensor<float, 4> backward(Tensor<float,4> &dLdZ);

};