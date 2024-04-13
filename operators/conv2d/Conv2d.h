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
    Tensor<float, 1> bias;

    int padding;
    int stride;
    bool use_bias;
    bool use_relu;

    Conv2d(int input_channels, int output_channels, int filter_size, int padding=0, int stride=1, bool use_bias=true, bool use_relu=false) :
    input_channels(input_channels),
    output_channels(output_channels),
    filter_size(filter_size),
    weights({output_channels, input_channels, filter_size, filter_size}, true, true),
    padding(padding),
    stride(stride),
    use_relu(use_relu),
    use_bias(use_bias)
    {
        if (use_bias){
            bias = Tensor<float, 1>({output_size}, true, true);
        }
    }

    Tensor<float, 4> forward(Tensor<float,4> &input);
    Tensor<float, 4> backward(Tensor<float,4> &dLdZ);

};