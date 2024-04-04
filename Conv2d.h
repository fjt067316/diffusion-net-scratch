// #include "../utils/Layer.cpp"
#include "../utils/Tensor.cpp"

#pragma once 

class Conv2d // public Layer
{
public:
    int input_channels;
    int output_channels;
    int filter_size;
    Tensor<float, 4> weights;
    Tensor<float, 1> bias;

    Conv2d(int input_channels, int output_channels, int filter_size) :
    input_channels(input_channels),
    output_channels(output_channels),
    filter_size(filter_size),
    weights({output_channels, input_channels, filter_size, filter_size}),
    bias({output_channels})
    // Layer("conv")
    {}

    Tensor<float, 4> forward(Tensor<float,4> &input);
    Tensor<float, 4> backward(Tensor<float,4> &dLdZ);

};