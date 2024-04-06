#include "../../utils/Tensor.h"

#pragma once 

// take one (3,1,1) chunk of input and multiply it by entire (3,3,3) output (filter_size=3)
// must prefill output array with zeros as kernels will only atomic add -> one kernel call for filling with zeros
class ConvTranspose2d 
{
public:
    int input_channels;
    int output_channels;
    int filter_size;
    Tensor<float, 4> weights;
    Tensor<float, 1> bias;

    int padding;
    int stride;

    ConvTranspose2d(int input_channels, int output_channels, int filter_size, int padding=0, int stride=1) :
    input_channels(input_channels),
    output_channels(output_channels),
    filter_size(filter_size),
    weights({output_channels, input_channels, filter_size, filter_size}),
    bias({output_channels}),
    padding(padding),
    stride(stride)
    // Layer("conv")
    {}

    Tensor<float, 4> forward(Tensor<float,4> &input);
    Tensor<float, 4> backward(Tensor<float,4> &dLdZ);

};