#include "../../utils/Tensor.h"

#pragma once 

class Linear // public Layer
{
public:
    int input_size;
    int output_size;
    Tensor<float, 2> input;
    Tensor<float, 2> weights;
    Tensor<float, 2> output;
    Tensor<float, 1> bias;
    Tensor<float, 1> db;
    Tensor<float, 2> dw;

    bool use_bias;
    bool use_relu;

    Linear(int input_size, int output_size, bool use_bias=true, bool use_relu=false, float learning_rate = 0.0001);

    Tensor<float, 2> forward(Tensor<float,2> &input); // dims = (batch, len)
    Tensor<float, 2> backward(Tensor<float,2> &dLdZ);


};