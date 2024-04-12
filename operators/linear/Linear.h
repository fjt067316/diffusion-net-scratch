#include "../../utils/Tensor.h"

#pragma once 

class Linear // public Layer
{
public:
    int input_size;
    int output_size;
    Tensor<float, 2> weights;
    Tensor<float, 1> bias;
    bool use_bias;
    bool use_relu;

    Linear(int input_size, int output_size, bool use_bias=false, bool use_relu=false) : 
    input_size(input_size), 
    output_size(output_size), 
    weights(output_size, input_size),
    use_bias(use_bias)
    {
        if (use_bias){
            bias = Tensor<float, 1>({output_size});
        }

    }

    Tensor<float, 2> forward(Tensor<float,2> &input); // dims = (batch, len)
    Tensor<float, 2> backward(Tensor<float,2> &dLdZ);


};