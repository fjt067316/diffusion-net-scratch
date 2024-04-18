#include "../../utils/Tensor.h"

#pragma once

class BatchNorm2d { // BN2D works per output channel ie for every input in batch normalize all their ch1 together then ch2 then ch3 
public:
    int out_channels;
    float epsilon;
    Tensor<float, 1> gamma;
    Tensor<float, 1> beta;
    Tensor<float, 1> d_gamma;
    Tensor<float, 1> d_beta;
    Tensor<float, 1> running_mean;
    Tensor<float, 1> running_var;
    Tensor<float, 1> means;
    Tensor<float, 1> vars;
    Tensor<float, 1> vars_inv;
    Tensor<float, 4> x_norm;
    Tensor<float, 4> x_mu;
    Tensor<float, 1> x_mu_sum;
    Tensor<float, 1> d_mu;
    Tensor<float, 1> d_var;
    Tensor<float, 4> input;
    bool do_allocs;


    BatchNorm2d(int out_channels, float epsilon = 1e-5);

    Tensor<float, 4> forward(Tensor<float, 4>& input);
    Tensor<float, 4> backward(Tensor<float, 4>& dLdZ);
};
