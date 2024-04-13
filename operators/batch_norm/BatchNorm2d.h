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


    BatchNorm2d(int out_channels, float epsilon = 1e-5) :
        out_channels(out_channels),
        epsilon(epsilon),
        gamma({out_channels}, true, true),
        beta({out_channels}, true, true),
        d_gamma({out_channels}, true, true),
        d_beta({out_channels}, true, true),
        running_mean({out_channels}, true, true),
        running_var({out_channels}, true, true),
        means({out_channels}, true, true),
        vars({out_channels}, true, true),
        vars_inv({out_channels}, true, true),
        d_mu({out_channels}, true, true),
        d_var({out_channels}, true, true),


    {
        cudaMemset(running_mean.data, 0, running_mean.size*sizeof(float));
        cudaMemset(running_var.data, 0, running_var.size*sizeof(float));
        // cudaMemset(bias.data, 0, bias.size*sizeof(float));
    }

    Tensor<float, 4> forward(Tensor<float, 4>& input);
    Tensor<float, 4> backward(Tensor<float, 4>& dLdZ);
};
