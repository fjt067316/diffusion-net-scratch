#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <cassert>
#include <initializer_list>
#include <utility>
#include <cmath>

#include "./Conv2d.h"

__global__ void conv_forward(Tensor<float, 4> input, Tensor<float, 4> output, Tensor<float, 4> weights, Tensor<float,1> bias) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c_out = blockIdx.z * blockDim.z;

    int batch_size = input.dims[0],
        input_channels = input.dims[1],
        output_channels = output.dims[1],
        height = input.dims[2],
        width = input.dims[3],
        filter_size = weights.dims[2];

    if (x+filter_size > width || y+filter_size > height || c_out >= output_channels){
        return;
    }

    float bias_val = bias(c_out);

    for(int b=0; b < batch_size; b++){
        float sum = bias_val;

        for(int c=0; c<input_channels; c++){
            for(int row=y; row<y+filter_size; row++){
                for(int col=x; col<x+filter_size; col++){
                    sum += input(b, c, row, col) * weights(c_out, c, row-y, col-x);
                }
            }
        }
        
        if(threadIdx.x == 0){
            printf("sum %f \n", sum);

        }


        output(b, c_out, y, x) = sum;
    }
}

/*
assumes that data is already on gpu
*/
Tensor<float, 4> Conv2d::forward(Tensor<float,4> &input){
    // temp
    float* d_in;
    cudaMalloc(&d_in, input.size * sizeof(float));
    cudaMemcpy(d_in, input.data, input.size * sizeof(float), cudaMemcpyHostToDevice);
    float* d_weights;
    cudaMalloc(&d_weights, this->weights.size * sizeof(float));
    cudaMemcpy(d_weights, this->weights.data, this->weights.size * sizeof(float), cudaMemcpyHostToDevice);
    float* d_bias;
    cudaMalloc(&d_bias, this->bias.size * sizeof(float));
    cudaMemcpy(d_bias, this->bias.data, this->bias.size * sizeof(float), cudaMemcpyHostToDevice);

    this->bias.data = d_bias;
    this->weights.data = d_weights;
    input.data = d_in;

    // cudaMalloc(&d_in, 4*input.dims(0)*input.dims(1)*input.dims(2)*input.dims(3));
    // cudaMemcpy(input, d_in, sizeof(input), cudaMemcpyHostToDevice);
    //
    assert(input.dims[1] == this->input_channels);
    
    int batch_size = input.dims[0], height = input.dims[2], width = input.dims[3];

    int out_width = width - filter_size + 1;
    int out_height = height - filter_size + 1;

    Tensor<float, 4> output(batch_size, output_channels, out_height, out_width );

    int N = 768;// out_height*out_width;
    int tds = 16; // 2d block -> 256 threads per thread block
    int blocks = (int) ceil(N / tds);

    dim3 threadDim(tds, tds);
    dim3 blockDim(blocks, blocks, output_channels );

    conv_forward <<<blockDim, threadDim>>>(input, output, weights, bias);
    cudaDeviceSynchronize();

    Tensor<float, 4> result(batch_size, output_channels, out_height, out_width);

    // output.toHost(result.data, result.size*sizeof(float));

    cudaMemcpy(result.data, output.data, sizeof(float) * output.size, cudaMemcpyDeviceToHost);
    result.print();
    // free(output.data);

    // Free the device memory allocated for input tensor
    cudaFree(d_in);
    // cudaMemcpy(h_out, output, sizeof(output), cudaMemcpyDeviceToHost);

    return result;

    // if padding dont actually extend dims just dont compute conv elements if its idx in padding area

}