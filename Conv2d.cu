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

    if (x+filter_size >= width || y+filter_size >= height || c_out >= output_channels){
        return;
    }

    float bias_val = bias(c_out);

    for(int b=0; b < batch_size; b++){
        float sum = bias_val;

        for(int c=0; c<input_channels; c++){
            for(int row=y; row<y+filter_size; row++){
                for(int col=x; col<x+filter_size; col++){
                    sum += input(b, c, row, col) * weights(c_out, row, col);
                }
            }
        }

        output(b, c_out, x, y) = sum;
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
    // cudaMalloc(&d_in, 4*input.dims(0)*input.dims(1)*input.dims(2)*input.dims(3));
    // cudaMemcpy(input, d_in, sizeof(input), cudaMemcpyHostToDevice);
    //
    assert(input.dims[1] == this->input_channels);
    
    int batch_size = input.dims[0], height = input.dims[2], width = input.dims[3];

    int out_width = width - filter_size + 1;
    int out_height = height - filter_size + 1;

    Tensor<float, 4> output(batch_size, output_channels, out_height, out_width );

    int N = out_height*out_width;
    int tds = 16; // 2d block -> 256 threads per thread block
    int blocks = (int) ceil(N / tds);

    dim3 threadDim(tds, tds);
    dim3 blockDim(blocks, blocks, output_channels );

    conv_forward <<<blockDim, threadDim>>>(input, output, weights, bias);
    
    // Temp 
    float* h_out;
    h_out = (float*) malloc(sizeof(int)*output.dims[0]*output.dims[1]*output.dims[2]*output.dims[3]);

    Tensor<float, 4> result(batch_size, output_channels, out_height, out_width);

    output.toHost(h_out, batch_size*output_channels*out_height*out_width);

    cudaMemcpy(result.data, h_out, sizeof(float) * output.dims[0] * output.dims[1] * output.dims[2] * output.dims[3], cudaMemcpyHostToDevice);

    free(h_out);

    // Free the device memory allocated for input tensor
    cudaFree(d_in);
    // cudaMemcpy(h_out, output, sizeof(output), cudaMemcpyDeviceToHost);

    return result;

    // if padding dont actually extend dims just dont compute conv elements if its idx in padding area

}