#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <cassert>
#include <initializer_list>
#include <utility>
#include <cmath>

#include "Conv2d.h"


__host__ __device__ float getElement(float *arr, int i) {
    return arr[i];
}

__host__ __device__ float getElement(float *arr, int* dims, int i, int j) {
    return arr[i * dims[1] + j];
}

__host__ __device__ float getElement(float *arr, int* dims, int i, int j, int k) {
    return arr[i * dims[1] * dims[2] + j * dims[2] + k];
}

__host__ __device__ float getElement(float *arr, int* dims, int i, int j, int k, int l) {
    return arr[i * dims[1] * dims[2] * dims[3] + j * dims[2] * dims[3] + k * dims[3] + l];
}

__host__ __device__ float getIdx(int* dims, int i, int j, int k, int l) {
    return i * dims[1] * dims[2] * dims[3] + j * dims[2] * dims[3] + k * dims[3] + l;
}


__global__ void conv_forward(float* input, float* output, float* weights, float* bias, int* in_dims, int* out_dims, int* w_dims, int padding, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c_out = blockIdx.z * blockDim.z;
    
    int batch_size = in_dims[0],
       input_channels = in_dims[1],
       output_channels = out_dims[1],
       height = in_dims[2],
       width = in_dims[3],
       filter_size = w_dims[2];

    if (x >= width+padding || y >= height+padding || x+filter_size-1 > width+padding || y+filter_size-1 > height+padding || c_out >= output_channels || (x%stride) != 0 || (y%stride) != 0){
        return;
    }

    float bias_val = bias[c_out];

    for(int b=0; b < batch_size; b++){
        float sum = bias_val;
        for(int c=0; c<input_channels; c++){
            for(int row=y; row<y+filter_size; row++){
                for(int col=x; col<x+filter_size; col++){
                    if(row < padding || col < padding || row >= height+padding || col >= width+padding ){ // padding guard
                        continue;
                    } else{
                        sum += getElement(input, in_dims, b, c, row - padding, col - padding) * getElement(weights, w_dims, c_out, c, row - y, col - x);
                    }
                }
            }
        }
        int idx = getIdx(out_dims, b, c_out, y, x);
        output[idx] = sum;
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

    assert(input.dim(1) == this->input_channels);
    
    int batch_size = input.dim(0), height = input.dim(2), width = input.dim(3);

    int out_width = (width - filter_size + 2 * padding) / stride + 1;
    int out_height = (height - filter_size + 2 * padding) / stride + 1;

    Tensor<float, 4> output({batch_size, output_channels, out_height, out_width}, false); // do_allocs=false
    float* d_out;
    cudaMalloc(&d_out, output.size * sizeof(float));
    output.data = d_out;
    // int N = 768;// out_height*out_width;
    int tds = 16; // 2d block -> 256 threads per thread block
    int block_height = (int) ceil((512 + 2*padding) / tds);
    int block_width = (int) ceil((768 + 2*padding) / tds);

    dim3 threadDim(tds, tds, 1);
    dim3 blockDim(block_width, block_height, output_channels);

    conv_forward <<<blockDim, threadDim>>>(input.data, output.data, weights.data, bias.data, input.d_dims, output.d_dims, weights.d_dims, padding, stride);

    Tensor<float, 4> result(batch_size, output_channels, out_height, out_width);

    cudaMemcpy(result.data, output.data, sizeof(float) * output.size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // Free the device memory allocated for input tensor
    cudaFree(d_in);

    return result;
}