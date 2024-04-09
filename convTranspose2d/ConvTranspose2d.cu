#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <cassert>
#include <initializer_list>
#include <utility>
#include <math.h>

#include "ConvTranspose2d.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


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

__global__ void rotate180(float* input, float* output, int* dims) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int filter_idx = blockIdx.z;


    int height = dims[2];
    int width = dims[3];
    int num_channels = dims[1];

    if (x < width && y < height) {
        int in_index = filter_idx * width * height * num_channels + y * width + x;
        int out_index = filter_idx * width * height * num_channels + (height - y - 1) * width + (width - x - 1);
        output[out_index] = input[in_index];
    }
}



__global__ void pad_image_tranpose(float* input, float* output, int* in_dims, int* out_dims, int z, int pp){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // int c_out = blockIdx.z * blockDim.z;
    int b = blockIdx.z;

    int batch_size = in_dims[0],
        input_channels = in_dims[1],
        output_channels = out_dims[1],
        out_h = out_dims[2],
        out_w = out_dims[3];

    assert(input_channels == output_channels);

    if(x >= out_w || y >= out_h || b >= batch_size){
        return;
    }

    // check if output index should have an input value copied or be filled with zeros
    int x_valid = (x-pp) % (z+1);
    int y_valid = (y-pp) % (z+1);

    int idx = getIdx(out_dims, b, 0, y, x);
    int off = out_h*out_w;
    // zeros condition
    if(x < pp || y < pp || x >= out_w-pp || y >= out_h-pp || x_valid != 0 || y_valid != 0){
        for(int i=0; i<input_channels; i++){
            output[idx+i*off] = 0;
        }
        return;
    }

    // else fill with input value 
    int in_x = (x-pp) / (z+1);
    int in_y = (y-pp) / (z+1);
    // printf("row %d col %d val %d", in_y, in_x, getElement(input, in_dims, b, c_out, in_y, in_x));
    for(int i=0; i<input_channels; i++){
        // printf("added %f from %d %d to %d %d\n",getElement(input, in_dims, b, i, in_y, in_x), in_y, in_x, y, x );
        output[idx + i*off] = getElement(input, in_dims, b, i, in_y, in_x); 

    }
}


__global__ void conv_forward(float* input, float* output, float* weights, float* bias, int* in_dims, int* out_dims, int* w_dims, int padding, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c_out = blockIdx.z * blockDim.z;
    
    // printf("%d %d %d %d\n", weights[0], weights[1], weights[2], weights[3]);

    int batch_size = in_dims[0],
       input_channels = in_dims[1],
       output_channels = out_dims[1],
       height = in_dims[2],
       width = in_dims[3],
       filter_size = w_dims[2];

    if (x >= width+padding || y >= height+padding || x+filter_size-1 >= width+padding || y+filter_size-1 >= height+padding || c_out >= output_channels || (x%stride) != 0 || (y%stride) != 0){
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


template<typename Scalar>
void checkMemoryLocation(const Scalar* data) {
    cudaPointerAttributes attributes;
    cudaError_t error = cudaPointerGetAttributes(&attributes, data);
    if (error != cudaSuccess) {
        std::cerr << "Error getting pointer attributes: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    if (attributes.type == cudaMemoryTypeHost) {
        std::cout << "Memory is allocated on the host." << std::endl;
    } else if (attributes.type == cudaMemoryTypeDevice) {
        std::cout << "Memory is allocated on the device." << std::endl;
    } else {
        std::cout << "Memory location is unknown." << std::endl;
    }
}
/*
assumes that data is already on gpu
https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11

MUST ROATE WEIGHTS MATRIX 180deg BEFORE DOING CONV2D TO GET SAME AS PYTORCH => DURING BACKPROP ROATE WEIGHT MATRIX 180 
*/
Tensor<float, 4> ConvTranspose2d::forward(Tensor<float,4> &input){

    assert(input.dim(1) == this->input_channels);

    int batch_size = input.dim(0), in_channels = input.dim(1), height = input.dim(2), width = input.dim(3);
    int output_channels = weights.dim(0);
    // insert z zeros between input vals and pp zeros around edge
    int z = stride-1;
    int pp = filter_size-padding-1;

    // create temp input to perform conv on because I dont want to do weird atomic add transpose thing
    int h_tmp = 2*pp+z*(height-1) + height;
    int w_tmp = 2*pp+z*(width-1) + width;

    // create and pad tmp array
    Tensor<float, 4> tmp({batch_size, in_channels, h_tmp, w_tmp}, true, true);

    int tds = 16; // 2d block -> 256 threads per thread block
    int block_height = (int)ceil(((double)h_tmp) / (double)tds);
    int block_width = (int)ceil(((double)w_tmp) / (double)tds);

    dim3 threadDim(tds, tds, 1);
    dim3 blockDim(block_width, block_height, batch_size);

    checkMemoryLocation(input.data);
    checkMemoryLocation(tmp.data);
    checkMemoryLocation(tmp.d_dims);
    checkMemoryLocation(tmp.d_dims);
    CUDA_CHECK(cudaGetLastError()); // Ensure there's no previous kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize());

    pad_image_tranpose<<<blockDim, threadDim>>>(input.data, tmp.data, input.d_dims, tmp.d_dims, z, pp);

    CUDA_CHECK(cudaGetLastError()); // Ensure there's no previous kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize());
    // create output array for convolution
    int out_height = (height-1) * stride + filter_size-2*padding;
    int out_width = (width-1) * stride + filter_size-2*padding;

    Tensor<float, 4> output({batch_size, output_channels, out_height, out_width}, false); // output for actual convolution
    cudaMalloc(&output.data, output.size*sizeof(float));

    tds = 16; // 2d block -> 256 threads per thread block
    block_height = (int) ceil((double)out_height / (double)tds);
    block_width = (int) ceil((double)out_width / (double)tds);

    dim3 threadDimOut(tds, tds, 1);
    dim3 blockDimOut(block_width, block_height, output_channels); // output_channel = num weights

    checkMemoryLocation(tmp.data);
    checkMemoryLocation(output.data);
    checkMemoryLocation(weights.data);
    checkMemoryLocation(bias.data);
    CUDA_CHECK(cudaGetLastError()); // Ensure there's no previous kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize());
    conv_forward <<<blockDimOut, threadDimOut>>>(tmp.data, output.data, weights.data, bias.data, tmp.d_dims, output.d_dims, weights.d_dims,0, 1); // no padding argument
    CUDA_CHECK(cudaGetLastError()); // Ensure there's no previous kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(tmp.data);

    return output;
}

__global__ void test(float* weights, int n){
    for(int i=0; i<n; i++){
        printf("%f ", weights[i]);
    }
    printf("\n");
}

Tensor<float, 4> conv_transpose_2d(Tensor<float,4> &input, Tensor<float, 4> weights, Tensor<float, 1> bias, int padding, int stride, bool rotate180_weights){

    if(rotate180_weights){
        float* rot;
        cudaMalloc(&rot, weights.size*sizeof(float));

        int rotTds = 16; // 2d block -> 256 threads per thread block
        int block_height = (int)ceil((double)weights.dim(2) / (double)rotTds);
        int block_width = (int)ceil((double)weights.dim(3) / (double)rotTds);
        checkMemoryLocation(weights.data);
        checkMemoryLocation(rot);

        dim3 rotBlockDim(rotTds, rotTds); // You may adjust block dimensions according to your matrix size
        dim3 rotGridDim(block_width, block_height, weights.dim(0));
        rotate180<<<rotGridDim, rotBlockDim>>>(weights.data, rot, weights.d_dims);
        cudaDeviceSynchronize();
        weights.data = rot;
    }

    int batch_size = input.dim(0), in_channels = input.dim(1), height = input.dim(2), width = input.dim(3), filter_size = weights.dim(2);
    int output_channels = weights.dim(0);
    // insert z zeros between input vals and pp zeros around edge
    int z = stride-1;
    int pp = filter_size-padding-1;

    // create temp input to perform conv on because I dont want to do weird atomic add transpose thing
    int h_tmp = 2*pp+z*(height-1) + height;
    int w_tmp = 2*pp+z*(width-1) + width;

    // create and pad tmp array
    Tensor<float, 4> tmp({batch_size, in_channels, h_tmp, w_tmp}, true, true);

    int tds = 16; // 2d block -> 256 threads per thread block
    int block_height = (int)ceil(((double)h_tmp) / (double)tds);
    int block_width = (int)ceil(((double)w_tmp) / (double)tds);

    dim3 threadDim(tds, tds, 1);
    dim3 blockDim(block_width, block_height, batch_size);

    checkMemoryLocation(input.data);
    checkMemoryLocation(tmp.data);
    checkMemoryLocation(tmp.d_dims);
    checkMemoryLocation(tmp.d_dims);
    CUDA_CHECK(cudaGetLastError()); // Ensure there's no previous kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize());

    pad_image_tranpose<<<blockDim, threadDim>>>(input.data, tmp.data, input.d_dims, tmp.d_dims, z, pp);

    CUDA_CHECK(cudaGetLastError()); // Ensure there's no previous kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize());
    // create output array for convolution
    int out_height = (height-1) * stride + filter_size-2*padding;
    int out_width = (width-1) * stride + filter_size-2*padding;

    Tensor<float, 4> output({batch_size, output_channels, out_height, out_width}, false); // output for actual convolution
    cudaMalloc(&output.data, output.size*sizeof(float));

    tds = 16; // 2d block -> 256 threads per thread block
    block_height = (int) ceil((double)out_height / (double)tds);
    block_width = (int) ceil((double)out_width / (double)tds);

    dim3 threadDimOut(tds, tds, 1);
    dim3 blockDimOut(block_width, block_height, output_channels); // output_channel = num weights

    checkMemoryLocation(tmp.data);
    checkMemoryLocation(output.data);
    checkMemoryLocation(weights.data);
    checkMemoryLocation(bias.data);
    CUDA_CHECK(cudaGetLastError()); // Ensure there's no previous kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize());
    conv_forward <<<blockDimOut, threadDimOut>>>(tmp.data, output.data, weights.data, bias.data, tmp.d_dims, output.d_dims, weights.d_dims,0, 1); // no padding argument
    CUDA_CHECK(cudaGetLastError()); // Ensure there's no previous kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(tmp.data);

    return output;
}