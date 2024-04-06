#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <cassert>
#include <initializer_list>
#include <utility>
#include <cmath>

#include "Linear.h"

__constant__ FloatArray in;

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


__global__ void linear_forward(float* input, float* output, float* weights, float* bias, bool use_bias, int* in_dims, int* out_dims, int* w_dims, int* bias_dims)
    {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = threadIdx.y;

    int batch_size = in_dims[0];
    int out_size = w_dims[0];
    int in_rows = in_dims[1];


    if(row >= out_size || batch >= batch_size){
        return;
    }

    float sum = use_bias ? bias[row] : 0;

    for(int i=0; i<in_rows; i++){
        
        sum = getElement(input, in_dims, batch, i) * getElement(weights, w_dims, row, i);
    }

    output[batch * out_size + row] = sum;
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


// assumes data already on gpu
Tensor<float, 2> Linear::forward(Tensor<float,2> &input){

    int batch_size = input.dim(0);
    
    assert(this->input_size == input.dim(1));

    int tds = 16; 
    int blocks = (int) ceil(this->output_size / tds);

    dim3 threadDim(tds, batch_size); // one thread per row of Linear layer and batch size
    dim3 blockDim(blocks, 1);
    
    CUDA_CHECK(cudaGetLastError()); // Ensure there's no previous kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize());

    Tensor<float, 2> output({batch_size, this->output_size}, true, true); // do_alloc = true, to_device = true

    checkMemoryLocation(input.data);
    checkMemoryLocation(output.data);
    checkMemoryLocation(weights.data);
    checkMemoryLocation(bias.data);

    checkMemoryLocation(input.d_dims);
    checkMemoryLocation(weights.d_dims);

    linear_forward <<<blockDim, threadDim>>>(input.data, output.data, weights.data, bias.data, this->use_bias, input.d_dims, output.d_dims, weights.d_dims, bias.d_dims);

    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to finish

    return output;

}
