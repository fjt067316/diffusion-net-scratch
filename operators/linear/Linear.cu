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

__global__ void get_dw(float* dw, float* dz, float* in, int* w_dims, int* dz_dims, int* in_dims){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;

    int batch_size = in_dims[0];
    float sum = 0;
    int in_size = in_dims[1];
    int dz_size = dz_dims[1];
    for(int b=0; b<batch_size; b++){
        // sum += out[row]*input[batch][col]
        int input_val = in[in_size*b+x];//getElement(in, in_dims, batch, x);
        int dz_val = dz[b*dz_size+y];
        sum += input_val*dz_val;
    }

    int idx = getIdx(dw, w_dims, y, x);
    dw[idx] = sum/batch_size;
}

__global__ void get_dldz_next(float* dz_next, float* dz, float* weights, int* next_dims, int* dz_dims, int* w_dims){
    int in_col = blockIdx.x*blockDim.x+threadIdx.x;
    int batch = blockIdx.y;

    int w_rows = w_dims[0];
    int w_row_size = w_dims[1];

    int dz_batch_size = dz_dims[1];

    float sum = 0;
    for(int i=0; i<w_rows; i++){
        int w = weights[i*w_row_size + in_col]; // coalesce this
        int dz = dz[batch*dz_batch_size+i];

        sum += w*dz;
    }

    dz_next[w_row_size*batch+in_col] = sum; // dz_next[b][in_col] = w*dz
}

__global__ void apply_dw(float* dw, float* weights, int* w_dims){
    x = blockIdx.x*blockDim.x+threadIdx.x;
    y = blockIdx.y*blockDim.y+threadIdx.y;

    int row_size = w_dims[1];

    weights[y*row_size + x] -= 0.0001*dw[y*row_size+x];
}

Tensor<float, 2> Linear::forward(Tensor<float,2> &dLdZ){


    Tensor<float, 2> dw({wights.dim(0), weights.dim(1)}, true, true);

    // one thread per dw
    int tds = 16; 
    int blocks_w = (int) ceil(this->weights.dim(1) / tds);
    int blocks_h = (int) ceil(this->weights.dim(0) / tds);

    dim3 threadDim(tds, tds); // one thread per row of Linear layer and batch size
    dim3 blockDim(blocks_w, blocks_h);

    get_dw<<<blockDim, threadDim>>>(dw.data, dLdZ.data, this->input.data, dw.d_dims, dLdZ.d_dims, this->input.d_dims);

    tds = 512; 
    blocks_w = (int) ceil(this->weights.dim(1) / tds);
    int batch_size = this->input.dim(0);

    dim3 threadDimDz(tds, 1); // one thread per row of Linear layer and batch size
    dim3 blockDimDz(blocks_w, batch_size);

    Tensor<float, 4> dLdZ_next({this->input.dim(0), input.dim(1), input.dim(2), input.dim(3)}, true, true);
    get_dldz_next(dLdZ_next.data, dLdZ.data, weights.data, dLdZ_next.d_dims, dLdZ.d_dims, weights.d_dims);

    // apply dw
    tds = 16; 
    blocks_w = (int) ceil(this->weights.dim(1) / tds);
    blocks_h = (int) ceil(this->weights.dim(0) / tds);

    dim3 threadDimDw(tds, tds); // one thread per row of Linear layer and batch size
    dim3 blockDimDw(blocks_w, batch_size);

    apply_dw<<<blockDimDw, threadDimDw>>>(dw.data, weights.data, weights.d_dims);

    return dLdZ_next;
}