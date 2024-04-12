#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <cassert>
#include <initializer_list>
#include <utility>
#include <cmath>

#include "BatchNorm2d.h"

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

__global__ get_variance_sum(float* sum_arr, float* input, float* mean_arr, int* in_dims){
    int channel = threadIdx.x;
    int batch = blockIdx.x;
    
    int batch_size = in_dims[0];
    int height = in_dims[2];
    int width = in_dims[3];

    int idx = getIdx(in_dims, batch, channel, 0, 0);
    float sum = 0;
    float mean = mean_arr[channel];

    for(int i=idx; i<(idx+width*height); i++){
        int val = input[i];
        sum += (val - mean)*(val - mean);
    }

    sum_arr[channel*batch_size + batch] = sum;
}


__global__ get_channel_means_batch(float* means_arr, float* input, int* in_dims){
    int channel = threadIdx.x;
    int batch = blockIdx.x;

    int batch_size = in_dims[0];
    int height = in_dims[2];
    int width = in_dims[3];

    int idx = getIdx(in_dims, batch, channel, 0, 0);
    float sum = 0;

    for(int i=idx; i<(idx+width*height); i++){
        sum += input[i];
    }

    sum /= width*height;

    means_arr[channel*batch_size+batch] = sum;

    
}

__global__ get_variance_batch_sum(float* vars_arr, float* vars_out, float* vars_inv, int* vars_arr_dim){
    int channel = threadIdx.x;

    int batch_size = vars_arr_dim[1];
    int start_idx = batch_size * channel;

    float sum =0;
    for(int i=start; i<start+batch_size; i++){
        sum += means_arr[idx];
    }

    float var = sqrt(sum / batch_size);
    vars_out[ch] = var;
    vars_inv[ch] = 1/sqrt(var+0.00000001); // avoid division by zero

}

__global__ get_channel_means_total(float* means_arr, float* means_out, int* means_arr_dim){
    // one thread per channel and compute channel mean
    int channel = threadIdx.x;

    int batch_size = means_arr_dim[1];
    int start_idx = batch_size * channel;

    float sum =0;
    for(int i=start; i<start+batch_size; i++){
        sum += means_arr[idx];
    }

    means_out[ch] = sum / batch_size;

}

__global__ normalize_scale_shift(float* input, float* channel_means, float* channel_stds, float* gamma, float* beta, float* x_norm, float* x_mu, int* in_dims){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y
    int ch = threadIdx.z;
    int batch = blockIdx.z;

    int in_idx = getIdx(in_dims, batch, ch, y, x);

    float val = input[in_idx];
    float mean = channel_means[ch];
    float std = channel_stds[ch];
    float g = gamma[ch];
    float b = beta[ch];

    float mu = (val-mean);
    float normalized_in = mu / (std+0.00000001); // avoid division by zero
    float scaled_shifted_in = normalized_in*g+b;

    x_norm[in_idx] = normalized_in;
    x_mu[in_idx] = mu;
    input[in_idx] = scaled_shifted_in;
}

Tensor<float, 4> BatchNorm2d::forward(Tensor<float,4> &input){

    assert(input.dim(1) == this->out_channels);

    Tensor<float, 4> x_norm({input.dim(0), input.dim(1), input.dim(2), input.dim(3)}, true, true);
    Tensor<float, 4> x_mu({input.dim(0), input.dim(1), input.dim(2), input.dim(3)}, true, true);

    int batch_size = input.dim(0);
    int in_height = input.dim(2);
    int in_width = input.dim(3);

    // calculate channel mean across batch

    Tensor<float, 1> batch_means({out_channels, batch_size});

    // one kernel to compute mean of each input sample
    // one kernel to compute the means of the input means
    // this should work as inputs are same size so its equal mean weighting
    dim3 threadDim(output_channels);
    dim3 blockDim(batch_size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    get_channel_means_batch<<<blockDim, threadDim>>>(batch_means.data, input.data, input.d_dims);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 threadDimMean(output_channels);
    dim3 blockDimMean(1);

    get_channel_means_total<<<blockDimMean, threadDimMean>>>(batch_means.data, means.data, batch_means.d_dims);
    
    // curr_means holds a entry for each channel mean
    // create kernel to calculate sum = (input-mean)**2 per input
    Tensor<float, 1> batch_vars({out_channels, batch_size});

    get_variance_sum<<<blockDim, threadDim>>>(batch_vars.data, input.data, means.data, input.d_dims);

    get_variance_batch_sum<<<blockDimMean, threadDimMean>>>(batch_vars.data, vars.data, vars_inv.data, batch_vars.d_dims);

    // now vars has a std per channel and means has a mean per channel

    // moving average and std for inference
    // TODO


    // normalize, scale, shift
    int tds = 16;
    int block_height = (int) ceil((in_height) / tds);
    int block_width = (int) ceil((in_width) / tds);

    dim3 threadDim3d(tds, tds, output_channels);
    dim3 blockDim3d(block_width, block_height, batch_size);

    normalize_scale_shift<<<blockDim3d, threadDim3d>>>(input.data, means.data, vars.data, gamma.data, beta.data, x_norm.data, x_mu.data, input.d_dims);
    this->x_norm = x_norm;
    this->x_mu = x_mu;
    return input;
}

__global__ void get_db_dg(float* dldz, float* x_norm, float* d_gamma, float* x_mu, float* var_inv, float* dx_mu_batch, float* d_beta, int* dz_dims){
    // this computes for a single dz in batch we still need to sum across all items in batch
    // we will just compute dx_norm again for dldz next to save memory
    int batch = threadIdx.x;
    int channel = blockIdx.x;

    int height = dz_dims[2];
    int width = dz_dims[3];

    int start = getIdx(dz_dims, batch, channel, 0, 0);
    float d_g = 0;
    float d_b = 0;

    float dvar_batch_ch = 0;
    float dmu_batch_ch = 0;

    for(int i=start; i<width*height; i++){
        float dz = dldz[i];

        dg += dz * x_norm[i];
        db += dz;

        dx_norm = dz * gamma;

        dvar_batch_ch += dx_norm * x_mu[i];
        dmu_batch_ch += dx_norm * var_inv[channel];

    }




}

__global__ void get_dw_dz_next(float* dLdZ, float* x_mu, float* vars, float* vars_inv, float* gamma, float* beta, float* d_gamma, float* d_beta, float* dldz_next, int* dz_dims){
    int channel
}

Tensor<float, 4> BatchNorm2d::backward(Tensor<float, 4> dLdZ){
    // substract means from dLdZ
    // get x_norm = normalized x and x_mu = x - mean
    // var_inv = 1/sqrt(var+1e-8)

    // dBeta[channel] = sum(dLdZ[channel])
    // dGamma[channel] = sum_across_batch_channel(dLdZ[channel][i] * x_norm[channel][i] for i in channel)

    // compute dgamma dbeta dx_norm dx_centered in one kernel -> one kernel for everything?


    // dX_norm = dLdZ[ch][i]*gamma[ch]
    // dvar = 1d shape out channels = sum_over_channels(dX_norm*X_mu) * -0.5*(this->var+1e-8)**(-3/2)
    // dmu = 1d shape out channels = sum_over_channels(dX_norm*-var_inv)

}