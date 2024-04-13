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
    atomicAdd(&sum_arr[channel], sum);
}


__global__ get_channel_sums(float* means_arr, float* input, int* in_dims){
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

    // sum /= width*height*batch_size; // (a+b+c+d) / N == a/N + b/N + c/N but takes more divisions

    atomicAdd(&means_arr[channel], sum);

    
}

__global__ get_variance(float* vars_arr, int elements_in_ch){
    int channel = threadIdx.x;

    float sum = vars_arr[ch];

    float var = sqrt(sum / elements_in_ch);
    vars_arr[ch] = var;
    vars_inv[ch] = 1/sqrt(var+0.00000001); // avoid division by zero

}

__global__ get_channel_means(float* means_arr, int elements_in_batch_ch){
    // one thread per channel and compute channel mean
    int channel = threadIdx.x;

    float val = means_arr[ch];
    means_arr[ch] = val / elements_in_batch_ch;

}

__global__ normalize_scale_shift(float* input, float* channel_means, float* channel_stds, float* gamma, float* beta, float* x_norm, float* x_mu, float* x_mu_sum, int* in_dims){
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

    atomicAdd(x_mu_sum[ch], -2* x_mu);
}

Tensor<float, 4> BatchNorm2d::forward(Tensor<float,4> &input){

    assert(input.dim(1) == this->out_channels);

    Tensor<float, 4> x_norm({input.dim(0), input.dim(1), input.dim(2), input.dim(3)}, true, true);
    Tensor<float, 4> x_mu({input.dim(0), input.dim(1), input.dim(2), input.dim(3)}, true, true);

    int batch_size = input.dim(0);
    int in_height = input.dim(2);
    int in_width = input.dim(3);

    // calculate channel mean across batch

    Tensor<float, 1> channel_means({out_channels}, true, true);
    Tensor<float, 1> x_mu_sum({out_channels}, true, true);
    // one kernel to compute mean of each input sample
    // one kernel to compute the means of the input means
    // this should work as inputs are same size so its equal mean weighting
    dim3 threadDim(output_channels);
    dim3 blockDim(batch_size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    get_channel_sums<<<blockDim, threadDim>>>(channel_means.data, input.data, input.d_dims);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 threadDimMean(output_channels); // shouldnt exceed 1024
    dim3 blockDimMean(1);

    int elements_in_batch_ch = in_height*in_width*batch_size;

    get_channel_means<<<blockDimMean, threadDimMean>>>(channel_means.data, elements_in_batch_ch);
    
    // curr_means holds a entry for each channel mean
    // create kernel to calculate sum = (input-mean)**2 per input
    Tensor<float, 1> vars({out_channels});

    get_variance_sum<<<blockDim, threadDim>>>(vars.data, input.data, means.data, input.d_dims);

    get_variance<<<blockDimMean, threadDimMean>>>(vars.data, vars_inv.data, elements_in_batch_ch);

    // now vars has a std per channel and means has a mean per channel

    // moving average and std for inference
    // TODO


    // normalize, scale, shift
    int tds = 16;
    int block_height = (int) ceil((in_height) / tds);
    int block_width = (int) ceil((in_width) / tds);

    dim3 threadDim3d(tds, tds, output_channels);
    dim3 blockDim3d(block_width, block_height, batch_size);

    normalize_scale_shift<<<blockDim3d, threadDim3d>>>(input.data, means.data, vars.data, gamma.data, beta.data, x_norm.data, x_mu.data, x_mu_sum.data, input.d_dims);
    this->x_norm = x_norm;
    this->x_mu = x_mu;
    this->x_mu_sum = x_mu_sum;
    return input;
}

__global__ void get_db_dg(float* dldz, float* x_norm, float* d_gamma, float* x_mu, float* var_inv, float* d_mu, float* d_beta, int* dz_dims){
    // this computes for a single dz in batch we still need to sum across all items in batch
    // we will just compute dx_norm again for dldz next to save memory
    int batch = threadIdx.x;
    int channel = blockIdx.x;

    int batch_size = dz_dims[0];
    int height = dz_dims[2];
    int width = dz_dims[3];

    int start = getIdx(dz_dims, batch, channel, 0, 0);
    float d_g = 0;
    float d_b = 0;

    float dvar_batch_ch = 0;
    float dmu_batch_ch = 0;
    float var_inv_ch = var_inv[ch];
    float x_mu_sum = 0;

    for(int i=start; i<width*height; i++){
        float dz = dldz[i];

        dg += dz * x_norm[i]; 
        db += dz;

        dx_norm = dz * gamma;
        x_mu_sum += x_mu[i];

        dvar_batch_ch += dx_norm * x_mu[i];
        dmu_batch_ch += dx_norm ; // * var_inv can be taken out of loop

    }

    // atomic add to dg db?
    atomicAdd(&d_gamma[ch], dg);
    atomicAdd(&d_beta[ch], db);

    float dvar_b_c = dvar_batch_ch* -0.5*(1 / (var_inv_ch*sqrt(var_inv_ch))); // **-1.5
    float d_mu_partial = dmu_batch_ch * -1*var_inv_ch //+ dvar_b_c*(1/batch_size) * x_mu_sum_ch
    atomicAdd(dvar[ch], dvar_b_c);
    atomicAdd(d_mu[ch], d_mu_partial);
}

__global__ void get_dmu_from_partial(float* d_mu, float* d_var, float* x_mu_sum , int batch_size){
    int channel = threadIdx.x;

    float val = d_mu[ch];

    d_mu[ch] = val + dvar[ch] * (1/batch_size) * x_mu_sum[ch];
}

__global__ void get_dldz_next(float* dldz_next, float* gamma, float* var_inv, float* d_mu, float* d_var, float* x_mu, int output_channels){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int in_ch = blockIdx.z % output_channels;
    int batch = blockIdx.z / output_channels; // should be able to do this 12288 < 65000
    
}

__global__ void apply_dz(float* gamma, float* beta, float* d_gamma, float* d_beta){
    int channel = threadIdx.x;

    gamma[channel] -= 0.0001 * d_gamma[channel];
    beta[channel] -= 0.0001 * d_beta[channel];
}

Tensor<float, 4> BatchNorm2d::backward(Tensor<float, 4> dLdZ){
    // substract means from dLdZ
    // get x_norm = normalized x and x_mu = x - mean
    // var_inv = 1/sqrt(var+1e-8)
    int batch_size = dLdZ.dim(0);
    
    // memset d_gamma d_beta d_mu d_var to zeros ie anything we atomic add without setting
    cudaMemset(d_gamma.data, 0, d_gamma.size*sizeof(float));
    cudaMemset(d_beta.data, 0, d_beta.size*sizeof(float));
    cudaMemset(d_mu.data, 0, d_mu.size*sizeof(float));
    cudaMemset(d_var.data, 0, d_var.size*sizeof(float));

    // compute dgamma dbeta dx_norm dx_centered in one kernel -> one kernel for everything?
    dim3 threadDimB(batch_size); // shouldnt exceed 1024
    dim3 blockDimC(out_channels);
    get_db_dg<<<blockDimC, threadDimB>>>(dldz.data, x_norm.data, d_gamma.data, x_mu.data, var_inv.data, d_mu.data, d_beta.data, dLdZ.d_dims);

    // dX_norm = dLdZ[ch][i]*gamma[ch]
    // dvar = 1d shape out channels = sum_over_channels(dX_norm*X_mu) * -0.5*(this->var+1e-8)**(-3/2)
    // dmu = 1d shape out channels = sum_over_channels(dX_norm*-var_inv)

    dim3 threadDim(out_channels); // shouldnt exceed 1024
    dim3 blockDim(1);
    get_dmu_from_partial<<<blockDim, threadDim>>>(d_mu.data, d_var.data, x_mu_sum.data, batch_size); // yes the batch size not num elements in batch

    Tensor<float, 4> dLdZ_next({dLdZ.dim(0), dLdZ.dim(1), dLdZ.dim(2), dLdZ.dim(3)}, true, true);
    
    int tds = 16; // 2d block -> 256 threads per thread block
    int block_height = (int) ceil((dLdZ.dim(2)) / tds);
    int block_width = (int) ceil((dLdZ.dim(3)) / tds);

    dim3 threadDimDz(tds, tds, 1);
    dim3 blockDimDz(block_width, block_height, output_channels*batch_size);
    get_dldz_next<<<blockDimDz, threadDimDz>>>(dLdZ_next.data, gamma.data, var_inv.data, d_mu.data, d_var.data, x_mu, output_channels);

    // apply d_gamma d_beta
    apply_dz<<<blockDim, threadDim>>>(gamma.data, beta.data, d_gamma.data, d_beta.data);
    
    return dLdZ_next;
}