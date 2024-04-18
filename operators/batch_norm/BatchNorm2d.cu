#include <cstdio>
#include <cuda_runtime.h>
#include <cassert>
#include <initializer_list>
#include <utility>
#include <cmath>
#include "../../utils/array_utils.h"

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

__global__ void get_variance_sum(float* sum_arr, float* input, float* mean_arr, int* in_dims){
    int channel = threadIdx.x;
    int batch = blockIdx.x;

    if(channel >= in_dims[1] || batch >= in_dims[0]){
        return;
    }
    
    // int batch_size = in_dims[0];
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


__global__ void get_channel_sums(float* means_arr, float* input, int* in_dims){
    int channel = threadIdx.x;
    int batch = blockIdx.x;

    if(channel >= in_dims[1] || batch >= in_dims[0]){
        return;
    }

    // int batch_size = in_dims[0];
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

__global__ void get_variance(float* vars_arr, float* vars_inv, int elements_in_ch, int channels){
    int channel = threadIdx.x;

    if(channel >= channels){
        return;
    }

    float sum = vars_arr[channel];

    float var = sqrt(sum / elements_in_ch);
    vars_arr[channel] = var;
    vars_inv[channel] = 1/sqrt(var+0.00000001); // avoid division by zero

}

__global__ void get_channel_means(float* means_arr, int elements_in_batch_ch, int channels){
    // one thread per channel and compute channel mean
    int channel = threadIdx.x;

    if(channel >= channels){
        return;
    }

    float val = means_arr[channel];
    means_arr[channel] = val / elements_in_batch_ch;

}

__global__ void normalize_scale_shift(float* input, float* channel_means, float* channel_stds, float* gamma, float* beta, float* x_norm, float* x_mu, float* x_mu_sum, int* in_dims){
    int x = blockIdx.z * blockDim.z + threadIdx.z; 
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int out_channels = in_dims[1];

    int z_idx = blockIdx.x*blockDim.x + threadIdx.x; // we reserve x idx which can hold a lot of blocks for our longest dim
    int ch = z_idx % out_channels;
    int batch = z_idx / out_channels;

    if(batch >= in_dims[0] || ch >= in_dims[1] || y >= in_dims[2] || x >= in_dims[3]){
        return;
    }

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

    atomicAdd(&x_mu_sum[ch], -2* mu);
}

__global__ void test(float *data, int size){
    for(int i=0; i<size; i++){
        float k = data[i];
        data[i] = k+1;
    }
}

BatchNorm2d::BatchNorm2d(int out_channels, float epsilon) :
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
        d_var({out_channels}, true, true)
    {
        cudaMemset(running_mean.data, 0, running_mean.size*sizeof(float));
        cudaMemset(running_var.data, 0, running_var.size*sizeof(float));
        do_allocs = true;
        // cudaMemset(bias.data, 0, bias.size*sizeof(float));
        unsigned long long seed = 123456; // Change this to any desired seed
        unsigned long long sequence_offset = 0;
        
        int grid = (int) ceil((double)(gamma.size) / 256);
        fill_rand<<<grid, 256>>>(gamma.data, gamma.size, seed, sequence_offset);
        grid = (int) ceil((double)(beta.size) / 256);
        fill_rand<<<grid, 256>>>(beta.data, beta.size, seed, sequence_offset);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

Tensor<float, 4> BatchNorm2d::forward(Tensor<float,4> &input){

    assert(input.dim(1) == this->out_channels);
    this->input = input;

    // Tensor<float, 4> x_norm({input.dim(0), input.dim(1), input.dim(2), input.dim(3)}, true, true);
    // Tensor<float, 4> x_mu({input.dim(0), input.dim(1), input.dim(2), input.dim(3)}, true, true);

    int batch_size = input.dim(0);
    int in_height = input.dim(2);
    int in_width = input.dim(3);

    // calculate channel mean across batch

    if(do_allocs){
        Tensor<float, 4> x_norm({batch_size, out_channels, in_height, in_width}, true, true);
        Tensor<float, 4> x_mu({batch_size, out_channels, in_height, in_width}, true, true);
        Tensor<float, 1> x_mu_sum({out_channels}, true, true);
        this->x_norm = x_norm;
        this->x_mu = x_mu;
        this->x_mu_sum = x_mu_sum;
        do_allocs = false;
    }
    cudaMemset(x_norm.data, 0, x_norm.size*sizeof(float));
    cudaMemset(x_mu.data, 0, x_mu.size*sizeof(float));
    cudaMemset(x_mu_sum.data, 0, x_mu_sum.size*sizeof(float));
    cudaMemset(means.data, 0, means.size*sizeof(float));
    cudaMemset(vars.data, 0, vars.size*sizeof(float));

    // Tensor<float, 1> channel_means({out_channels}, true, true);
    // Tensor<float, 1> x_mu_sum({out_channels}, true, true);
    // one kernel to compute mean of each input sample
    // one kernel to compute the means of the input means
    // this should work as inputs are same size so its equal mean weighting
    dim3 threadDim(out_channels);
    dim3 blockDim(batch_size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    get_channel_sums<<<blockDim, threadDim>>>(means.data, input.data, input.d_dims);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 threadDimMean(out_channels); // shouldnt exceed 1024
    dim3 blockDimMean(1);

    int elements_in_batch_ch = in_height*in_width*batch_size;

    get_channel_means<<<blockDimMean, threadDimMean>>>(means.data, elements_in_batch_ch, out_channels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // curr_means holds a entry for each channel mean
    // create kernel to calculate sum = (input-mean)**2 per input
    // Tensor<float, 1> vars({out_channels});

    get_variance_sum<<<blockDim, threadDim>>>(vars.data, input.data, means.data, input.d_dims);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    get_variance<<<blockDimMean, threadDimMean>>>(vars.data, vars_inv.data, elements_in_batch_ch, out_channels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // now vars has a std per channel and means has a mean per channel

    // moving average and std for inference
    // TODO

    // test<<<1,1>>>(x_norm.data, x_norm.size );
    // CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize());

    // normalize, scale, shift
    int tds = 16;
    int block_height = (int) ceil(((double)in_height) / tds);
    int block_width = (int) ceil(((double)in_width) / tds);
    int block_z = (int) ceil(((double)out_channels*batch_size) / 4);

    dim3 threadDim3d(3, tds, tds);
    dim3 blockDim3d(block_z, block_width, block_height);

    normalize_scale_shift<<<blockDim3d, threadDim3d>>>(input.data, means.data, vars.data, gamma.data, beta.data, x_norm.data, x_mu.data, x_mu_sum.data, input.d_dims);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return input; // should return a copy as its not a &function()
}

// __global__ void get_db_dg(float* dldz, float* x_norm, float* d_gamma, float* gamma, float* x_mu, float* var_inv, float* d_mu, float* d_var, float* d_beta, int* dz_dims){
//     // this computes for a single dz in batch we still need to sum across all items in batch
//     // we will just compute dx_norm again for dldz next to save memory
//     int batch = threadIdx.x;
//     int channel = blockIdx.x;

//     if(batch >= dz_dims[0] || channel >= dz_dims[1]){
//         return;
//     }

//     // int batch_size = dz_dims[0];
//     int height = dz_dims[2];
//     int width = dz_dims[3];

//     int start = getIdx(dz_dims, batch, channel, 0, 0);
//     float d_g = 0;
//     float d_b = 0;

//     float dvar_batch_ch = 0;
//     float dmu_batch_ch = 0;
//     float var_inv_ch = var_inv[channel];

//     for(int i=start; i<width*height; i++){
//         float dz = dldz[i];

//         d_g += dz * x_norm[i]; 
//         d_b += dz;

//         float dx_norm = dz * gamma[channel];

//         dvar_batch_ch += dx_norm * x_mu[i];
//         dmu_batch_ch += dx_norm ; // * var_inv can be taken out of loop

//     }

//     // atomic add to dg db?
//     atomicAdd(&d_gamma[channel], d_g);
//     atomicAdd(&d_beta[channel], d_b);

//     float dvar_b_c = dvar_batch_ch* -0.5*(1 / (var_inv_ch*sqrt(var_inv_ch))); // **-1.5
//     float d_mu_partial = dmu_batch_ch * -1*var_inv_ch; //+ dvar_b_c*(1/batch_size) * x_mu_sum_ch
//     atomicAdd(&d_var[channel], dvar_b_c);
//     atomicAdd(&d_mu[channel], d_mu_partial);
// }

__global__ void get_db_dg_dz_next(float* dz, float* input, float* d_gamma, float* d_beta, float* dz_next, float* gamma, float* beta, float* variance, int* dz_dims){
    // this computes for a single dz in batch we still need to sum across all items in batch
    // we will just compute dx_norm again for dldz next to save memory
    int batch = threadIdx.x;
    int channel = blockIdx.x;

    if(batch >= dz_dims[0] || channel >= dz_dims[1]){
        return;
    }

    // int batch_size = dz_dims[0];
    int height = dz_dims[2];
    int width = dz_dims[3];

    int start = getIdx(dz_dims, batch, channel, 0, 0);
    float d_g = 0;
    float d_b = 0;

    for(int i=start; i<width*height; i++){
        d_g += dz[i] * input[i];
        d_b += dz[i];
        dz_next[i] = dz[i] * gamma[channel] / sqrt(variance[channel] + 0.000001);
    }

    // atomic add to dg db?
    atomicAdd(&d_gamma[channel], d_g);
    atomicAdd(&d_beta[channel], d_b);

}

__global__ void get_dmu_from_partial(float* d_mu, float* d_var, float* x_mu_sum , int batch_size, int channels){
    int channel = threadIdx.x;

    if(channel >= channels){
        return;
    }

    float val = d_mu[channel];

    d_mu[channel] = val + d_var[channel] * (1/batch_size) * x_mu_sum[channel];
}

__global__ void get_dldz_next(float* dldz_next, float* dldz, float* gamma, float* var_inv, float* d_mu, float* d_var, float* x_mu, int* dz_dims, int output_channels, int batch_size){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int in_ch = blockIdx.z % output_channels;
    int batch = blockIdx.z / output_channels; // should be able to do this 12288 < 65000

    if(batch >= dz_dims[0] || in_ch >= dz_dims[1] || y >= dz_dims[2] || x >= dz_dims[3]){
        return;
    }

    int idx = getIdx(dz_dims, batch, in_ch, y, x);
    dldz_next[idx] = (dldz[idx]*gamma[in_ch]) + (d_mu[in_ch] / batch_size) + (d_var[in_ch]*2/batch_size*x_mu[idx]);
    
}

__global__ void apply_dz(float* gamma, float* beta, float* d_gamma, float* d_beta, int channels){
    int channel = threadIdx.x;

    if(channel >= channels){
        return;
    }
    // printf("d_gamma ch %f\n", d_gamma[channel]);
    // printf("d_beta ch %f\n", d_beta[channel]);

    gamma[channel] -= 0.01 * clip_to_range(d_gamma[channel]);
    beta[channel] -= 0.01 * clip_to_range(d_beta[channel]);
}


Tensor<float, 4> BatchNorm2d::backward(Tensor<float, 4> &dLdZ){
    // substract means from dLdZ
    // get x_norm = normalized x and x_mu = x - mean
    // var_inv = 1/sqrt(var+1e-8)
    int batch_size = dLdZ.dim(0);
    assert(input.dim(0) == dLdZ.dim(0));
    assert(input.dim(1) == dLdZ.dim(1));
    assert(input.dim(2) == dLdZ.dim(2));
    assert(input.dim(3) == dLdZ.dim(3));

    // compute dgamma dbeta dx_norm dx_centered in one kernel -> one kernel for everything?
    dim3 threadDimB(batch_size); // shouldnt exceed 1024
    dim3 blockDimC(out_channels);

    Tensor<float, 4> dLdZ_next({dLdZ.dim(0), dLdZ.dim(1), dLdZ.dim(2), dLdZ.dim(3)}, true, true);
    // float* dz, float* input, float* d_gamma, float* d_beta, float* dz_next, float* gamma, float* beta, int* dz_dims
    get_db_dg_dz_next<<<blockDimC, threadDimB>>>(dLdZ.data, input.data, d_gamma.data, d_beta.data, dLdZ_next.data, gamma.data, beta.data, vars.data, dLdZ.d_dims);

    // dX_norm = dLdZ[ch][i]*gamma[ch]
    // dvar = 1d shape out channels = sum_over_channels(dX_norm*X_mu) * -0.5*(this->var+1e-8)**(-3/2)
    // dmu = 1d shape out channels = sum_over_channels(dX_norm*-var_inv)

    dim3 threadDim(out_channels); // shouldnt exceed 1024
    dim3 blockDim(1);

    // apply d_gamma d_beta
    apply_dz<<<blockDim, threadDim>>>(gamma.data, beta.data, d_gamma.data, d_beta.data, out_channels);

    return dLdZ_next;
}

// Tensor<float, 4> BatchNorm2d::backward(Tensor<float, 4> &dLdZ){
//     // substract means from dLdZ
//     // get x_norm = normalized x and x_mu = x - mean
//     // var_inv = 1/sqrt(var+1e-8)
//     int batch_size = dLdZ.dim(0);
    
//     // memset d_gamma d_beta d_mu d_var to zeros ie anything we atomic add without setting
//     cudaMemset(d_gamma.data, 0, d_gamma.size*sizeof(float));
//     cudaMemset(d_beta.data, 0, d_beta.size*sizeof(float));
//     cudaMemset(d_mu.data, 0, d_mu.size*sizeof(float));
//     cudaMemset(d_var.data, 0, d_var.size*sizeof(float));

//     // compute dgamma dbeta dx_norm dx_centered in one kernel -> one kernel for everything?
//     dim3 threadDimB(batch_size); // shouldnt exceed 1024
//     dim3 blockDimC(out_channels);
//     get_db_dg<<<blockDimC, threadDimB>>>(dLdZ.data, x_norm.data, d_gamma.data, gamma.data, x_mu.data, vars_inv.data, d_mu.data, d_var.data, d_beta.data, dLdZ.d_dims);

//     // dX_norm = dLdZ[ch][i]*gamma[ch]
//     // dvar = 1d shape out channels = sum_over_channels(dX_norm*X_mu) * -0.5*(this->var+1e-8)**(-3/2)
//     // dmu = 1d shape out channels = sum_over_channels(dX_norm*-var_inv)

//     dim3 threadDim(out_channels); // shouldnt exceed 1024
//     dim3 blockDim(1);
//     get_dmu_from_partial<<<blockDim, threadDim>>>(d_mu.data, d_var.data, x_mu_sum.data, batch_size, out_channels); // yes the batch size not num elements in batch

//     Tensor<float, 4> dLdZ_next({dLdZ.dim(0), dLdZ.dim(1), dLdZ.dim(2), dLdZ.dim(3)}, true, true);
    
//     int tds = 16; // 2d block -> 256 threads per thread block
//     int block_height = (int) ceil((dLdZ.dim(2)) / tds);
//     int block_width = (int) ceil((dLdZ.dim(3)) / tds);

//     dim3 threadDimDz(tds, tds, 1);
//     dim3 blockDimDz(block_width, block_height, out_channels*batch_size);
//     get_dldz_next<<<blockDimDz, threadDimDz>>>(dLdZ_next.data, dLdZ.data, gamma.data, vars_inv.data, d_mu.data, d_var.data, x_mu.data, dLdZ_next.d_dims, out_channels, batch_size);

//     // apply d_gamma d_beta
//     apply_dz<<<blockDim, threadDim>>>(gamma.data, beta.data, d_gamma.data, d_beta.data, out_channels);

//     return dLdZ_next;
// }