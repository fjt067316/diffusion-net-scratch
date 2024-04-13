#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <cassert>
#include <initializer_list>
#include <utility>
#include <cmath>

#include "Conv2d.h"

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


__global__ void conv_forward(float* input, float* output, float* weights, float* bias, int* in_dims, int* out_dims, int* w_dims, int padding, int stride, bool use_bias, bool use_relu) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c_out = blockIdx.z;
    
    int batch_size = in_dims[0],
       input_channels = in_dims[1],
       output_channels = out_dims[1],
       height = in_dims[2],
       width = in_dims[3],
       filter_size = w_dims[2];

    if (x >= width+padding || y >= height+padding || x+filter_size-1 >= width+padding || y+filter_size-1 >= height+padding || c_out >= output_channels || (x%stride) != 0 || (y%stride) != 0){
        return;
    }

    float bias_val = use_bias ? bias[c_out] : 0;

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

        int idx = getIdx(out_dims, b, c_out, y/stride, x/stride);
        if(use_relu && sum < 0){
            output[idx] = 0;
        }else{
            output[idx] = sum;
        }

    }
}


__global__ void conv_dldz_next(float* aug_w, float* dldz_next, float* dldz, int* w_dims, int* next_dims, int* dz_dims, int padding, int stride, int output_channels) {
    // one thread per dl_dz_next element (excluding batch dim)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int in_ch = blockIdx.z % output_channels;
    int batch = blockIdx.z / output_channels; // should be able to do this 12288 < 65000
    
    // we want a 2d filter to sweep across all 3 channels and compute the 2d*2d conv on each layer
    // ie 3d input 2d weight (one filter / layer of dldz) 3d output
    int batch_size = next_dims[0];
    int filter_size_h = dz_dims[2];
    int filter_size_w = dz_dims[3];
    int n_filter = w_dims[0];
    int height = aug_w[2];
    int width = aug_w[3];

    if (x+filter_size-1 >= width || y+filter_size-1 >= height || in_ch >= next_dims[1] || (x%stride) != 0 || (y%stride) != 0){
        return;
    }
    // we will just perform a 2d by 2d convoultion and spawn 1 thread for every element in output and across every filter (excluding batch size)
    // for(int b=0; b<batch_size; b++){
    float sum = 0;
    for(int filter_idx=0; filter_idx<n_filter; filter++){
        for(int row=y; row<y+filter_size_h; row++){
            for(int col=x; col<x+filter_size_w; col++){
                sum += getElement(dldz, dz_dims, batch, filter_idx, row-y, col-x) * getElement(aug_w, w_dims, filter_idx, in_ch, row, col);
            }
        }
    }

    int idx = getIdx(next_dims, batch, in_ch, y/stride, x/stride);
    dldz_next[idx] = sum;
    // }
    
}

/*
assumes that data is already on gpu
*/
Tensor<float, 4> Conv2d::forward(Tensor<float,4> &input){
    // temp
    this->input = input;

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
    int block_height = (int) ceil((height + 2*padding) / tds);
    int block_width = (int) ceil((width + 2*padding) / tds);

    dim3 threadDim(tds, tds, 1);
    dim3 blockDim(block_width, block_height, output_channels);

    conv_forward <<<blockDim, threadDim>>>(input.data, output.data, weights.data, bias.data, input.d_dims, output.d_dims, weights.d_dims, padding, stride, this->use_bias, this->use_relu);

    Tensor<float, 4> result(batch_size, output_channels, out_height, out_width);

    cudaMemcpy(result.data, output.data, sizeof(float) * output.size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    // Free the device memory allocated for input tensor
    cudaFree(d_in);

    return result;
}


// Backprop stuff below

// this is correct stop looking at it
__global__ void get_dw(float* input, float* dLdZ, float* output, int* in_dim, int* dz_dim, int* out_dim, int padding, int stride, int n_filters){
    int x = blockIdx.z * blockDim.z + threadIdx.z; 
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int z_idx = blockIdx.x*blockDim.x + threadIdx.x; // we reserve x idx which can hold a lot of blocks for our longest dim
    int filter_idx = z_idx % n_filters;
    int channel = z_idx / n_filters;


    int filter_size = out_dim[2];
    int batch_size = in_dim[0];

    if(x >= filter_size || y >= filter_size || filter_idx >= dz_dim[1] || channel >= in_dim[1]){
        return;
    }

    int in_w = in_dim[3] + 2*padding;
    int in_h = in_dim[2] + 2*padding;

    int w_moves = (in_w-filter_size)/stride+1;
    int h_moves = (in_h-filter_size)/stride+1;

    
    assert(h_moves == dz_dim[2]);
    assert(w_moves == dz_dim[3]);
    float dw = 0;


    for(int b=0; b<batch_size; b++){
        int dz_idx = getIdx(dz_dim, b, filter_idx, 0,0); // should be 1 dz for every move made 

        for(int i=0, y_off=0; i<h_moves; i++, y_off += stride){ 
            for(int j=0, x_off=0; j<w_moves; j++, x_off += stride, dz_idx++){

            if(idx_x < padding || idx_y < padding || idx_x >= in_w-padding || idx_y >= in_h-padding){
                continue;
            }
            
            int idx_x = x+x_off;
            int idx_y = y+y_off;

            float in = getElement(input, in_dim, b, channel, idx_y-padding, idx_x-padding);
            dw += in; // maybe make it dw += in; then at end dw *= dz
            }
        }
    }
    dw *= * dZ;
    int idx = getIdx(out_dim, filter_idx, channel, y, x);
    output[idx] = dw / batch_size;

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

Tensor<float, 4> conv_transpose_2d_dldz(Tensor<float,4> &input, Tensor<float, 4> weights, Tensor<float, 1> bias, int padding, int stride, bool rotate180_weights , bool use_bias = false){

    if(rotate180_weights){
        float* rot;
        cudaMalloc(&rot, weights.size*sizeof(float));

        int rotTds = 16; // 2d block -> 256 threads per thread block
        int block_height = (int)ceil((double)weights.dim(2) / (double)rotTds);
        int block_width = (int)ceil((double)weights.dim(3) / (double)rotTds);

        dim3 rotBlockDim(rotTds, rotTds); // You may adjust block dimensions according to your matrix size
        dim3 rotGridDim(block_width, block_height, weights.dim(0));
        rotate180<<<rotGridDim, rotBlockDim>>>(weights.data, rot, weights.d_dims);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
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
    block_depth = (int) ceil((double)out_width / (double)tds);

        // one thread per dl_dz_next element (excluding batch dim)
    // int x = blockIdx.x * blockDim.x + threadIdx.x;
    // int y = blockIdx.y * blockDim.y + threadIdx.y;
    // int in_ch = threadIdx.z;
    // int batch = blockIdx.z; // should be able to do this
    // float* aug_w, float* dldz_next, float* dldz, int* w_dims, int* next_dims, int* dz_dims, int padding, int strid
    dim3 threadDimOut(tds, tds, 1);
    dim3 blockDimOut(block_width, block_height, output_channels*batch_size); // output_channel = num weights

    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaDeviceSynchronize());
    conv_dldz_next<<<blockDimOut, threadDimOut>>>(tmp.data, output.data, weights.data, weights.d_dims, output.d_dims, tmp.d_dims, 0, 1, output_channels);
    // conv_forward <<<blockDimOut, threadDimOut>>>(tmp.data, output.data, weights.data, bias.data, tmp.d_dims, output.d_dims, weights.d_dims,0, 1, use_bias); // no padding argument
    CUDA_CHECK(cudaGetLastError()); // Ensure there's no previous kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(tmp.data);

    return output;
}

// __global__ void remove_pad(float* padded_in, float* out, int* in_dim, int* out_dim, int padding){
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     int in_ch = theadIdx.z;
//     int batch = blockIdx.z;

//     // we want to 

//     if(x >= in_dim[3]-padding || y >= in_dim[2]-padding || batch >= in_dim[0] || in_ch >= in_dim[1] || x < padding || y < padding){
//         return;
//     }

//     int unpad_x = x - padding;
//     int unpad_y = y-padding;

//     float* val = getElement(padded_in, in_dim, batch, in_ch, y, x);

//     int idx = getIdx(in_dim, batch, in_ch, unpad_y, unpad_x);

//     out[idx] = val;

// }


__global__ void apply_dw(float* weights, float* dw, int* w_dims, int n_filters){
    int x = blockIdx.z * blockDim.z + threadIdx.z; 
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int z_idx = blockIdx.x*blockDim.x + threadIdx.x; // we reserve x idx which can hold a lot of blocks for our longest dim
    int filter_idx = z_idx % n_filters;
    int channel = z_idx / n_filters;

    int idx = getIdx(w_dims, filter_n, filter_ch, y, x);

    weights[idx] -= 0.0001*dw[idx];
}

__global__ void apply_relu(float* dLdZ, float* input, int* in_dims){
    // 4d dLdZ because conv
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = theadIdx.z;
    int batch = blockIdx.z;

    int idx = getIdx(in_dims, batch, channel, y, x);
    float val = input[idx];

    if(val < 0){
        dLdZ[idx] = 0;
    }else{
        dLdZ[idx] = val;
    }
}

Tensor<float, 4> Conv2d::backward(Tensor<float,4> &dLdZ){

    if(this->use_relu){ // apply relu backwards
        int tds = 16;
        int block_height = (int)ceil(((double)input.dim(2)) / (double)tds);
        int block_width = (int)ceil(((double)input.dim(3)) / (double)tds);
        int batch_size = input.dim(0);
    
        dim3 threadDimRelu(tds, tds, input_channels);
        dim3 blockDimRelu(block_width, block_height, batch_size);
        apply_relu<<<blockDimRelu, threadDimRelu>>>(dLdZ.data, input.data, input.d_dims);
    }

    Tensor<float, 4> dLdZ_next = conv_transpose_2d_dldz(this->weights, dLdZ, this->padding, this->stride, true);
    Tensor<float, 4> dWdZ({output_channels, input_channels, filter_size, filter_size}, true, true);

    tds = 8; // 2d block -> 256 threads per thread block
    block_height = (int)ceil(((double)filter_size) / (double)tds);
    block_width = (int)ceil(((double)filter_size) / (double)tds);

    dim3 threadDim(3, tds, tds);
    dim3 blockDim(input_channels*output_channels, block_height, block_width);
    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaDeviceSynchronize());
    // sum and average weights across mini-batch before updating weights
    get_dw<<<blockDim, threadDim>>>(this->input.data, dLdZ.data, dWdZ.data, input.d_dims, dLdZ.d_dims, dWdZ.d_dims, padding, stride, output_channels);
    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaDeviceSynchronize());
    // cut out padding before returning to next layer
    assert(input.dim(0) == dLdZ.dim(0));
    assert(input.dim(1) == dLdZ.dim(1));
    assert(input.dim(2) == dLdZ_next.dim(2));
    assert(input.dim(3) == dLdZ_next.dim(3));


    // int batch_size = dLdZ.dim(0);
    // block_height = (int)ceil(((double)dLdZ.dim(2)) / (double)tds);  // one thread per padded input height width row 
    // block_width = (int)ceil(((double)dLdZ.dim(3)) / (double)tds);

    // dim3 padThreadDim(tds, tds, input_channels);
    // dim3 padBlockDim(block_width, block_height, batch_size);

    // Tensor<float, 4> unpad_dldz({input.dim(0), input.dim(1), input.dim(2), input.dim(3)}, true, true);
    // CUDA_CHECK(cudaGetLastError()); 
    // CUDA_CHECK(cudaDeviceSynchronize());
    // remove_pad<<padBlockDim, padThreadDim>>(dLdZ.data, unpad_dldz.data, dLdZ.d_dims, unpad_dldz.d_dims, padding);
    // CUDA_CHECK(cudaGetLastError()); 
    // CUDA_CHECK(cudaDeviceSynchronize());

    // dWdZ.toHost();
    // input.toHost();
    // for(int i=0; i<4; i++){
    //     printf(" %f ", dWdZ.data[i]);
    // }
    // printf("\n");
    // pass back 4D tensor with individual grads pre input

    tds = 8; // small thread count as filters are small but filter_n is big
    block_height = (int)ceil(((double)filter_size) / (double)tds);
    block_width = (int)ceil(((double)filter_size) / (double)tds);

    dim3 threadDimDw(3, tds, tds);
    dim3 blockDimDw( input_channels*output_channels, block_height, block_width); // move biggest dims onto block x as it can hold a fuck tonne of blocks

    apply_dw<<<blockDimDw, threadDimDw>>>(this->weights.data, dWdZ.data, this->weights.d_dims, output_channels);
    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaDeviceSynchronize());

    return dLdZ_next;

}