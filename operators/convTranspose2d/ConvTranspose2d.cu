#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <cassert>
#include <initializer_list>
#include <utility>
#include <math.h>
#include "../../utils/array_utils.h"

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



__global__ void conv_forward(float* input, float* output, float* weights, float* bias, int* in_dims, int* out_dims, int* w_dims, int padding, int stride, bool use_bias = true) {
    int x = blockIdx.z * blockDim.z + threadIdx.z; 
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int output_channels = out_dims[1];

    int z_idx = blockIdx.x*blockDim.x + threadIdx.x; // we reserve x idx which can hold a lot of blocks for our longest dim
    int c_out = z_idx % output_channels;
    int b = z_idx / output_channels;
    
    int batch_size = in_dims[0],
       input_channels = in_dims[1],
       height = in_dims[2],
       width = in_dims[3],
       filter_size = w_dims[2];

    if (x >= width+padding || y >= height+padding || x+filter_size-1 >= width+padding || y+filter_size-1 >= height+padding || c_out >= output_channels || (x%stride) != 0 || (y%stride) != 0 || b >= batch_size){
        return;
    }

    float bias_val = use_bias ? bias[c_out] : 0;

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
    output[idx] = sum;
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

ie for expected behaviour with hard to code transpose we need to rotate weights 180 for this op for weights to be same position as in normal complex transpose

!!!!!!!!!!
for backprop we can rotate weights 180 not not rotate during forward or we can do vice versa it shouldnt matter
might be smarter to do it on backprop for faster inference however its easier for me to debug/conceptualize in forward pass
!!!!!!!!!!!!
*/
Tensor<float, 4> ConvTranspose2d::forward(Tensor<float,4> &input){

    assert(input.dim(1) == this->input_channels);
    // free(input.data);
    this->input.data = input.data;

    int batch_size = input.dim(0), in_channels = input.dim(1), height = input.dim(2), width = input.dim(3);
    int output_channels = weights.dim(0);

    float* rot;
    cudaMalloc(&rot, weights.size*sizeof(float));

    int rotTds = 16; // 2d block -> 256 threads per thread block
    int block_height = (int)ceil((double)weights.dim(2) / (double)rotTds);
    int block_width = (int)ceil((double)weights.dim(3) / (double)rotTds);

    dim3 rotBlockDim(rotTds, rotTds); // You may adjust block dimensions according to your matrix size
    dim3 rotGridDim(block_width, block_height, weights.dim(0));
    rotate180<<<rotGridDim, rotBlockDim>>>(weights.data, rot, weights.d_dims);
    CUDA_CHECK(cudaGetLastError()); // Ensure there's no previous kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize());
    // weights.data = rot;
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

    dim3 threadDimOut(3, tds, tds);
    dim3 blockDimOut(output_channels*batch_size, block_height,block_width ); // output_channel = num weights

    checkMemoryLocation(tmp.data);
    checkMemoryLocation(output.data);
    checkMemoryLocation(weights.data);
    checkMemoryLocation(bias.data);
    CUDA_CHECK(cudaGetLastError()); // Ensure there's no previous kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize());
    conv_forward <<<blockDimOut, threadDimOut>>>(tmp.data, output.data, rot, bias.data, tmp.d_dims, output.d_dims, weights.d_dims,0, 1); // no padding argument, rot = weights.data
    CUDA_CHECK(cudaGetLastError()); // Ensure there's no previous kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize());
    // cudaFree(tmp.data);


    return output;
}

// Tensor<float, 4> conv_transpose_2d(Tensor<float,4> &input, Tensor<float, 4> weights, Tensor<float, 1> bias, int padding, int stride, bool rotate180_weights){

//     if(rotate180_weights){
//         float* rot;
//         cudaMalloc(&rot, weights.size*sizeof(float));

//         int rotTds = 16; // 2d block -> 256 threads per thread block
//         int block_height = (int)ceil((double)weights.dim(2) / (double)rotTds);
//         int block_width = (int)ceil((double)weights.dim(3) / (double)rotTds);
//         checkMemoryLocation(weights.data);
//         checkMemoryLocation(rot);

//         dim3 rotBlockDim(rotTds, rotTds); // You may adjust block dimensions according to your matrix size
//         dim3 rotGridDim(block_width, block_height, weights.dim(0));
//         rotate180<<<rotGridDim, rotBlockDim>>>(weights.data, rot, weights.d_dims);
//         cudaDeviceSynchronize();
//         weights.data = rot;
//     }

//     int batch_size = input.dim(0), in_channels = input.dim(1), height = input.dim(2), width = input.dim(3), filter_size = weights.dim(2);
//     int output_channels = weights.dim(0);
//     // insert z zeros between input vals and pp zeros around edge
//     int z = stride-1;
//     int pp = filter_size-padding-1;

//     // create temp input to perform conv on because I dont want to do weird atomic add transpose thing
//     int h_tmp = 2*pp+z*(height-1) + height;
//     int w_tmp = 2*pp+z*(width-1) + width;

//     // create and pad tmp array
//     Tensor<float, 4> tmp({batch_size, in_channels, h_tmp, w_tmp}, true, true);

//     int tds = 16; // 2d block -> 256 threads per thread block
//     int block_height = (int)ceil(((double)h_tmp) / (double)tds);
//     int block_width = (int)ceil(((double)w_tmp) / (double)tds);

//     dim3 threadDim(tds, tds, 1);
//     dim3 blockDim(block_width, block_height, batch_size);

//     checkMemoryLocation(input.data);
//     checkMemoryLocation(tmp.data);
//     checkMemoryLocation(tmp.d_dims);
//     checkMemoryLocation(tmp.d_dims);
//     CUDA_CHECK(cudaGetLastError()); // Ensure there's no previous kernel launch errors
//     CUDA_CHECK(cudaDeviceSynchronize());

//     pad_image_tranpose<<<blockDim, threadDim>>>(input.data, tmp.data, input.d_dims, tmp.d_dims, z, pp);

//     CUDA_CHECK(cudaGetLastError()); // Ensure there's no previous kernel launch errors
//     CUDA_CHECK(cudaDeviceSynchronize());
//     // create output array for convolution
//     int out_height = (height-1) * stride + filter_size-2*padding;
//     int out_width = (width-1) * stride + filter_size-2*padding;

//     Tensor<float, 4> output({batch_size, output_channels, out_height, out_width}, false); // output for actual convolution
//     cudaMalloc(&output.data, output.size*sizeof(float));

//     tds = 16; // 2d block -> 256 threads per thread block
//     block_height = (int) ceil((double)out_height / (double)tds);
//     block_width = (int) ceil((double)out_width / (double)tds);

//     dim3 threadDimOut(tds, tds, 1);
//     dim3 blockDimOut(block_width, block_height, output_channels); // output_channel = num weights

//     checkMemoryLocation(tmp.data);
//     checkMemoryLocation(output.data);
//     checkMemoryLocation(weights.data);
//     checkMemoryLocation(bias.data);
//     CUDA_CHECK(cudaGetLastError()); // Ensure there's no previous kernel launch errors
//     CUDA_CHECK(cudaDeviceSynchronize());
//     conv_forward <<<blockDimOut, threadDimOut>>>(tmp.data, output.data, weights.data, bias.data, tmp.d_dims, output.d_dims, weights.d_dims,0, 1); // no padding argument
//     CUDA_CHECK(cudaGetLastError()); // Ensure there's no previous kernel launch errors
//     CUDA_CHECK(cudaDeviceSynchronize());
//     cudaFree(tmp.data);

//     return output;
// }

// backward stuff
__global__ void apply_dw(float* weights, float* dw, int* w_dims, int n_filters){
    int x = blockIdx.z * blockDim.z + threadIdx.z; 
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int z_idx = blockIdx.x*blockDim.x + threadIdx.x; // we reserve x idx which can hold a lot of blocks for our longest dim
    int filter_idx = z_idx % n_filters;
    int channel = z_idx / n_filters;

    int idx = getIdx(w_dims, filter_n, filter_ch, y, x);

    weights[idx] -= 0.000001*dw[idx];
}

__global__ void get_dwdz(float* dLdZ, float* dWdZ, float* input, int* dz_dims, int* dw_dims, int* in_dims, int padding, int stride, int n_filters) {
    // padding to thing we convolve over ie dl_dz
    // 3d input conv dldz for every dldz in batch
    // one thread per dWdZ element
    int x = blockIdx.z * blockDim.z + threadIdx.z; 
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int z_idx = blockIdx.x*blockDim.x + threadIdx.x; // we reserve x idx which can hold a lot of blocks for our longest dim
    int filter_idx = z_idx % n_filters;
    int channel = z_idx / n_filters;

    int batch_size = in_dims[0],
       input_channels = in_dims[1],
       height = dz_dims[2],
       width = dz_dims[3],
       filter_size = in_dims[2],
       n_filters = dLdZ[1];

    if (x >= width+padding || y >= height+padding || x+filter_size-1 >= width+padding || y+filter_size-1 >= height+padding || c_out >= output_channels || (x%stride) != 0 || (y%stride) != 0){
        return;
    }
    
    float sum = 0;
    for(int b=0; b < batch_size; b++){
        for(int row=y; row<y+filter_size; row++){
            for(int col=x; col<x+filter_size; col++){
                if(row < padding || col < padding || row >= height+padding || col >= width+padding ){ // padding guard
                    continue;
                } else{
                    sum += getElement(dLdZ, dz_dims, b, filter_idx, row - padding, col - padding) * getElement(input, in_dims, b, filter_ch, row - y, col - x);
                }
            }
        }
    }
    int idx = getIdx(dw_dims, filter_idx, filter_ch, y/stride, x/stride);
    dWdZ[idx] += sum / batch_size;
}

__global__ void get_dldz_next(float* dLdZ, float* weights, float* dl_dz_next, int* dz_dims, int* w_dims, int* next_dims, int padding, int stride, int n_filters) {
    // padding to thing we convolve over ie dl_dz
    // one thread per output element 
    int x = blockIdx.z * blockDim.z + threadIdx.z; 
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int z_idx = blockIdx.x*blockDim.x + threadIdx.x; // we reserve x idx which can hold a lot of blocks for our longest dim
    int filter_idx = z_idx % n_filters;
    int channel = z_idx / n_filters;

    int batch_size = dl_dz_next[0],
       input_channels = dl_dz_next[1],
       height = dLdZ[2],
       width = dLdZ[3],
       filter_size = w_dims[2],
       n_filters = w_dims[0];

    if (x >= width+padding || y >= height+padding || x+filter_size-1 >= width+padding || y+filter_size-1 >= height+padding || filter_ch >= input_channels || (x%stride) != 0 || (y%stride) != 0){
        return;
    }

    for(int b=0; b < batch_size; b++){
        float sum = 0;
        for(int row=y; row<y+filter_size; row++){
            for(int col=x; col<x+filter_size; col++){
                if(row < padding || col < padding || row >= height+padding || col >= width+padding ){ // padding guard
                    continue;
                } else{
                    sum += getElement(dLdZ, dz_dims, b, filter_idx, row - padding, col - padding) * getElement(weights, w_dims, filter_num, filter_ch, row - y, col - x);
                }
            }
        }
        int idx = getIdx(next_dims, b, filter_ch, y/stride, x/stride);
        dl_dz_next[idx] += sum;
    }
}

Tensor<float, 4> ConvTranspose2d::backward(Tensor<float,4> &dLdZ){
    // will just implement it as backward conv bc idk how to do it using convolutions
    int input_w = input.dim(3),
        input_h = input.dim(2),
        input_channels = input.dim(1),
        n_filters = this->weights.dim(0);

    Tensor<float, 4> dLdZ_next({input.dim(0), input.dim(1), input_h, input_w}, true, true); // should be same shape as the input
    
    int tds = 16;
    int block_height = (int)ceil(((double)input_h) / (double)tds);
    int block_width = (int)ceil(((double)input_w) / (double)tds);

    dim3 threadDim(3, tds, tds);
    dim3 blockDim(n_filters*input_channels, block_height, block_width);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    get_dldz_next<<<blockDim, threadDim>>>(dLdZ.data, this->weights.data, dLdZ_next.data, dLdZ.d_dims, this->weights.d_dims, dLdZ_next.d_dims, this->padding, this->stride, n_filters); 
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Tensor<float, 4> dWdZ({n_filters, input_channels, weights.dim(2), weights.dim(3)}, true, true);
    
    tds = 4; // filter width is max 4 so why not save threads
    block_height = (int)ceil(((double)weights.dim(2)) / (double)tds);
    block_width = (int)ceil(((double)weights.dim(3)) / (double)tds);
    block_z = (int)ceil(((double) n_filters*input_channels) / (double)4);

    dim3 threadDimDw(3, tds, tds); // max out z dims as conv filter width/height shouldnt be over 4 
    dim3 blockDimDw(block_z, block_height, block_width );
    
    get_dwdz<<<blockDimDw, threadDimDw>>>(dLdZ.data, dWdZ.data, this->input, dLdZ.d_dims, this->weights.d_dims, this->input.d_dims, this->padding, this->stride, n_filters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    apply_dw<<<blockDimDw, threadDimDw>>>(this->weights.data, dWdZ.data, this->weights.d_dims);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return dLdZ_next;

}