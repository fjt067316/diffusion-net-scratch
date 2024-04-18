#include <cstdio>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cassert>
#include <initializer_list>
#include <utility>
#include <cmath>
#include "../../utils/array_utils.h"

#include "Linear.h"
#include <curand_kernel.h>

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

Linear::Linear(int input_size, int output_size, bool use_bias, bool use_relu) : 
    input_size(input_size), 
    output_size(output_size), 
    weights({output_size, input_size}, true, true),
    use_bias(use_bias),
    use_relu(use_relu)
    {
        unsigned long long seed = 123456; // Change this to any desired seed
        unsigned long long sequence_offset = 0;

        if (use_bias){
            bias = Tensor<float, 1>({output_size}, true, true);
            // int grid = (int) ceil((double)(bias.size) / 256);
            // fill_rand<<<grid, 256>>>(bias.data, bias.size, seed, sequence_offset);
            float bias_vals[bias.size];
            fill_rand(bias_vals, bias.size);
            cudaMemcpy(bias.data, bias_vals, bias.size*sizeof(float), cudaMemcpyHostToDevice);
        }else{
            bias = Tensor<float, 1>({output_size}, false);
        }

        // fill weights and bias with random numbers
        int grid = (int) ceil((double)(weights.size) / 256);
        fill_rand<<<grid, 256>>>(weights.data, weights.size, seed, sequence_offset);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        // fill_rand(init_vals, weights.size);
        // cudaMemcpy(weights.data, init_vals, weights.size*sizeof(float), cudaMemcpyHostToDevice); // (1)

    }

__global__ void linear_forward(float* input, float* output, float* weights, float* bias, bool use_bias, bool use_relu, int* in_dims, int* out_dims, int* w_dims, int* bias_dims)
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
    // printf("bias %f in_rows %d ", sum, in_rows);

    for(int i=0; i<in_rows; i++){
        sum += getElement(input, in_dims, batch, i) * getElement(weights, w_dims, row, i);
    }
    if(use_relu && (sum < 0)){
        output[batch * out_size + row] = 0;
    } else {
        // printf("row %d sum %f", row, sum);
        output[batch * out_size + row] = sum;
    }
}


// assumes data already on gpu
Tensor<float, 2> Linear::forward(Tensor<float,2> &input){

    int batch_size = input.dim(0);

    assert(this->input_size == input.dim(1));
    this->input = input;
    int tds = 16; 
    int blocks = (int) ceil((double)this->output_size / tds);

    dim3 threadDim(tds, batch_size); // one thread per row of Linear layer and batch size
    dim3 blockDim(blocks, 1);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Tensor<float, 2> output({batch_size, this->output_size}, true, true); // do_alloc = true, to_device = true

    // checkMemoryLocation(input.data);
    // checkMemoryLocation(output.data);
    // checkMemoryLocation(weights.data);
    // checkMemoryLocation(bias.data);

    // checkMemoryLocation(input.d_dims);
    // checkMemoryLocation(weights.d_dims);

    linear_forward <<<blockDim, threadDim>>>(input.data, output.data, weights.data, bias.data, this->use_bias, this->use_relu, input.d_dims, output.d_dims, weights.d_dims, bias.d_dims);

    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to finish

    if(use_relu){
        this->output = output;
    }

    return output;

}

__global__ void get_dw(float* dw, float* dz, float* in, int* w_dims, int* dz_dims, int* in_dims){
    int x = blockIdx.x*blockDim.x + threadIdx.x; // x = dw col
    int y = blockIdx.y*blockDim.y+threadIdx.y; // y = dw row
    
    if(y >= dz_dims[1] || x >= in_dims[1]){
        return;
    }

    int batch_size = in_dims[0];
    float sum = 0;
    int in_size = in_dims[1];
    int dz_size = dz_dims[1];
    for(int b=0; b<batch_size; b++){
        // sum += out[row]*input[batch][col]
        // assert(in_size*b+x < in_size*batch_size);
        float input_val = in[in_size*b+x];
        float dz_val = dz[b*dz_size+y];
        sum += input_val*dz_val;
    }

    int idx = getIdx(w_dims, y, x);
    dw[idx] = sum/batch_size;
}

__global__ void get_dldz_next(float* dz_next, float* dz, float* weights, int* next_dims, int* dz_dims, int* w_dims){
    int in_col = blockIdx.x*blockDim.x+threadIdx.x;
    int batch = blockIdx.y;

    if(in_col >= w_dims[1] || batch >= dz_dims[0]){
        return;
    }

    int w_rows = w_dims[0];
    int w_row_size = w_dims[1];

    int dz_batch_size = dz_dims[1];

    float sum = 0;
    for(int i=0; i<w_rows; i++){
        float w = weights[i*w_row_size + in_col]; // coalesce this
        float dz_val = dz[batch*dz_batch_size+i];

        sum += w*dz_val;
    }

    dz_next[w_row_size*batch+in_col] = sum; // dz_next[b][in_col] = w*dz
}

__global__ void apply_dw(float* dw, float* weights, int* w_dims){
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;

    if(x >= w_dims[1] || y >= w_dims[0]){
        return;
    }

    int row_size = w_dims[1];
    weights[y*row_size + x] -= 0.01*clip_to_range(dw[y*row_size+x]);
    weights[y*row_size + x] = clip_to_range(weights[y*row_size+x], 10);
}

__global__ void apply_relu_back(float* dLdZ, float* output, int* dz_dims){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.y;

    if(x >= dz_dims[1] || batch >= dz_dims[0]){
        return;
    }

    int idx = getIdx(dz_dims, batch, x);
    float val = output[idx];

    if(val <= 0){ // should just be == 0 not <= 0 but whatevs
        dLdZ[idx] = 0;
    }
}

Tensor<float, 2> Linear::backward(Tensor<float,2> &dLdZ){
    int batch_size = dLdZ.dim(0);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if(this->use_relu){
        int tds = 32;
        int blocks_h = (int)ceil(((double)dLdZ.dim(1)) / (double)tds);
        // int blocks_w = (int)ceil(((double)dLdZ.dim(2)) / (double)tds);
    
        dim3 threadDimRelu(tds);
        dim3 blockDimRelu(blocks_h, batch_size);
        apply_relu_back<<<blockDimRelu, threadDimRelu>>>(dLdZ.data, output.data, output.d_dims);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    Tensor<float, 2> dw({weights.dim(0), weights.dim(1)}, true, true);
    Tensor<float, 1> db({bias.dim(0)}, true, true);

    // one thread per dw
    int tds = 16; 
    int blocks_w = (int) ceil((double)this->weights.dim(1) / tds);
    int blocks_h = (int) ceil((double)this->weights.dim(0) / tds);

    dim3 threadDim(tds, tds); // one thread per row of Linear layer and batch size
    dim3 blockDim(blocks_w, blocks_h);
    assert(dLdZ.dim(0) == input.dim(0));
    assert(dLdZ.dim(1) == dw.dim(0));
    get_dw<<<blockDim, threadDim>>>(dw.data, dLdZ.data, this->input.data, dw.d_dims, dLdZ.d_dims, this->input.d_dims);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // dw.toHost();
    // dw.print();
    // dw.toDevice();
    tds = 512; 
    blocks_w = (int) ceil((double)this->weights.dim(1) / tds);

    dim3 threadDimDz(tds, 1); // one thread per row of Linear layer and batch size
    dim3 blockDimDz(blocks_w, batch_size);

    Tensor<float, 2> dLdZ_next({this->input.dim(0), input.dim(1)}, true, true);
    get_dldz_next<<<blockDimDz, threadDimDz>>>(dLdZ_next.data, dLdZ.data, weights.data, dLdZ_next.d_dims, dLdZ.d_dims, weights.d_dims);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // apply dw
    tds = 16; 
    blocks_w = (int) ceil((double)this->weights.dim(1) / tds);
    blocks_h = (int) ceil((double)this->weights.dim(0) / tds);

    dim3 threadDimDw(tds, tds); // one thread per row of Linear layer and batch size
    dim3 blockDimDw(blocks_w, batch_size);

    apply_dw<<<blockDimDw, threadDimDw>>>(dw.data, weights.data, weights.d_dims);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // cudaFree(input.data);
    // cudaFree(output.data);
    return dLdZ_next;
}