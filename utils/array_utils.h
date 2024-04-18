#include <cuda_runtime.h>
#include <cstdio>
#include <curand_kernel.h>
#include <random>
#include <ctime>

// #pragma once

__host__ __device__ inline float getElement(float *arr, int i) {
    return arr[i];
}

__host__ __device__ inline float getElement(float *arr, int* dims, int i, int j) {
    return arr[i * dims[1] + j];
}

__host__ __device__ inline float getElement(float *arr, int* dims, int i, int j, int k) {
    return arr[i * dims[1] * dims[2] + j * dims[2] + k];
}

__host__ __device__ inline float getElement(float *arr, int* dims, int i, int j, int k, int l) {
    return arr[i * dims[1] * dims[2] * dims[3] + j * dims[2] * dims[3] + k * dims[3] + l];
}

__host__ __device__ inline float getIdx(int* dims, int i, int j, int k, int l) {
    return i * dims[1] * dims[2] * dims[3] + j * dims[2] * dims[3] + k * dims[3] + l;
}


__host__ __device__ inline int getIdx(int* dims, int i, int j, int k) {
    return i * dims[1] * dims[2] + j * dims[2] + k;
}

__host__ __device__ inline int getIdx(int* dims, int i, int j) {
    return i * dims[1] + j;
}

__global__ inline  void fill_rand(float* arr, int size, unsigned long long seed, unsigned long long sequence_offset) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, tid, sequence_offset, &state);
    if (tid < size) {
        arr[tid] = 2.0f * curand_uniform(&state) - 1.0f;
    }
}

void inline fill_rand(float* arr, int size) {
    // Seed the random number generator
    std::mt19937 rng(std::time(nullptr));
    // Define a distribution for random numbers between -1 and 1
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Fill the array with random numbers
    for (int i = 0; i < size; ++i) {
        arr[i] = dist(rng);
        // arr[i] = 2*(i/size) - 1;
    }
}

__host__ __device__ inline float clip_to_range(float value, int factor=1) {
    if (value < (-1.0f*factor)) {
        return -1.0f*factor;
    } else if (value > (1.0f*factor)) {
        return 1.0f*factor;
    } else {
        return value;
    }
}