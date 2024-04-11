#include <cstdio>
#include <cassert>
#include <initializer_list>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iomanip> 

#pragma once

template<typename Scalar, size_t Rank>
class Tensor {
public:
    int dims[Rank];
    int* d_dims;
    float* data;
    size_t size;

    Tensor(std::initializer_list<int> dims, bool do_allocs = true, bool to_device=false) : size(calcSize(dims)) {
        assert(dims.size() == Rank);

		if (do_allocs) {
            if(to_device){
                cudaMalloc(&this->data, this->size * sizeof(float));
                // cudaMalloc(&this->dims, Rank * sizeof(float));
            } else {
			    this->data = (float *)malloc(this->size * sizeof(float));
                // this->dims = (int*)malloc(Rank*sizeof(int));
            }
		}

        auto iter = dims.begin();

		for (size_t i = 0; i < Rank; i++){
			this->dims[i] = *iter;
            iter++;
        }

        cudaMalloc(&this->d_dims, Rank * sizeof(int));
        cudaMemcpy(this->d_dims, this->dims, Rank * sizeof(int), cudaMemcpyHostToDevice);
    }

    Tensor() : size(0), data(nullptr) {}

    Tensor(int batch_size, int channels, int width, int height) :
        Tensor({ batch_size, channels, width, height }) {}

    Tensor(int n, int m, int p) :
        Tensor({ n, m, p }) {}

    Tensor(int n, int m) :
        Tensor({ n, m }) {}

    Tensor(int n) :
        Tensor({ n }) {}
    
    ~Tensor() {
        // if (this->data != nullptr) {
        //     free(this->data);
        // }
    }

        // Move tensor data to CUDA device memory
    __host__ __device__
    void toDevice() {
        float* d_data;
        cudaMalloc(&d_data, size * sizeof(float));
        cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);
        free(data);
        this->data = d_data;

        // int* tmp;
        // cudaMalloc(&tmp, Rank*sizeof(int));
        // cudaMemcpy(tmp, this->dims, Rank * sizeof(int), cudaMemcpyHostToDevice);

        // free(this->dims);
        // this->dims = tmp;

    }
    
    // Move tensor data back to host memory
    __host__ __device__
    void toHost() {
        float *new_data = (float*)malloc(size*sizeof(float));
        cudaMemcpy(new_data, this->data, this->size*sizeof(float), cudaMemcpyDeviceToHost);

        // cudaMemcpy(new_data, this->data, size*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(this->data);
        this->data = new_data;

        // int* tmp;
        // tmp = (int*)malloc( Rank*sizeof(float));
        // cudaMemcpy(tmp, this->dims, Rank * sizeof(float), cudaMemcpyDeviceToHost);

        // cudaFree(this->dims);
        // this->dims = tmp;
    }

    __host__ __device__ // allows both host and device to use same method for getting dims instead of separate device and host dim arrays
    int dim(int idx) const {
        assert(0 <= idx && idx < int(Rank));

        return dims[idx];
    }


    // Method to calculate the size of tensor based on dimensions
    __host__ __device__
    size_t calcSize(const std::initializer_list<int>& dims) const {
        size_t size = 1;
        for (int dim : dims)
            size *= dim;
        return size;
    }

    // Method to access elements using operator()
    __host__ __device__
    float& operator()(int i, int j, int k, int l) {
        assert(0 <= i && i < this->dim(0) ? true : (printf("i is out of bounds: %d max %d\n", i, this->dim(0)), false));
        assert(0 <= j && j < this->dim(1) ? true : (printf("j is out of bounds: %d max %d\n", j, this->dim(1)), false));
        assert(0 <= k && k < this->dim(2) ? true : (printf("k is out of bounds: %d max %d\n", k, this->dim(2)), false));
        assert(0 <= l && l < this->dim(3) ? true : (printf("l is out of bounds: %d max %d\n", l, this->dim(3)), false));


        assert(4 == Rank);
        size_t index =  l + dims[3] * (k + dims[2] * (j + dims[1] * (i)));
        return data[index];
    }

    __host__ __device__
    float& operator()(int i, int j, int k) {
        assert(0 <= i && i < this->dim(0));
		assert(0 <= j && j < this->dim(1));
		assert(0 <= k && k < this->dim(2));

        assert(3 == Rank);
        size_t index = i*dims[1]*dims[2] + j*dims[2] + k;
        return data[index];
    }

    __host__ __device__
    float& operator()(int i, int j) {
        assert(0 <= i && i < this->dim(0) ? true : (printf("i is out of bounds: %d max %d\n", i, this->dim(0)), false));
		assert(0 <= j && j < this->dim(1));

        assert(2 == Rank);
        size_t index = i*this->dims[1]+j;
        return data[index];
    }

    __host__ __device__
    float& operator()(int i) {
        assert(0 <= i && i < this->dim(0));
        assert(1 == Rank);

        return data[i];
    }


    // Method to print tensor data
    __host__ __device__
    void print() const {
        size_t sliceSize = size / this->dim(0);
        for (size_t i = 0; i < size; ++i) {
            std::cout << std::setw(2) << data[i] << " "; // Adjust setw as needed
            if ((i + 1) % sliceSize == 0 && i != 0) {
                std::cout << std::endl;
                if (i + 1 < size) { // Avoid printing a new line at the end
                    std::cout << "------------------------" << std::endl;
                }
            }
        }
        std::cout << std::endl;
    }
};

// int tensor_test() {
//     Tensor<int, 4> a(2, 3, 4, 5);

//     // Accessing elements
//     a(0, 0, 0, 1) = 10;

//     // Adding element to tensor positions
//     a.addElement(5, {0, 0, 0, 1});

//     a.print();


//     return 0;
// }
typedef struct {
    float* data;
    int* dims; // Assuming you have dimension information
} FloatArray;
