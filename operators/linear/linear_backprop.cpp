#include "Linear.h"
#include "../../utils/Tensor.h"
#include <cstdlib> // for rand() function
#include <ctime>   // for seeding rand() function
#include <iostream> // for cout
#include <iomanip> // for setw
#include <cmath> // for round
#include <chrono>

#pragma once

Tensor<float, 2> linear_cpu(Tensor<float, 2> input, Tensor<float, 2> weights, Tensor<float, 1> bias, bool use_bias) {
    // Ensure dimensions match for matrix multiplication
    assert(input.dim(1) == weights.dim(1));

    int batch_size = input.dim(0);
    int output_size = weights.dim(0);

    // Allocate memory for the output tensor
    Tensor<float, 2> output({batch_size, output_size}, true);

    // Perform matrix multiplication
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            float sum = use_bias ? bias(j) : 0;
            // printf("bias[%d] %f ", j, bias(j));
            for (int k = 0; k < input.dim(1); ++k) {
                sum += input(i, k) * weights(j, k);
            }
            output(i, j) = sum;
        }
    }

    return output;
}


// Function to generate random float between -1 and 1
float rand_float() {
    return (2.0f * rand() / RAND_MAX) - 1.0f;
}

/*
CPU time: 439.001 seconds
GPU time: 11.5202 seconds
*/
int main(){
    // Seed the random number generator
    srand(time(0));

    // Define the dimensions of the tensor
    int batch_size = 1;
    int input_size = 4;
    int output_size = 3;

    // Create the tensor
    Tensor<float, 2> input({batch_size, input_size}, true);
    Tensor<float, 2> dLdZ({batch_size, output_size}, true);
    Linear linear(input_size, output_size, true, false); // input_size, output_size, use_bias
    linear.weights.toHost();

    // Fill the tensor with random values between -1 and 1
    float in[4] = {1,3,2,4};
    float weights[12] = {-1, 0.5, 2, -0.5, 3, -2, 1, 2, 4, 3, 5, -1};
    float dz[3] = {0.088, -0.75, 0.66};

    for(int i=0; i<4; i++){
        input.data[i] = in[i];
    }
    for(int i=0; i<3; i++){
        dLdZ.data[i] = dz[i];
    }
    for(int i=0; i<12; i++){
        linear.weights.data[i] = weights[i];
    }

    linear.weights.toDevice();

    // move data to gpu
    linear.weights.toDevice();
    linear.bias.toDevice();
    input.toDevice();
    dLdZ.toDevice();

    auto out_forward = linear.forward(input);

    out_forward.toHost();
    float ans[3] = {2.5, 7, 19};
    for(int i=0; i<3; i++){
        printf("%f ", out_forward.data[i]);
        assert(out_forward.data[i] - ans[i] < 1);
    }
    // Time GPU calculation
    std::cout << "GPU calculation" << std::endl;
    auto start_gpu = std::chrono::steady_clock::now();
    Tensor<float, 2> output = linear.backward(dLdZ);
    auto end_gpu = std::chrono::steady_clock::now();
    std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;
    std::cout << "GPU time: " << gpu_duration.count() << " seconds" << std::endl;

    output.toHost();
    output.print();
    
    // std::cout << "Checking results" << std::endl;

    // assert((output.dim(0) == out_cpu.dim(0)) ? true : (printf("Mismatch in dimension 0: %d vs %d\n", output.dim(0), 5), false));
    // assert((output.dim(1) == out_cpu.dim(1)) ? true : (printf("Mismatch in dimension 1: %d vs %d\n", output.dim(1), 3), false));

    // for (int b = 0; b < batch_size; ++b) {
    //     for (int c = 0; c < output_size; ++c) {
    //         assert((output(b, c) - out_cpu(b, c)) < 0.0001 ? true : 
    //            (printf("Mismatch at (b=%d, c=%d): %f vs %f\n", b, c, output(b, c), out_cpu(b, c)), false));
    //     }
    // }

    printf("passed!\n");

    return 0;
}
