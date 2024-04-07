#include "ConvTranspose2d.h"
#include "../../utils/Tensor.h"
#include <cstdlib> // for rand() function
#include <ctime>   // for seeding rand() function
#include <iostream> // for cout
#include <iomanip> // for setw
#include <cmath> // for round
#include <chrono>

#pragma once

Tensor<float, 4> conv2d_cpu(Tensor<float, 4> input, Tensor<float, 4> kernel, int padding = 1, int stride = 1) {
    const int batch_size = input.dim(0);
    const int in_height = input.dim(2);
    const int in_width = input.dim(3);
    const int in_channels = input.dim(1);

    const int kernel_height = kernel.dim(1);
    const int kernel_width = kernel.dim(2);
    const int out_channels = kernel.dim(0);

    int padded_height = in_height + 2 * padding;
    int padded_width = in_width + 2 * padding;
    int out_height = (padded_height - kernel_height) / stride + 1;
    int out_width = (padded_width - kernel_width) / stride + 1;

    Tensor<float, 4> output(batch_size, out_channels, out_height, out_width );

    // Iterate over batches
    for (int b = 0; b < batch_size; ++b) {
        // Iterate over output height
        for (int i = 0; i < out_height; ++i) {
            // Iterate over output width
            for (int j = 0; j < out_width; ++j) {
                // Iterate over output channels
                for (int c = 0; c < out_channels; ++c) {
                    float sum = 0.0f;
                    // Iterate over kernel height
                    for (int kh = 0; kh < kernel_height; ++kh) {
                        // Iterate over kernel width
                        for (int kw = 0; kw < kernel_width; ++kw) {
                            // Iterate over input channels
                            for (int ic = 0; ic < in_channels; ++ic) {
                                int input_row = i * stride + kh - padding;
                                int input_col = j * stride + kw - padding;
                                // Check if the input indices are within bounds
                                if (input_row >= 0 && input_row < in_height && input_col >= 0 && input_col < in_width) {
                                    sum += input(b, ic, input_row, input_col) * kernel(c, ic, kh, kw);
                                }
                            }
                        }
                    }
                    output(b, c, i, j) = sum;
                }
            }
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
    int batch_size = 5;
    int channels = 1;
    int height = 2;
    int width = 2;
    int filter_size = 2;

    // Create the tensor
    Tensor<float, 4> input(batch_size, channels, height, width);
    Tensor<float, 4> filters(channels, channels, filter_size, filter_size);
    ConvTranspose2d conv(channels, channels, filter_size, 0, 1); // in_channel, out_channel, filter_size, padding, stride

    // Fill the tensor with random values between -1 and 1
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
        //     for (int h = 0; h < height; ++h) {
        //         for (int w = 0; w < width; ++w) {
        //             input(b, c, h, w) = rand_float();
        //         }
        //     }
            input(b, c, 0, 0) = 1;
            input(b, c, 0, 1) = 2;
            input(b, c, 1, 0) = 3;
            input(b, c, 1, 1) = 4;

        }
    }

    for (int b = 0; b < channels; ++b) {
        for (int c = 0; c < channels; ++c) {
            // for (int h = 0; h < filter_size; ++h) {
            //     for (int w = 0; w < filter_size; ++w) {
            //         filters(b, c, h, w) = rand_float();
            //     }
            // }
            filters(b, c, 0, 0) = 1;
            filters(b, c, 0, 1) = 2;
            filters(b, c, 1, 0) = 3;
            filters(b, c, 1, 1) = 4;

        }
        conv.bias(b) = 0;//rand_float();
    }

    conv.weights = filters;

    // std::cout << "CPU calculation" << std::endl;
    // auto start_cpu = std::chrono::steady_clock::now();
    // Tensor<float, 4> out_cpu = conv2d_cpu(input, filters);
    // auto end_cpu = std::chrono::steady_clock::now();
    // std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    // std::cout << "CPU time: " << cpu_duration.count() << " seconds" << std::endl;

    input.toDevice();
    conv.weights.toDevice();
    conv.bias.toDevice();

    // Time GPU calculation
    std::cout << "GPU calculation" << std::endl;
    auto start_gpu = std::chrono::steady_clock::now();
    // Tensor<float, 4> output = conv.forward(input);
    Tensor<float, 4> output = conv_transpose_2d(input, conv.weights,conv.bias, 0, 1, true);
    auto end_gpu = std::chrono::steady_clock::now();
    std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;
    std::cout << "GPU time: " << gpu_duration.count() << " seconds" << std::endl;


    output.toHost();
    // should be [5, 3, 24, 40]
    // printf("%d %d %d %d\n", output.dim(0), output.dim(1), output.dim(2), output.dim(3));
    // for(int i=0; i<output.size; i++){
    //     printf("outidx %d sum %f\n", i, output.data[i]);
    // }

    std::cout << "Checking results" << std::endl;
    // assert((output.dim(0) == 5) ? true : (printf("Mismatch in dimension 0: %d vs %d\n", output.dim(0), 5), false));
    // assert((output.dim(1) == 3) ? true : (printf("Mismatch in dimension 1: %d vs %d\n", output.dim(1), 3), false));
    // assert((output.dim(2) == 24) ? true : (printf("Mismatch in dimension 2: %d vs %d\n", output.dim(2), 24), false));
    // assert((output.dim(3) == 40) ? true : (printf("Mismatch in dimension 3: %d vs %d\n", output.dim(3), 40), false));

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < output.dim(1); ++c) {
            for (int h = 0; h < output.dim(2); ++h) {
                for (int w = 0; w < output.dim(3); ++w) {
                    assert(output(b, c, h, w) != 0.0);          
                }
            }
        }
    }

    printf("%d %d %d %d\n", output.dim(0),output.dim(1),output.dim(2),output.dim(3));
    
    int ans[9] = {1,4,4,6,20,16,9,24,16};
    for(int b=0; b<batch_size; b++){
        assert(output(b,0,0,0) == ans[0]);
        assert(output(b,0,0,1) == ans[1]);
        assert(output(b,0,0,2) == ans[2]);
        assert(output(b,0,1,0) == ans[3]);
        assert(output(b,0,1,1) == ans[4]);
        assert(output(b,0,1,2) == ans[5]);
        assert(output(b,0,2,0) == ans[6]);
        assert(output(b,0,2,1) == ans[7]);
        assert(output(b,0,2,2) == ans[8]);
    }

    printf("passed!\n");

    // Print out a sample of the tensor values
    // for (int b = 0; b < 1; ++b) {
    //     for (int c = 0; c < 1; ++c) {
    //         for (int h = 0; h < 2; ++h) {
    //             for (int w = 0; w < 2; ++w) {
    //                 std::cout << std::fixed << std::setprecision(3) << std::setw(8) << input(b, c, h, w) << " ";
    //             }
    //             std::cout << std::endl;
    //         }
    //     }
    // }

    return 0;
}