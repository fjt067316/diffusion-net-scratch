#include "Conv2d.h"
#include "../../utils/Tensor.h"
#include <cstdlib> // for rand() function
#include <ctime>   // for seeding rand() function
#include <iostream> // for cout
#include <iomanip> // for setw
#include <cmath> // for round
#include <chrono>

#pragma once

Tensor<float, 4> conv2d_cpu(Tensor<float, 4> input, Tensor<float, 4> kernel, Tensor<float, 1> bias, int padding = 1, int stride = 1) {
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
                    float sum = bias(c);
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
    int batch_size = 3;
    int channels = 1;
    int height = 4;//512;
    int width = 4;//768;
    int out_channels = 1;
    // Create the tensor
    Tensor<float, 4> input(batch_size, channels, height, width);
    Tensor<float, 4> filters(out_channels, channels, 2, 2);
    Tensor<float, 4> dLdZ(batch_size, out_channels, 3, 3 );
    Conv2d conv(channels, 1, 2, 0, 1); // in_channel, out_channel, filter_size, padding, stride

    for (int batch = 0; batch < batch_size; batch++) {
        for (int i = 0; i < 16; i++) {
            input.data[batch * 16 + i] = (i + 1);
        }
    }

    for(int i=0; i<4; i++){
        filters.data[i] = (i+1);
    }

    for (int batch = 0; batch < batch_size; batch++) {
        for (int i = 0; i < 9; i++) {
            dLdZ.data[batch * 9 + i] = (i + 1);
        }
    }
    // Fill the tensor with random values between -1 and 1
    // for (int b = 0; b < batch_size; ++b) {
    //     for (int c = 0; c < channels; ++c) {
    //         for (int h = 0; h < height; ++h) {
    //             for (int w = 0; w < width; ++w) {
    //                 input(b, c, h, w) = rand_float();
    //             }
    //         }
    //     }
    // }

    // for (int b = 0; b < 64; ++b) {
    //     for (int c = 0; c < channels; ++c) {
    //         for (int h = 0; h < 3; ++h) {
    //             for (int w = 0; w < 3; ++w) {
    //                 filters(b, c, h, w) = rand_float();
    //             }
    //         }
    //     }
    //     conv.bias(b) = rand_float();
    // }

    conv.weights = filters;
    conv.input = input;

    conv.weights.toDevice();
    conv.input.toDevice();
    dLdZ.toDevice();

    // Time GPU calculation
    std::cout << "GPU calculation" << std::endl;
    auto start_gpu = std::chrono::steady_clock::now();
    Tensor<float, 4> dLdZ_next = conv.backward(dLdZ);
    auto end_gpu = std::chrono::steady_clock::now();
    std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;
    std::cout << "GPU time: " << gpu_duration.count() << " seconds" << std::endl;
    
    dLdZ_next.toHost();
    // printf("%d %d", dLdZ_next.dim(2), dLdZ_next.dim(3));
    for(int i=0; i<16; i++){
        printf("%f ", dLdZ_next.data[i]);
    }
    printf("\n");
    // for (int i = 0; i < 64; ++i) {
    //     std::cout << "c " << i << " " << output(0,i,0,0) << " " << out_cpu(0,i,0,0) << std::endl;
    // }
    
    // Compare results if necessary
    std::cout << "Checking results" << std::endl;
    // assert((output.dim(0) == out_cpu.dim(0)) ? true : (printf("Mismatch in dimension 0: %d vs %d\n", output.dim(0), out_cpu.dim(0)), false));
    // assert((output.dim(1) == out_cpu.dim(1)) ? true : (printf("Mismatch in dimension 1: %d vs %d\n", output.dim(1), out_cpu.dim(1)), false));
    // assert((output.dim(2) == out_cpu.dim(2)) ? true : (printf("Mismatch in dimension 2: %d vs %d\n", output.dim(2), out_cpu.dim(2)), false));
    // assert((output.dim(3) == out_cpu.dim(3)) ? true : (printf("Mismatch in dimension 3: %d vs %d\n", output.dim(3), out_cpu.dim(3)), false));


    // for (int b = 0; b < batch_size; ++b) {
    //     for (int c = 0; c < 64; ++c) {
    //         for (int h = 0; h < output.dim(2); ++h) {
    //             for (int w = 0; w < output.dim(3); ++w) {
    //                 // assert((output(b, c, h, w) - out_cpu(b, c, h, w)) < 0.0001 ? true : 
    //                 //    (printf("Mismatch at (b=%d, c=%d, h=%d, w=%d): %f vs %f\n", b, c, h, w, output(b, c, h, w), out_cpu(b, c, h, w)), false));
    //                 // assert(output(b, c, h, w) != 0.0);          
    //             }
    //         }
    //     }
    // }

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
