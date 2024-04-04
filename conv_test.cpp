#include "Conv2d.h"
#include "../utils/Tensor.cpp"
#include <cstdlib> // for rand() function
#include <ctime>   // for seeding rand() function
#include <iostream> // for cout
#include <iomanip> // for setw
#include <cmath> // for round

#pragma once

// Function to generate random float between -1 and 1
float rand_float() {
    return (2.0f * rand() / RAND_MAX) - 1.0f;
}

int main(){
    // Seed the random number generator
    srand(time(0));

    // Define the dimensions of the tensor
    int batch_size = 5;
    int channels = 3;
    int height = 512;
    int width = 768;

    // Create the tensor
    Tensor<float, 4> input(batch_size, channels, height, width);
    Tensor<float, 4> filters(64, channels, 3, 3);
    Conv2d conv(3, 64, 3); // in_channel, out_channel, filter_size

    // Fill the tensor with random values between -1 and 1
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    input(b, c, h, w) = 1;
                }
            }
        }
    }

    for (int b = 0; b < 64; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < 3; ++h) {
                for (int w = 0; w < 3; ++w) {
                    filters(b, c, h, w) = 3;//rand_float();
                }
            }
        }
    }

    conv.weights = filters;
    Tensor<float, 4> output = conv.forward(input);
    
    // std::cout << output.dims[3] << std::endl;
    // output.print();
    std::cout << output(1,1,1,1) << std::endl;
    assert(output.dims[0] == batch_size);
    assert(output.dims[1] == 64);
    assert(output.dims[2] == 512-2);
    assert(output.dims[3] == 768-2);

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < 64; ++c) {
            for (int h = 0; h < output.dims[2]; ++h) {
                for (int w = 0; w < output.dims[3]; ++w) {
                    assert(output(b, c, h, w) == 81.0); // 9*3*3
                }
            }
        }
    }


    // Print out a sample of the tensor values
    for (int b = 0; b < 1; ++b) {
        for (int c = 0; c < 1; ++c) {
            for (int h = 0; h < 2; ++h) {
                for (int w = 0; w < 2; ++w) {
                    std::cout << std::fixed << std::setprecision(3) << std::setw(8) << input(b, c, h, w) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    return 0;
}
