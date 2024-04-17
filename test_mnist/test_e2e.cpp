#include "../operators/conv2d/Conv2d.h"
#include "../operators/linear/Linear.h"
#include "../operators/batch_norm/BatchNorm2d.h"
#include "../utils/Tensor.h"
#include <cstdlib> // for rand() function
#include <ctime>   // for seeding rand() function
#include <iostream> // for cout
#include <iomanip> // for setw
#include <cmath> // for round
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>

#pragma once

using namespace std;

// Function to generate random float between -1 and 1
float rand_float() {
    return (2.0f * rand() / RAND_MAX) - 1.0f;
}

int batch_size = 1;

auto decodeCsvString(string csv){ // if I try to do this function any other way it fucking explodes when returning a value
    vector<float> values;
    vector<float> labels;
    stringstream ss(csv);
    string item;
    while (getline(ss, item, ',')) {
        values.push_back(std::stoi(item)); // simulate 3 in channels
    }
    labels.push_back(values.front()); // correct image value
    values.erase(values.begin());
    // for (int value : values) {
    //     std::cout << value << " ";
    // }
    return make_pair(values, labels);
}

Tensor<float, 2> softmax(Tensor<float, 2> logits) {
    float sum_exp = 0;
    float max_n = 0;

    for (size_t i = 0; i < 10; ++i) {
        max_n = max(max_n, logits.data[i]);
    }

    for (size_t i = 0; i < 10; ++i) {
        logits.data[i] = std::exp(logits.data[i]-max_n);
        sum_exp += logits.data[i];
    }

    for (size_t i = 0; i < 10; ++i) {
        logits.data[i] /= sum_exp;
    }
    return logits;
}


/*
CPU time: 439.001 seconds
GPU time: 11.5202 seconds
*/
int main(){
    // Seed the random number generator
    srand(time(0));

    std::ifstream inputFile("../mnist/mnist_small.csv");

    if (!inputFile.is_open()) {
        cerr << "Error: could not open file" << endl;
        return 1;
    }

    string row;
    getline(inputFile, row); // discard first header row

    // Define the dimensions of the tensor
    // int batch_size = 5;
    int channels = 1;
    int height = 28;//512;
    int width = 28;//768;

    // in_channel, out_channel, filter_size, padding, stride  bool use_bias=true, bool use_relu=false
    Conv2d conv1(channels, 16, 3, 0, 1, false, true); // Bx28x28->16x26,26
    BatchNorm2d bn1(16);
    Conv2d conv2(16, 32, 4, 0, 1, false, true); // 16x26,26 -> 32x23x23
    BatchNorm2d bn2(32);
    Conv2d conv3(32, 32, 4, 0, 2, false, true); // 32x23x23 -> 32x10x10

    // int input_size, int output_size, bool use_bias=true, bool use_relu=false
    Linear lin1(32*10*10, 256, false, true);
    Linear lin2(256, 10, false, false);
    
    int iterations = 3;

    int num_correct = 0;
    float loss = 0;
    Tensor<float, 4> input_tensor({batch_size, channels, 28, 28});
    Tensor<float, 2> dLdZ({batch_size, 10}, true);

    for(int n=0; n<iterations; n++){
        row.erase();
        getline(inputFile, row);
        printf("hit1\n");

        auto data_pair = decodeCsvString(row); // input = (784)
        vector<float> input = data_pair.first;
        vector<float> labels = data_pair.second;
        printf("hit2\n");

        input_tensor.data = input.data();
        input_tensor.toDevice();
        printf("hit3\n");

        auto x = conv1.forward(input_tensor);
        printf("hit4\n");

        // printf("conv1 %d %d %d %d\n", x.dim(0), x.dim(1), x.dim(2), x.dim(3));

        x = bn1.forward(x);
        // printf("bn1 %d %d %d %d\n", x.dim(0), x.dim(1), x.dim(2), x.dim(3));

        x = conv2.forward(x);
        // printf("conv2 %d %d %d %d\n", x.dim(0), x.dim(1), x.dim(2), x.dim(3));

        x = bn2.forward(x);

        // printf("%d %d %d %d\n", x.dim(0), x.dim(1), x.dim(2), x.dim(3));

        x = conv3.forward(x);
        // x.toHost();
        // x.print();
        // x.toDevice();
        Tensor<float, 2> flat_x({batch_size, 32*10*10}, false);

        flat_x.data = x.data;
        // flat_x.toHost();
        // flat_x.print();
        // flat_x.toDevice();
        flat_x = lin1.forward(flat_x);

        Tensor<float, 2> out = lin2.forward(flat_x);

        // out.toHost();
        // out.print();
        int correct_idx = labels[0];

        out.toHost();
        // Compute softmax
        auto predictions = softmax(out);
        predictions.print();
        printf("loss %f\n", -1*std::log(predictions.data[correct_idx]));

        float dLdZ_correct = predictions.data[correct_idx] - 1;
        for (int i = 0; i < out.size; i++) {
            if (i == correct_idx) {
                out.data[i] = predictions.data[i] - 1;
            } else {
                out.data[i] = predictions.data[i];
            }
        }       

        // Scale the gradients by dL/dZ for the correct class
        for (int i = 0; i < out.size; i++) {
            out.data[i] *= -1;
        }
        // float label_prob = out.data[correct_idx];
        // float exp_label = std::exp( out.data[correct_idx]);

        // for(int i=0; i<10; i++){
        //     dLdZ.data[i] = 0;
        // }
        // dLdZ.data[correct_idx] = -1 / predictions.data[correct_idx];

        // float dz_exp[10];
        // float sum_exp = 0;

        // for(int i=0; i<10; i++){
        //     dz_exp[i] = std::exp(out.data[i]);
        //     sum_exp += dz_exp[i];
        // }


        // // compute loss
        // for(int i=0; i<out.size; i++){
        //     // out.data[i] = label_prob*-std::exp(out.data[i])*exp_label / (sum_exp*sum_exp);
        //     out.data[i] = -dz_exp[correct_idx]*dz_exp[i] / (sum_exp*sum_exp);
        // }

        // out.data[correct_idx] = dz_exp[correct_idx] * (sum_exp-dz_exp[correct_idx]) / (sum_exp*sum_exp);
        // out.print();
        out.toDevice();

        auto dz = lin2.backward(out);

        dz = lin1.backward(dz);

        Tensor<float, 4> dz_conv({batch_size, 32,10,10}, false);

        dz_conv.data = dz.data;

        dz_conv = conv3.backward(dz_conv);

        dz_conv = bn2.backward(dz_conv);
        dz_conv = conv2.backward(dz_conv);
        dz_conv = bn1.backward(dz_conv);
        dz_conv = conv1.backward(dz_conv);

        // model->forward(image_1x28x28, image_label, &loss);
    }

    // Time GPU calculation
    // std::cout << "GPU calculation" << std::endl;
    // auto start_gpu = std::chrono::steady_clock::now();
    // Tensor<float, 4> output = conv.forward(input);
    // auto end_gpu = std::chrono::steady_clock::now();
    // std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;
    // std::cout << "GPU time: " << gpu_duration.count() << " seconds" << std::endl;
    
    
    return 0;
}
