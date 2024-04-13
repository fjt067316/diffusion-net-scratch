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

int batch_size = 5;

auto decodeCsvString(string csv){
    vector<float> values;
    vector<float> labels;
    stringstream ss(csv);
    string item;

    for(int i=0; i<batch_size; i++){
        while (getline(ss, item, ',')) {
            labels.push_back(values.front()); // correct image value
            values.erase(values.begin());
            values.push_back(std::stoi(item)); // simulate 3 in channels
            values.push_back(std::stoi(item));
            values.push_back(std::stoi(item));

        }
    }
    // for (int value : values) {
    //     std::cout << value << " ";
    // }
    return make_pair(values, labels);
}

/*
CPU time: 439.001 seconds
GPU time: 11.5202 seconds
*/
int main(){
    // Seed the random number generator
    srand(time(0));

    ifstream inputFile("../mnist/mnist_small.csv");
    if (!inputFile.is_open()) {
        cerr << "Error: could not open file" << endl;
        return 1;
    }

    string row;
    getline(inputFile, row); // discard first header row

    // Define the dimensions of the tensor
    // int batch_size = 5;
    int channels = 3;
    int height = 28;//512;
    int width = 28;//768;

    // in_channel, out_channel, filter_size, padding, stride  bool use_bias=true, bool use_relu=false
    Conv2d conv1(channels, 16, 3, 0, 1, false, true); // Bx28x28->16x26,26
    BatchNorm2d bn1(16);
    Conv2d conv2(channels, 32, 4, 0, 1, false, true); // 16x26,26 -> 32x23x23
    BatchNorm2d bn2(32);
    Conv2d conv3(channels, 32, 4, 2, 1, false, true); // 32x23x23 -> 32x9x9

    // int input_size, int output_size, bool use_bias=true, bool use_relu=false
    Linear lin1(32*9*9, 256, false, true);
    Linear lin2(256, 10, false, false);


    
    int iterations = 10000;

    int num_correct = 0;
    float loss = 0;
    for(int n=0; n<iterations; n++){
        row.erase();
        getline(inputFile, row);
        auto data_pair = decodeCsvString(row); // input = (784)

        vector<float> input = data_pair.first;
        vector<float> labels = data_pair.second;
        Tensor<float, 4> input_tensor({batch_size, 3, 28, 28});

        input_tensor.data = input.data();
        input_tensor.toDevice();

        auto x = conv1.forward(input_tensor);
        x = bn1.forward(x);
        x = conv2.forward(x);
        x = bn2.forward(x);
        x = conv3.forward(x);

        Tensor<float, 2> flat_x({batch_size, 32*9*9}, false);
        flat_x.data = x.data;

        flat_x = lin1.forward(flat_x);
        auto out = lin2.forward(flat_x);

        // compute loss

        // auto dz = lin2.backward(loss);
        // dz = lin1.backward(dz);
        // dz = conv3.backward(dz);
        // dz = bn2.backward(dz);
        // dz = conv2.backward(dz);
        // dz = bn1.backward(dz);
        // dz = conv1.backward(dz);




        // model->forward(image_1x28x28, image_label, &loss);

        printf("[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%\n", n + 1, static_cast<float>(loss) / 100, num_correct);

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
