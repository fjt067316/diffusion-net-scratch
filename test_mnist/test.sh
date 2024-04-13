#!/bin/bash

# Compile each CUDA source file separately
nvcc -c -arch=sm_70 ../operators/linear/Linear.cu -o linear.o
nvcc -c -arch=sm_70 ../operators/conv2d/Conv2d.cu -o conv2d.o
nvcc -c -arch=sm_70 ../operators/batch_norm/BatchNorm2d.cu -o batch_norm.o

# Compile the test_e2e.cpp file
g++ -c -o test_e2e.o -I/usr/local/cuda-11.8/include test_e2e.cpp

# Link all object files together
g++ -o program.o test_e2e.o linear.o conv2d.o batch_norm.o -L/usr/local/cuda-11.8/lib64 -lcudart

# Clean up object files
rm linear.o conv2d.o batch_norm.o test_e2e.o

echo "Compilation finished."
