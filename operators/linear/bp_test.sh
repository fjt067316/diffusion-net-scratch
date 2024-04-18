#!/bin/bash

# Compile Conv2d.cu
nvcc -c -arch=sm_89 Linear.cu -o linear.o

g++ -o program.o -I/usr/local/cuda-11.8/include linear_backprop.cpp linear.o -L/usr/local/cuda-11.8/lib64 -lcudart

# Clean up object files
rm linear.o

echo "Compilation finished."
