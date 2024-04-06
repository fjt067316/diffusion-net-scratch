#!/bin/bash
#!/bin/bash

# Set the path to CUDA toolkit if it's not already set
CUDA_PATH=/usr/local/cuda

# Compiler flags
NVCC_FLAGS="-std=c++11 -arch=sm_89"

# Compile Conv2d.cu
nvcc -c -arch=sm_70 ConvTranspose2d.cu -o ConvTranspose2d.o

# Compile test_conv.cpp
# nvcc $NVCC_FLAGS -c test_conv.cpp ../utils/Tensor.cpp -o test_conv.o

g++ -o program.o -I/usr/local/cuda-11.8/include conv_transpose_test.cpp ConvTranspose2d.o -L/usr/local/cuda-11.8/lib64 -lcudart
# Link object files and create executable
#nvcc $NVCC_FLAGS Conv2d.o test_conv.o -o test_conv.o

# Clean up object files
rm ConvTranspose2d.o

echo "Compilation finished."
