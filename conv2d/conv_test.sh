#!/bin/bash
#!/bin/bash

# Set the path to CUDA toolkit if it's not already set
CUDA_PATH=/usr/local/cuda

# Compiler flags
NVCC_FLAGS="-std=c++11 -arch=sm_89"

# Compile Conv2d.cu
nvcc -c -arch=sm_70 Conv2d.cu -o Conv2d.o

# Compile test_conv.cpp
# nvcc $NVCC_FLAGS -c test_conv.cpp ../utils/Tensor.cpp -o test_conv.o

g++ -o program.o -I/usr/local/cuda-11.8/include conv_test.cpp Conv2d.o -L/usr/local/cuda-11.8/lib64 -lcudart
# Link object files and create executable
#nvcc $NVCC_FLAGS Conv2d.o test_conv.o -o test_conv.o

# Clean up object files
rm Conv2d.o

echo "Compilation finished."
