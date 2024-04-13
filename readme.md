g++ -o my_program main.cpp -lopencv_core -lopencv_imgcodecs


nvcc main.cpp kernel.cu -o executable_name

/usr/local/cuda-11.8/extras/demo_suite/deviceQuery

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-11.8/bin

 export PATH="/usr/local/cuda-11.8/bin:$PATH"
 export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"






/usr/local/cuda-11.8/extras/demo_suite/deviceQuery 
 
Device 0: "NVIDIA GeForce RTX 4060 Ti"                                                                                    
CUDA Driver Version / Runtime Version          12.4 / 11.8                                                              
CUDA Capability Major/Minor version number:    8.9                                                                      
Total amount of global memory:                 16380 MBytes (17175150592 bytes)                                       
MapSMtoCores for SM 8.9 is undefined.  Default to use 128 Cores/SM                                                      
MapSMtoCores for SM 8.9 is undefined.  Default to use 128 Cores/SM                                                        
(34) Multiprocessors, (128) CUDA Cores/MP:     4352 CUDA Cores                                                          
GPU Max Clock rate:                            2610 MHz (2.61 GHz)                                                      
Memory Clock rate:                             9001 Mhz                                                                 
Memory Bus Width:                              128-bit                                                                  
L2 Cache Size:                                 33554432 bytes                                                           
Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)                
Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers                                                  
Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers                                           
Total amount of constant memory:               65536 bytes                                                              
Total amount of shared memory per block:       49152 bytes                                                              
Total number of registers available per block: 65536                                                                    
Warp size:                                     32                                                                       
Maximum number of threads per multiprocessor:  1536                                                                     
Maximum number of threads per block:           1024                                                                     
Max dimension size of a thread block (x,y,z): (1024, 1024, 64)                                                          
Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)                                                
Maximum memory pitch:                          2147483647 bytes                                                         
Texture alignment:                             512 bytes                                                               
 Concurrent copy and kernel execution:          Yes with 5 copy engine(s)                                                
 Run time limit on kernels:                     Yes                                                                      
 Integrated GPU sharing Host Memory:            No                                                                       
 Support host page-locked memory mapping:       Yes                                                                      
 Alignment requirement for Surfaces:            Yes                                                                      
 Device has ECC support:                        Disabled                                                                 
 Device supports Unified Addressing (UVA):      Yes                                                                      
 Device supports Compute Preemption:            Yes                                                                      
 Supports Cooperative Kernel Launch:            Yes                                                                      
 Supports MultiDevice Co-op Kernel Launch:      No                                                                       
 Device PCI Domain ID / Bus ID / location ID:   0 / 48 / 0                                                               
 Compute Mode:                                                                                                              
 < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >                                                                                                                                                   
 deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.4, CUDA Runtime Version = 11.8, NumDevs = 1, Device0 = NVIDIA GeForce RTX 4060 Ti



Expected shapes Batch Size 1

time mlp torch.Size([1, 32])
conv torch.Size([1, 32])
conv1 torch.Size([1, 128, 512, 768])
linear torch.Size([1, 128, 512, 768])
conv2 torch.Size([1, 128, 512, 768])
conv1 torch.Size([1, 256, 256, 384])
linear torch.Size([1, 256, 256, 384])
conv2 torch.Size([1, 256, 256, 384])
conv1 torch.Size([1, 512, 128, 192])
linear torch.Size([1, 512, 128, 192])
conv2 torch.Size([1, 512, 128, 192])
conv1 torch.Size([1, 1024, 64, 96])
linear torch.Size([1, 1024, 64, 96])
conv2 torch.Size([1, 1024, 64, 96])
conv1 torch.Size([1, 512, 32, 48])
linear torch.Size([1, 512, 32, 48])
conv2 torch.Size([1, 512, 32, 48])
conv1 torch.Size([1, 256, 64, 96])
linear torch.Size([1, 256, 64, 96])
conv2 torch.Size([1, 256, 64, 96])
conv1 torch.Size([1, 128, 128, 192])
linear torch.Size([1, 128, 128, 192])
conv2 torch.Size([1, 128, 128, 192])
conv1 torch.Size([1, 64, 256, 384])
linear torch.Size([1, 64, 256, 384])
conv2 torch.Size([1, 64, 256, 384])
output torch.Size([1, 64, 512, 768])