##########################################################################################

1. This folder contains 4 folders namely executeFirstLayerCUDA, poolingAverageCUDA, executepoolingCUDA, executeFCLayerCUDA.

2. All these folders are an optimized version of the Kernels from GPU_ResNet_rn_kernel.cu file from the ResNet Tango benchmark Suite.

3. Each folder contains file name cmpe214_matMul_shared.cu and Makefile. Use the following commands to execute each Kernel.
   
  (i)   make clean
  (ii)  make def=TM
  (iii) ./matrixMul_shared (to run the executable)

4. All the kernels have been verified using the random values (initialized input using rand()/(float)RAND_MAX method) for the input instead of fetching the input from the files(data folder in the original benchmark suite).

5. In each file I have created 2 kernels. For example executeFirstLayerCUDA folder contains executeFirstLayerCUDA and executeFirstLayerCUDAOptimized. The output from each kernel is stored in a different matrix. I have then checked the output of the Optimized Kernel with the Unoptimized Kernel to make sure that the result after optimizations are same as the original Kernel.

##########################################################################################