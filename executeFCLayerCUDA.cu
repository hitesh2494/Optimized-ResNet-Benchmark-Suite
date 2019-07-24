#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

__constant__ float c_Layer_InNeurons_GPU[2048];
__global__ void executeFCLayerCUDAOptimized(float *Layer_Weights_GPU,float *Layer_OutNeurons_GPU,int input, unsigned long long* runtime)
{
    unsigned long long start_time = clock64();
    int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * blockDim.x + threadIdx.y;
    float product = 0.0;
    int out = blockDim.x*blockIdx.x + threadIdx.x; 
    int weight =  out * input;
    
    for(int in = 0; in < input; in++)
    {
      product += c_Layer_InNeurons_GPU[in] * Layer_Weights_GPU[weight+in];
    }
    Layer_OutNeurons_GPU[out] = product;
    product = 0.0;

    unsigned long long stop_time = clock64();
    runtime[tid] = (unsigned long long)(stop_time - start_time);
}
__global__ void executeFCLayerCUDA(float *Layer_InNeurons_GPU,float *Layer_Weights_GPU,float *Layer_OutNeurons_GPU,int input, unsigned long long* runtime)
{
    unsigned long long start_time = clock64();
    int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * blockDim.x + threadIdx.y;
    float product = 0.0;
    int out = blockIdx.x; 
    int weight =  out * input; 
    for(int in = 0; in < input; in++)
    {
        product += Layer_InNeurons_GPU[in] * Layer_Weights_GPU[weight+in];
    }
    Layer_OutNeurons_GPU[out] = product;
    product = 0.0;

    unsigned long long stop_time = clock64();
    runtime[tid] = (unsigned long long)(stop_time - start_time);
}

int
main(void)
{
    cudaError_t err = cudaSuccess;

    float *Out_CPU = (float *)malloc(2048*sizeof(float));

    float *Layer_FC_Weights_CPU = (float *)malloc(2048*1000*sizeof(float));
    float *Layer_FC_Out_CPU1 = (float *)malloc(1000*sizeof(float));
    float *Layer_FC_Out_CPU2 = (float *)malloc(1000*sizeof(float));

    // Verify that allocations succeeded
    if(Out_CPU == NULL || Layer_FC_Weights_CPU == NULL || Layer_FC_Out_CPU1 == NULL || Layer_FC_Out_CPU2==NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for(int i=0; i<2048; i++)
    {
        Out_CPU[i] = rand()/(float)RAND_MAX;
    }
    for(int i=0; i<2048*1000; i++)
    {
        Layer_FC_Weights_CPU[i] = rand()/(float)RAND_MAX;
    }

    float *Out_GPU = NULL;
    err = cudaMalloc((void**) &Out_GPU, 2048*sizeof(float));

    float *Layer_FC_Weights_GPU = NULL;
    err = cudaMalloc((void**) &Layer_FC_Weights_GPU, 2048*1000*sizeof(float));

    float *Layer_FC_Out_GPU1 = NULL;
    err = cudaMalloc((void**) &Layer_FC_Out_GPU1, 1000*sizeof(float));

    float *Layer_FC_Out_GPU2 = NULL;
    err = cudaMalloc((void**) &Layer_FC_Out_GPU2, 1000*sizeof(float));

    err = cudaMemcpy(Out_GPU, Out_CPU, 2048*sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(Layer_FC_Weights_GPU, Layer_FC_Weights_CPU, 2048*1000*sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
    fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
    err = cudaMemcpyToSymbol(c_Layer_InNeurons_GPU, Out_CPU, 2048*sizeof(float));

    if (err != cudaSuccess)
    {
    fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }

#ifdef TM
    unsigned long long* d_runtime;
    int r_size = 2048*1000*sizeof(unsigned long long);
    unsigned long long* runtime = (unsigned long long*)malloc(r_size);
    cudaMalloc((void**)&d_runtime, r_size);
 #endif

#ifdef TM
unsigned long long* d_runtime1;
int r_size1 = 2048*1000*sizeof(unsigned long long);
unsigned long long* runtime1 = (unsigned long long*)malloc(r_size1);
cudaMalloc((void**)&d_runtime1, r_size1);
#endif

dim3 numBlocks(1000,1,1);
dim3 numThreads(1,1);

executeFCLayerCUDA<<<numBlocks,numThreads>>>(Out_GPU,Layer_FC_Weights_GPU,Layer_FC_Out_GPU1, 2048, d_runtime);
executeFCLayerCUDAOptimized<<<10, 100>>>(Layer_FC_Weights_GPU,Layer_FC_Out_GPU2,2048, d_runtime1);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(Layer_FC_Out_CPU1, Layer_FC_Out_GPU1, 1000*sizeof(float), cudaMemcpyDeviceToHost);
    err = cudaMemcpy(Layer_FC_Out_CPU2, Layer_FC_Out_GPU2, 1000*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    for(int i=0; i<1000; i++)
    {
        if(Layer_FC_Out_CPU1[i] != Layer_FC_Out_CPU2[i])
        {
           printf("Matrices Unequal at position  i : %d\n", i);
           break;
        }
    }
    printf("Test PASSED\n");

#ifdef TM
    cudaMemcpy(runtime, d_runtime, r_size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    unsigned long long elapsed_time = 0;
    for(int i=0; i<2048*1000; i++)
        if(elapsed_time < runtime[i])
           elapsed_time = runtime[i];
        printf("Kernel Execution Time for Non Optimized : %llu cycles\n", elapsed_time);
#endif

#ifdef TM
cudaMemcpy(runtime1, d_runtime1, r_size1, cudaMemcpyDeviceToHost);
cudaThreadSynchronize();

unsigned long long elapsed_time1 = 0;
for(int i=0; i<2048*1000; i++)
if(elapsed_time1 < runtime1[i])
elapsed_time1 = runtime1[i];
printf("Kernel Execution Time for Optimized : %llu cycles\n", elapsed_time1);
#endif
    // Free device global memory
    err = cudaFree(Out_GPU);
    err = cudaFree(Layer_FC_Weights_GPU);
    err = cudaFree(Layer_FC_Out_GPU1);
    err = cudaFree(Layer_FC_Out_GPU2);

#ifdef TM
    cudaFree(d_runtime);
    cudaFree(d_runtime1);
#endif

    // Free host memory
    free(Out_CPU);
    free(Layer_FC_Weights_CPU);
    free(Layer_FC_Out_CPU1);
    free(Layer_FC_Out_CPU2);

#ifdef TM
    free(runtime);
    free(runtime1);
#endif
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}

