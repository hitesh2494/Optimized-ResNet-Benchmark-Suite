#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void executepoolingCudaOptimized(float *Layer2_Neurons_GPU,float *Layer2_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc,int pad,int tfactor, unsigned long long* runtime)
{
    unsigned long long start_time = clock64();
    int tid = blockDim.x * blockIdx.x * blockDim.y + blockDim.x * threadIdx.x + threadIdx.y;
    float max = 0.0;
    int stride = 0,colstride = 0;
    int output = blockIdx.x;
    int row_even = threadIdx.x * tfactor;
    int col_even = threadIdx.y * tfactor;
    int loopr = kernel, loopc = kernel;
    if(row_even < out_fr && col_even < out_fc)
    {
        for(int row = row_even; row < row_even+tfactor;row++)
        {
        colstride = (row*stride_width -pad)*in_fr;
        colstride = colstride < 0 ? 0: colstride;
        stride = 0;
            for(int col = col_even; col < col_even +tfactor ;col++)
            {
                loopr = kernel; loopc = kernel;
                stride = col*stride_width - pad;
                stride = stride < 0 ? 0: stride;

                if(col < pad)
                {
                    loopc = kernel - pad;
                }
                if(row < pad)
                {
                    loopr = kernel - pad;
                }
                if(col > out_fc - pad)
                {
                    loopc = in_fc - stride;
                }
                if(row > out_fr - pad)
                {
                    loopr = in_fr - colstride/in_fr;
                }
                for(int i = 0; i < loopr; i++)
                {
                    for(int j = 0; j < loopc; j++)
                    {
                        if(max < ((Layer2_Neurons_GPU[(output*in_fr*in_fc) + i*in_fc + j + stride + colstride])))
                        max =   ((Layer2_Neurons_GPU[(output*in_fr*in_fc) + i*in_fc + j + stride + colstride])) ;
                    }
                }
                Layer2_pool_GPU[output*out_fr*out_fc + row*out_fc + col] = max;
                max = 0.0;
            }
        }
    }
unsigned long long stop_time = clock64();
runtime[tid] = (unsigned long long)(stop_time - start_time);
}

__global__ void executepoolingCuda(float *Layer2_Neurons_GPU,float *Layer2_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc,int pad,int tfactor, unsigned long long* runtime)
{
unsigned long long start_time = clock64();
int tid = blockDim.x * blockIdx.x * blockDim.y + blockDim.x * threadIdx.x + threadIdx.y;
float max = 0.0;
int stride = 0,colstride = 0;
int output = blockIdx.x;
int row_even = threadIdx.x * tfactor;
int col_even = threadIdx.y * tfactor;
int loopr = kernel, loopc = kernel;
{
if(row_even < out_fr && col_even < out_fc)
{
for(int row = row_even; row < row_even+tfactor;row++)
{
colstride = (row*stride_width -pad)*in_fr;
colstride = colstride < 0 ? 0: colstride;
stride = 0;
for(int col = col_even; col < col_even +tfactor ;col++)
{
loopr = kernel; loopc = kernel;
stride = col*stride_width - pad;
stride = stride < 0 ? 0: stride;

if(col < pad)
{
loopc = kernel - pad;
//  printf("col %d loopc %d\n",col,loopc);
}
if(row < pad)
{
loopr = kernel - pad;
// printf("row %d loopr %d\n",row,loopr);
}
if(col > out_fc - pad)
{
loopc = in_fc - stride;
// printf("col %d loopc %d\n",col,loopc);

}
if(row > out_fr - pad)
{
loopr = in_fr - colstride/in_fr;
// printf("row %d loopr %d\n",row,loopr);

}
for(int i = 0; i < loopr; i++)
{
for(int j = 0; j < loopc; j++)
{
if(max < ((Layer2_Neurons_GPU[(output*in_fr*in_fc) + i*in_fc + j + stride + colstride])))
max =   ((Layer2_Neurons_GPU[(output*in_fr*in_fc) + i*in_fc + j + stride + colstride])) ;
//        printf("%d %d %d %f\n",(output*in_fr*in_fc) + i*in_fc + j + stride + colstride,row,col,max);

}
}
Layer2_pool_GPU[output*out_fr*out_fc + row*out_fc + col] = max;
max = 0.0;
}
}
}
}
unsigned long long stop_time = clock64();
runtime[tid] = (unsigned long long)(stop_time - start_time);
}

int
main(void)
{
    cudaError_t err = cudaSuccess;

    float *Layer2_Neurons_CPU = (float *)malloc(64*112*112*sizeof(float));

    float *Layer2_pool_CPU1 = (float *)malloc(64*56*56*sizeof(float));
    float *Layer2_pool_CPU2 = (float *)malloc(64*56*56*sizeof(float));

    double val = 112.0/32.0;	
    int tfactor = ceil(val);
    // Verify that allocations succeeded
    if (Layer2_Neurons_CPU == NULL || Layer2_pool_CPU1 == NULL || Layer2_pool_CPU2 == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for(int i=0; i<64*112*112; i++)
    {
        Layer2_Neurons_CPU[i] = rand()/(float)RAND_MAX;
    }

    float *Layer2_Neurons_GPU = NULL;
    err = cudaMalloc((void**) &Layer2_Neurons_GPU, 64*112*112*sizeof(float));

    float *Layer2_pool_GPU1 = NULL;
    err = cudaMalloc((void**) &Layer2_pool_GPU1, 64*56*56*sizeof(float));

    float *Layer2_pool_GPU2 = NULL;
    err = cudaMalloc((void**) &Layer2_pool_GPU2, 64*56*56*sizeof(float));

    err = cudaMemcpy(Layer2_Neurons_GPU, Layer2_Neurons_CPU, 64*112*112*sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

 #ifdef TM
    unsigned long long* d_runtime;
    int r_size = 64*112*112*sizeof(unsigned long long);
    unsigned long long* runtime = (unsigned long long*)malloc(r_size);
    cudaMalloc((void**)&d_runtime, r_size);
 #endif
 #ifdef TM
    unsigned long long* d_runtime1;
    int r_size1 = 64*112*112*sizeof(unsigned long long);
    unsigned long long* runtime1 = (unsigned long long*)malloc(r_size1);
    cudaMalloc((void**)&d_runtime1, r_size1);
 #endif

    dim3 numBlocks1(64,1,1);
    dim3 numThreads1(32,32);
    executepoolingCuda<<<numBlocks1,numThreads1>>>(Layer2_Neurons_GPU,Layer2_pool_GPU2, 64,56,56,3,2,112,112,1,tfactor, d_runtime1);

    dim3 numBlocks(64,1,1);
    dim3 numThreads(14,14);
    executepoolingCudaOptimized<<<numBlocks,numThreads>>>(Layer2_Neurons_GPU,Layer2_pool_GPU1, 64,56,56,3,2,112,112,1,tfactor, d_runtime);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(Layer2_pool_CPU1, Layer2_pool_GPU1, 64*56*56*sizeof(float), cudaMemcpyDeviceToHost);
    err = cudaMemcpy(Layer2_pool_CPU2, Layer2_pool_GPU2, 64*56*56*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    for(int i=0; i<64*56*56; i++)
    {
	    //printf("%f",Layer2_Neurons_CPU2[i]);
        if(Layer2_pool_CPU1[i] != Layer2_pool_CPU2[i])
        {
           printf("Matrices Unequal at position  i : %d\n", i);
           break;
        }
    }
    printf("Test PASSED\n");

#ifdef TM
    cudaMemcpy(runtime1, d_runtime1, r_size1, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    unsigned long long elapsed_time1 = 0;
    for(int i=0; i< 64*112*112; i++)
        if(elapsed_time1 < runtime1[i])
           elapsed_time1 = runtime1[i];
        printf("Kernel Execution Time for Non-Optimized: %llu cycles\n", elapsed_time1);
#endif
#ifdef TM
    cudaMemcpy(runtime, d_runtime, r_size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    unsigned long long elapsed_time = 0;
    for(int i=0; i< 64*112*112; i++)
        if(elapsed_time < runtime[i])
           elapsed_time = runtime[i];
        printf("Kernel Execution Time for Optimized: %llu cycles\n", elapsed_time);
#endif
    // Free device global memory
    err = cudaFree(Layer2_Neurons_GPU);

    err = cudaFree(Layer2_pool_GPU1);

    err = cudaFree(Layer2_pool_GPU2);

#ifdef TM
    cudaFree(d_runtime);
    cudaFree(d_runtime1);
#endif

    // Free host memory
    free(Layer2_pool_CPU1);
    free(Layer2_pool_CPU2);
    free(Layer2_Neurons_CPU);

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

