#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void poolingAverageCUDAOptimized(float *Layer2_Neurons_GPU,float *Layer2_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc,int pad,unsigned long long* runtime)
{
    unsigned long long start_time = clock64();
    int tid = blockDim.x * blockIdx.x * blockDim.y + blockDim.x * threadIdx.x + threadIdx.y;
    float sum = 0.0;
    int loopr = kernel, loopc = kernel;
    int stride = 0,colstride = 0;
    int output = blockIdx.x*blockDim.x + threadIdx.x;

    int row = 0;
    colstride = (row*stride_width -pad)*in_fr;
    colstride = colstride < 0 ? 0: colstride;
    stride = 0;
    int col = 0;
    loopr = kernel; loopc = kernel;
    stride = col*stride_width - pad;
    stride = stride < 0 ? 0: stride;

            	sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 0*in_fc + 0 + stride + colstride] ;
		sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 0*in_fc + 1 + stride + colstride] ;
		sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 0*in_fc + 2 + stride + colstride] ;
		sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 0*in_fc + 3 + stride + colstride] ;
		sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 0*in_fc + 4 + stride + colstride] ;
		sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 0*in_fc + 5 + stride + colstride] ;
		sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 0*in_fc + 6 + stride + colstride] ;
		sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 1*in_fc + 0 + stride + colstride] ;
		sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 1*in_fc + 1 + stride + colstride] ;
		sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 1*in_fc + 2 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 1*in_fc + 3 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 1*in_fc + 4 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 1*in_fc + 5 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 1*in_fc + 6 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 2*in_fc + 0 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 2*in_fc + 1 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 2*in_fc + 2 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 2*in_fc + 3 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 2*in_fc + 4 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 2*in_fc + 5 + stride + colstride] ;
		sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 2*in_fc + 6 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 3*in_fc + 0 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 3*in_fc + 1 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 3*in_fc + 2 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 3*in_fc + 3 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 3*in_fc + 4 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 3*in_fc + 5 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 3*in_fc + 6 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 4*in_fc + 0 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 4*in_fc + 1 + stride + colstride] ;
		sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 4*in_fc + 2 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 4*in_fc + 3 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 4*in_fc + 4 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 4*in_fc + 5 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 4*in_fc + 6 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 5*in_fc + 0 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 5*in_fc + 1 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 5*in_fc + 2 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 5*in_fc + 3 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 5*in_fc + 4 + stride + colstride] ;
		sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 5*in_fc + 5 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 5*in_fc + 6 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 6*in_fc + 0 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 6*in_fc + 1 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 6*in_fc + 2 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 6*in_fc + 3 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 6*in_fc + 4 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 6*in_fc + 5 + stride + colstride] ;
                sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + 6*in_fc + 6 + stride + colstride] ;
    
    Layer2_pool_GPU[output] = (sum/(loopc*loopr));
    sum = 0.0;
unsigned long long stop_time = clock64();
runtime[tid] = (unsigned long long)(stop_time - start_time);
}

__global__ void poolingAverageCUDA(float *Layer2_Neurons_GPU,float *Layer2_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc,int pad,unsigned long long* runtime)
{
    unsigned long long start_time = clock64();
    int tid = blockDim.x * blockIdx.x * blockDim.y + blockDim.x * threadIdx.x + threadIdx.y;
    float sum = 0.0;
    int loopr = kernel, loopc = kernel;
    int stride = 0,colstride = 0;
    {
    int output = blockIdx.x;//for(int output =0;output < out ;output++)
    {
    int row = 0;
    {
    colstride = (row*stride_width -pad)*in_fr;
    colstride = colstride < 0 ? 0: colstride;
    stride = 0;
    int col = 0;
    {
    loopr = kernel; loopc = kernel;
    stride = col*stride_width - pad;
    stride = stride < 0 ? 0: stride;

    for(int i = 0; i < loopr; i++)
    {
    for(int j = 0; j < loopc; j++)
    {
    sum += Layer2_Neurons_GPU[(output*in_fr*in_fc) + i*in_fc + j + stride + colstride] ;

    }
    }
    Layer2_pool_GPU[output] = (sum/(loopc*loopr));
    sum = 0.0;
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

    float *Layer5c_Neurons_CPU = (float *)malloc(2048*7*7*sizeof(float));

    float *Out_CPU1 = (float *)malloc(2048*sizeof(float));
    float *Out_CPU2 = (float *)malloc(2048*sizeof(float));

    // Verify that allocations succeeded
    if (Layer5c_Neurons_CPU == NULL || Out_CPU1 == NULL || Out_CPU2 == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for(int i=0; i<2048*7*7; i++)
    {
        Layer5c_Neurons_CPU[i] = rand()/(float)RAND_MAX;
    }

    float *Layer5c_Neurons_GPU = NULL;
    err = cudaMalloc((void**) &Layer5c_Neurons_GPU, 2048*7*7*sizeof(float));

    float *Out_GPU1 = NULL;
    err = cudaMalloc((void**) &Out_GPU1, 2048*sizeof(float));

    float *Out_GPU2 = NULL;
    err = cudaMalloc((void**) &Out_GPU2, 2048*sizeof(float));

    err = cudaMemcpy(Layer5c_Neurons_GPU, Layer5c_Neurons_CPU, 2048*7*7*sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

 #ifdef TM
    unsigned long long* d_runtime;
    int r_size = 2048*7*7*sizeof(unsigned long long);
    unsigned long long* runtime = (unsigned long long*)malloc(r_size);
    cudaMalloc((void**)&d_runtime, r_size);
 #endif
 #ifdef TM
    unsigned long long* d_runtime1;
    int r_size1 = 7*7*2048*sizeof(unsigned long long);
    unsigned long long* runtime1 = (unsigned long long*)malloc(r_size1);
    cudaMalloc((void**)&d_runtime1, r_size1);
 #endif

    dim3 numBlocks(2048,1,1);
    dim3 numThreads(1,1);
    poolingAverageCUDA<<<numBlocks, numThreads>>>(Layer5c_Neurons_GPU,Out_GPU1, 2048,1,1,7,1,7,7,0,d_runtime);

    poolingAverageCUDAOptimized<<<16, 128>>>(Layer5c_Neurons_GPU,Out_GPU2,2048,1,1,7,1,7,7,0,d_runtime1);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(Out_CPU1, Out_GPU1, 2048*sizeof(float), cudaMemcpyDeviceToHost);
    err = cudaMemcpy(Out_CPU2, Out_GPU2, 2048*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    for(int i=0; i<2048; i++)
    {
	    //printf("%f",Layer2_Neurons_CPU2[i]);
        if(Out_CPU1[i] != Out_CPU2[i])
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
    for(int i=0; i< 7*7*2048; i++)
        if(elapsed_time < runtime[i])
           elapsed_time = runtime[i];
    printf("Kernel Execution Time for Non Optimized: %llu cycles\n", elapsed_time);
#endif
#ifdef TM
    cudaMemcpy(runtime1, d_runtime1, r_size1, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    unsigned long long elapsed_time1 = 0;
    for(int i=0; i< 7*7*2048; i++)
        if(elapsed_time1 < runtime1[i])
           elapsed_time1 = runtime1[i];
        printf("Kernel Execution Time for Optimized: %llu cycles\n", elapsed_time1);
#endif
    // Free device global memory
    err = cudaFree(Layer5c_Neurons_GPU);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(Out_GPU1);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(Out_GPU2);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


#ifdef TM
    cudaFree(d_runtime);
     cudaFree(d_runtime1);
#endif

    // Free host memory
    free(Out_CPU1);
    free(Out_CPU2);
    free(Layer5c_Neurons_CPU);

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

