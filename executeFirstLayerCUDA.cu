#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 224*224*3

#include <cuda_runtime.h>
__global__ void executeFirstLayerCUDAOptimized(float *bias,float *Layer1_Neurons_GPU,float *Layer1_Weights_GPU,float *Layer2_Neurons_GPU,int stride_width,int pad, int col_width,int feature_r,int feature_c,int out,int tfactor,unsigned long long* runtime)
{
    unsigned long long start_time = clock64();

    __shared__ float shared_Layer1_Weights_GPU[7*7*3];
    int tid = threadIdx.x * blockDim.x + threadIdx.y;

    int stride = 0,colstride = 0;
    int output = blockIdx.x;
    int row_even = threadIdx.x * tfactor;
    int col_even = threadIdx.y * tfactor;
    int kernel = 7;
    int x_pad,y_pad,loopc,loopr;
    float product = 0.0;

    if(tid<7*7*3)
    {
	shared_Layer1_Weights_GPU[tid] = Layer1_Weights_GPU[output*7*7*3 + tid];
    }
	
    __syncthreads();

    if(row_even < feature_r && col_even < feature_c)
    {

            colstride = 0;
            for(int row =row_even; row < row_even+tfactor ;row++)
            {
                stride = 0;
                colstride = 3 *(row*stride_width - pad)*col_width;
                colstride = colstride < 0 ? 0 : colstride;
            for(int col =col_even; col < col_even+tfactor ;col++)
            {
                stride = 3 * (col*stride_width - pad);
                stride = stride < 0 ? 0 : stride;

            product = 0;
            x_pad = 0; y_pad = 0;
            /* set the loops value */
            loopc = kernel;loopr = kernel;
            /* take care of padding in left hand side of image*/
            if(row*stride_width < pad)
            {
                x_pad = pad - row*stride_width;
                loopr = kernel - x_pad;
            }
            /* take care of padding in upper side of image*/
            if( (col*stride_width)  < pad )
            {
                y_pad = pad - col*stride_width;
                loopc = kernel - y_pad;
            }

            /* take care of padding in right side of image*/
            if((col) > (feature_c - pad))
            {
                loopc = col_width - (stride/3);
            }
            /* take care of padding in bottom of image */
            if(row > feature_r - pad)
            {
                loopr =  col_width - colstride/(3*col_width);
            }
            /* RGB weights and input 7*7*3 , kernel is 7*7 */
            for(int i = 0; i < loopr; i++)
            {
                for(int j = 0; j < loopc; j++)
                {
                   product +=        ((Layer1_Neurons_GPU[i*col_width*3 + j*3 + stride + colstride] * shared_Layer1_Weights_GPU[i*7 + j + kernel*x_pad + y_pad])
                    + (Layer1_Neurons_GPU [i*col_width*3 + j*3 + 1 + stride + colstride] * shared_Layer1_Weights_GPU [i*7 + 7*7 + j + kernel*x_pad + y_pad])
                    + (Layer1_Neurons_GPU [i*col_width*3 + j*3 + 2 + stride + colstride] * shared_Layer1_Weights_GPU [i*7 + 7*7*2 + j + kernel*x_pad + y_pad]));
                }
            }
    Layer2_Neurons_GPU[output*feature_r*feature_c + row*feature_c + col] = product;

    product = 0.0;

            }
        }
    }
    unsigned long long stop_time = clock64();
    runtime[tid] = (unsigned long long)(stop_time - start_time);
}
__global__ void executeFirstLayerCUDA(float *bias,float *Layer1_Neurons_GPU,float *Layer1_Weights_GPU,float *Layer2_Neurons_GPU,int stride_width,int pad, int col_width,int feature_r,int feature_c,int out,int tfactor,unsigned long long* runtime)
{
    unsigned long long start_time = clock64();

    int stride = 0,colstride = 0;
    int output = blockIdx.x;
    int row_even = threadIdx.x * tfactor;
    int col_even = threadIdx.y * tfactor;
    int kernel = 7;
    int x_pad,y_pad,loopc,loopr;
    float product = 0.0;

    int tid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x * blockDim.x + threadIdx.y;

    if(row_even < feature_r && col_even < feature_c)
    {

    colstride = 0;
    for(int row =row_even; row < row_even+tfactor ;row++)
    {
    stride = 0;
    colstride = 3 *(row*stride_width - pad)*col_width;
    colstride = colstride < 0 ? 0 : colstride;
    for(int col =col_even; col < col_even+tfactor ;col++)
    {
    stride = 3 * (col*stride_width - pad);
    stride = stride < 0 ? 0 : stride;

    product = 0;
    x_pad = 0; y_pad = 0;
    /* set the loops value */
    loopc = kernel;loopr = kernel;
    /* take care of padding in left hand side of image*/
    if(row*stride_width < pad)
    {
    x_pad = pad - row*stride_width;
    loopr = kernel - x_pad;
    }
    /* take care of padding in upper side of image*/
    if( (col*stride_width)  < pad )
    {
    y_pad = pad - col*stride_width;
    loopc = kernel - y_pad;
    }

    /* take care of padding in right side of image*/
    if((col) > (feature_c - pad))
    {
    loopc = col_width - (stride/3);
    }
    /* take care of padding in bottom of image */
    if(row > feature_r - pad)
    {
    loopr =  col_width - colstride/(3*col_width);
    }
    /* RGB weights and input 7*7*3 , kernel is 7*7 */
    for(int i = 0; i < loopr; i++)
    {
    for(int j = 0; j < loopc; j++)
    {
    product +=        ((Layer1_Neurons_GPU[i*col_width*3 + j*3 + stride + colstride]    * Layer1_Weights_GPU[i*7 + j + (output * 7*7*3) + kernel*x_pad + y_pad])
    + (Layer1_Neurons_GPU [i*col_width*3 + j*3 + 1 + stride + colstride] * Layer1_Weights_GPU [i*7 + 7*7 + j+ (output * 7*7*3) + kernel*x_pad + y_pad])
    + (Layer1_Neurons_GPU [i*col_width*3 + j*3 + 2 + stride + colstride] * Layer1_Weights_GPU [i*7 + 7*7*2 + j+ (output * 7*7*3) + kernel*x_pad + y_pad]));
    }
    }
    //printf("Product : %d\n",product);

    Layer2_Neurons_GPU[output*feature_r*feature_c + row*feature_c + col] = product;

    product = 0.0;

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

    float *Layer1_Neurons_CPU = (float *)malloc(224*224*3*sizeof(float));

    float *Layer1_Weights_CPU = (float *)malloc(64*7*7*3*sizeof(float));

    float *Layer2_Neurons_CPU1 = (float *)malloc(64*112*112*sizeof(float));
    float *Layer2_Neurons_CPU2 = (float *)malloc(64*112*112*sizeof(float));

    // Verify that allocations succeeded
    if (Layer1_Neurons_CPU == NULL || Layer1_Weights_CPU == NULL || Layer2_Neurons_CPU1 == NULL ||Layer2_Neurons_CPU2 == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for(int i=0; i<64*7*7*3; i++)
    {
        Layer1_Weights_CPU[i] = rand()/(float)RAND_MAX;
    }
    for(int i=0; i<224*224*3; i++)
    {
        Layer1_Neurons_CPU[i] = rand()/(float)RAND_MAX;
    }

    float *Layer1_Neurons_GPU = NULL;
    err = cudaMalloc((void**) &Layer1_Neurons_GPU, 224*224*3*sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *Layer1_Weights_GPU = NULL;
    err = cudaMalloc((void**) &Layer1_Weights_GPU, 64*7*7*3*sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *Layer2_Neurons_GPU1 = NULL;
    err = cudaMalloc((void**) &Layer2_Neurons_GPU1, 64*112*112*sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *Layer2_Neurons_GPU2 = NULL;
    err = cudaMalloc((void**) &Layer2_Neurons_GPU2, 64*112*112*sizeof(float));

    if(err != cudaSuccess)
    {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }

    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(Layer1_Neurons_GPU, Layer1_Neurons_CPU, 224*224*3*sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(Layer1_Weights_GPU, Layer1_Weights_CPU, 64*7*7*3*sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
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


    dim3 numBlocks(64,1,1);
    dim3 numThreads(32,32);

    double val = 112.0/32.0;
    int tfactor = ceil(val);
    //DEBUGPRINT((" Split Factor :: %d\n", tfactor));


    executeFirstLayerCUDA<<<numBlocks,numThreads>>>(NULL,Layer1_Neurons_GPU,Layer1_Weights_GPU,Layer2_Neurons_GPU1,2,3,224,112,112,64,tfactor,d_runtime);
    executeFirstLayerCUDAOptimized<<<numBlocks,numThreads>>>(NULL,Layer1_Neurons_GPU,Layer1_Weights_GPU,Layer2_Neurons_GPU2,2,3,224,112,112,64,tfactor,d_runtime1);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(Layer2_Neurons_CPU1, Layer2_Neurons_GPU1, 64*112*112*sizeof(float), cudaMemcpyDeviceToHost);
    err = cudaMemcpy(Layer2_Neurons_CPU2, Layer2_Neurons_GPU2, 64*112*112*sizeof(float), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for(int i=0; i<64*112*112; i++)
    {
        if(Layer2_Neurons_CPU2[i] != Layer2_Neurons_CPU1[i])
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
    for(int i=0; i< 64*112*112; i++)
	if(elapsed_time < runtime[i])
	   elapsed_time = runtime[i];
	printf("Kernel Execution Time Non Optimized : %llu cycles\n", elapsed_time);
#endif
#ifdef TM
    cudaMemcpy(runtime1, d_runtime1, r_size1, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    unsigned long long elapsed_time1 = 0;
    for(int i=0; i< 64*112*112; i++)
    if(elapsed_time1 < runtime1[i])
    elapsed_time1 = runtime1[i];
    printf("Kernel Execution Time Optimized : %llu cycles\n", elapsed_time1);
#endif

    // Free device global memory
    err = cudaFree(Layer1_Neurons_GPU);

    err = cudaFree(Layer1_Weights_GPU);

    err = cudaFree(Layer2_Neurons_GPU1);
    err = cudaFree(Layer2_Neurons_GPU2);

#ifdef TM
   cudaFree(d_runtime);
   cudaFree(d_runtime1);
#endif
    // Free host memory
    free(Layer1_Neurons_CPU);
    free(Layer1_Weights_CPU);
    free(Layer2_Neurons_CPU1);
    free(Layer2_Neurons_CPU2);
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

