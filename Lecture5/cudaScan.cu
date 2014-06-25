#include <stdio.h>
#include <cuda.h>
/* Thread block size = number of threads of a block*/
/* Notice: in this example, the input data size = BLOCK_SIZE */
/*         (different with the CUDA Reduction assigment)     */
#define BLOCK_SIZE 16
__global__ void work_efficient_scan(const float* input, float* output, int size)
{
  /*Declare the shared memory*/
  __shared__ float XY[BLOCK_SIZE];
  /**/
  unsigned int t = threadIdx.x;
  /*load data from global memory to shared memory*/
  XY[t] = input[t];
  /*****************************************************************/
  /*  YOUR TODO-1 STARTS HERE                                      */
  /*  Implement the Reduction step,                                */
  /*  (the final results is kept in the last element               */
  /*****************************************************************/  
  for(int stride = 1;stride<blockDim.x;stride*=2)
  {
    __syncthreads();
    int index = (t+1)*stride*2-1;
    if(index < blockDim.x) XY[index]+=XY[index-stride];
  }
  /*************************************************************/
  /*    YOUR TODO-1 ENDS HERE                                  */
  /*************************************************************/
  /*************************************************************/
  /*  YOUR TODO-2 STARTS HERE                                  */
  /*  Implement the "post scan" step                           */ 
  /*    to finish the inclusive scan                           */
  /*************************************************************/
  for(int stride=size/4;stride>0;stride/=2)
  {
    __syncthreads();
    int index = (t+1)*stride*2-1;
    if(index+stride < blockDim.x) XY[index+stride] += XY[index];
  }
  /*************************************************************/
  /*    YOUR TODO-2 ENDS HERE                                  */
  /*************************************************************/
  __syncthreads();
  /* write the final output to global memory */
  output[t] = XY[t];
}
/**/
void checkCUDAError(const char *msg);
/**/
int main(int argc, char* argv[])
{   
    int i;
    /**/
    float* h_input, *h_output;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    /*******************/
    /** READING INPUT **/
    /*******************/
    int size = 0; //dimension of matrices
    /* read the value of size from stdin*/
    scanf("%d", &size);
    /* Allocate host memory */
    h_input = (float*) malloc(sizeof(float)*size);
    h_output = (float*) malloc(sizeof(float)*size);
    /* read input from stdin */
    for(i=0;i<size*size;++i){ scanf("%f", &h_input[i]);}
    /********************/
    /** FINISHED INPUT **/
    /********************/
    /*************************/
    /*  allocate device      */
    /*    memory for A,B,C     */
    /*************************/
    float* d_input, *d_output;
    cudaMalloc(&d_input,sizeof(float)*size);
    cudaMalloc(&d_output,sizeof(float)*size);
    cudaEventRecord(start,0);
    /***********************************/
    /*      copy input data to device  */
    /***********************************/
    cudaMemcpy(d_input, h_input, size*sizeof(float), cudaMemcpyHostToDevice);
    /*************************************/
    /*       call kernel                 */
    /*       1 block, BLOCK_SIZE threads */
    /*************************************/
    work_efficient_scan<<<1,BLOCK_SIZE>>>(d_input, d_output,size);
    checkCUDAError("Kernel Invoking");
    /**************************/
    /*       copy result back */
    /**************************/
    cudaMemcpy(h_output, d_output, sizeof(float)*size, cudaMemcpyDeviceToHost);
    /**/
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(stderr,"Elapsed time = %f (s)\n",elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    /*******************************************/
    /* Print the final scan result             */
    /*******************************************/
    printf("The final inclusive scan result:\n");
    for(int i=0;i<size;++i)printf("%4.1f  ",h_output[i]);
    /* free device memory */
    cudaFree(d_input);
    cudaFree(d_output);
    /* free host memory */
    free(h_input);
    free(h_output);
    /**/
    return 0;
}
/*function to test CUDA command*/
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg,
        cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                        
}
