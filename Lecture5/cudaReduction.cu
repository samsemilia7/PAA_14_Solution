#include <stdio.h>
#include <cuda.h>
/* Thread block size = number of threads of a block*/
/* Notice: in this example, the input data size = 2*BLOCK_SIZE */
#define BLOCK_SIZE 8
/*The kernel*/
__global__ void Reduction(const float* input, float* output)
{
  /*Declare the shared memory*/
  __shared__ float partialSum[2*BLOCK_SIZE];
  /**/
  unsigned int t = threadIdx.x;
  unsigned int start = 2*blockIdx.x * blockDim.x;
  /*load data from global memory to shared memory*/
  partialSum[t] = input[start+t];
  partialSum[blockDim.x+t] = input[start+blockDim.x+t];
  /***************************************************************/
  /* YOUR TODO STARTS HERE                                       */
  /* calculate the elements in partialSum, based on stride and t */
  /* so that the ids of active threads are consecutive           */
  /* in order to reduce the number of thread divergences         */
  /***************************************************************/
  for(unsigned int stride = blockDim.x;stride>0;stride/=2)
  {
    __syncthreads();
    if(t<stride) partialSum[t] += partialSum[t+stride];
  }
  /***************************************************************/
  /*            YOUR TODO ENDS HERE                              */
  /***************************************************************/
  /* the final result is store in partialSum[0] => write to output */
  if(t == 0) output[0] = partialSum[0];
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
    h_output = (float*) malloc(sizeof(float));
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
    cudaMalloc(&d_output,sizeof(float));
    cudaEventRecord(start,0);
    /***********************************/
    /*      copy input data to device  */
    /***********************************/
    cudaMemcpy(d_input, h_input, size*sizeof(float), cudaMemcpyHostToDevice);
    /*************************************/
    /*       call kernel                 */
    /*       1 block, BLOCK_SIZE threads */
    /*************************************/
    Reduction<<<1,BLOCK_SIZE>>>(d_input, d_output);
    checkCUDAError("Kernel Invoking");
    /**************************/
    /*       copy result back */
    /**************************/
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    /**/
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(stderr,"Elapsed time = %f (s)\n",elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    /*******************************************/
    /* Print the final reduction result        */
    /*******************************************/
    printf("The final reduction of the array: %4.1f\n",h_output[0]);
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
