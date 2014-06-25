#include <stdio.h>
#include <cuda.h>
#define MAX_TILE_SIZE 32
#define MAX_MASK_WIDTH 11
/*Declare the constant memory*/
__constant__ float M[MAX_MASK_WIDTH];
/***********************/
/** TODO, write KERNEL */
/***********************/
__global__ void Conv1D(float* N, float* P, int Mask_Width, int Width) 
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ float N_ds[MAX_TILE_SIZE+MAX_MASK_WIDTH-1];
    int n = Mask_Width/2;
    /******************************************************************/
    /* Your TODO-1 starts here:                                       */ 
    /* Load the data with halo from N to the shared memory N_ds       */ 
    /*  remember that you need to load:                               */
    /*  + the left halo                                               */
    /*  + the data                                                    */    
    /*  + the right halo                                              */    
    /******************************************************************/
    int halo_index_left = (blockIdx.x-1) *blockDim.x + threadIdx.x;
    int halo_index_right = (blockIdx.x+1)*blockDim.x + threadIdx.x;
    if(threadIdx.x>=blockDim.x-n)
        N_ds[threadIdx.x-(blockDim.x-n)] = (halo_index_left <0) ? 0:N[halo_index_left];
    N_ds[n+threadIdx.x] = N[blockIdx.x*blockDim.x+threadIdx.x];
    if(threadIdx.x<n)
        N_ds[n+blockDim.x+threadIdx.x] = (halo_index_right >= Width) ? 0:N[halo_index_right];
    __syncthreads();
    /***********************/
    /*Your TODO-1 ends here*/               
    /***********************/
    /******************************************************************/
    /* Your TODO-2 starts here:                                       */ 
    /* Calculate the value coresponding to each thread                */ 
    /* The result is saved into the array P                           */
    /* It should be noted that the mask M is already copy to the      */                                                  
    /* constant memory                                                */    
    /******************************************************************/
    float Pvalue = 0;
    for(int j=0;j<Mask_Width;j++) Pvalue += N_ds[threadIdx.x+j]*M[j];
    P[i] = Pvalue;
    /***********************/
    /*Your TODO-2 ends here*/               
    /***********************/   
}
/**/
void test(float* C, int length);
void checkCUDAError(const char *msg);
/**/
int main(int argc, char* argv[])
{
    int i;
    /*******************/
    /** READING INPUT **/
    /*******************/
    /* dimension of mask */
    int size_m = 0;
    scanf("%d", &size_m);
    int full_size_m = size_m*sizeof(float);
    float* h_M = (float*)malloc(full_size_m);
    for(i=0;i<size_m;++i){ scanf("%f", &h_M[i]);}
    /* dimension of array */
    int size = 0; 
    scanf("%d", &size);
    int full_size = sizeof(float)*size;
    /* Allocate host memory */
    float* h_N = (float*)malloc(full_size);
    float* h_P = (float*)malloc(full_size);
    for(i=0;i<size;++i){ scanf("%f", &h_N[i]);}
    /********************/
    /** FINISHED INPUT **/
    /********************/
    /*************************/
    /* allocate device memory */
    /*************************/
    float* d_N,*d_P;
    cudaMalloc(&d_N, full_size);
    cudaMalloc(&d_P, full_size);
    /******************************/
    /* copy array & mask to device */
    /******************************/
    cudaMemcpy(d_N,h_N,full_size,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(M,h_M,full_size_m);
    /****************/
    /** CALL KERNEL */ 
    /****************/
    int threadsPerBlock = size;
    Conv1D<<<1, threadsPerBlock>>>(d_N, d_P,size_m, size);
    checkCUDAError("Kernel Invoking");
    /**************************/
    /*   copy result back     */
    /**************************/
    cudaMemcpy(h_P, d_P, full_size, cudaMemcpyDeviceToHost);
    /*******************************************/
    /*  Testing output, don't change anything! */
    /*******************************************/
    test(h_P, size);
    free(h_N);
    free(h_P);
    cudaFree(d_N);
    cudaFree(d_P);
    return 0;
}
/* to test the input, don't change anything! */
void test(float* C, int length){
    int i=0;
    for(i=0;i<length;++i){
        printf("%.1f ", C[i]);
    }
    printf("\n");
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
