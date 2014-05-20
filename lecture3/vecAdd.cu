#include <stdio.h>
#include <cuda.h>

void test(int* C, int length);
/***********************/
/*  TODO, write KERNEL */
/***********************/
__global__ void VecAdd(int* A, int* B, int* C, int N) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id<N){
        C[id] = A[id]+B[id];
    }
}
 
int main(int argc, char* argv[]){
    int i,z;
    int* h_A;
    int* h_B;
    int* h_C;
    /* going 3 rounds */
    for(z=0;z<3;++z){
    /*******************/
    /** READING INPUT **/
    /*******************/
    int size = 0; //length of input vectors
    scanf("%d", &size);
    /* Allocate host memory */
    h_A = (int*)malloc(sizeof(int)*size);
    h_B = (int*)malloc(sizeof(int)*size);
    h_C = (int*)malloc(sizeof(int)*size);
    
    for(i=0;i<size;++i){ scanf("%d", &h_A[i]); }
    for(i=0;i<size;++i){ scanf("%d", &h_B[i]); }
    /********************/
    /** FINISHED INPUT **/
    /********************/
    
    int* d_A;
    int* d_B;
    int* d_C;
    
    /*************************/
    /*  allocate device      */
    /*	memory for A,B,C     */
    /*************************/
    cudaMalloc(&d_A, sizeof(int)*size);
    cudaMalloc(&d_B, sizeof(int)*size);
    cudaMalloc(&d_C, sizeof(int)*size);
    
    /***********************************/
    /* TODO copy vectors A&B to device */
    /***********************************/
    cudaMemcpy(d_A, h_A, size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size*sizeof(int), cudaMemcpyHostToDevice);
        
    /*********************/
    /*       call kernel */
    /*********************/
    int threadsPerBlock = 256;
    int blocksPerGrid = (size+threadsPerBlock-1)/threadsPerBlock;
    VecAdd<<<blocksPerGrid,threadsPerBlock>>>(d_A, d_B, d_C, size);
    
	/**************************/
	/* TODO, copy result back */
	/**************************/
    cudaMemcpy(h_C, d_C, sizeof(int)*size, cudaMemcpyDeviceToHost);
    
    /*******************************************/
    /** Testing output, don't change anything! */
    /*******************************************/
    test(h_C, size);
    //free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    //free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    }
	return 0;
}
//function to test the input, don't change anything!
void test(int* C, int length){
    int i=0;
    int result = 0.0;
    for(i=0;i<length;++i){
        result += C[i];
    }
    printf("%d\n", result);
}