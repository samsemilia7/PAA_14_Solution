#include <stdio.h>
#include <cuda.h>

/* Matrices are stored in row-major order: */
/* M(row, col) = (M.width*row +col); */
typedef struct{
    /* suppose we use only square matrices */
    int width;
    int *elements;
} Matrix;

/* Thread block size */
#define BLOCK_SIZE 2

/***********************/
/*  TODO, write KERNEL */
/***********************/
__global__ void MatMul(const Matrix A, const Matrix B, Matrix C){
    
	int Cvalue = 0;
	int i;
	int size = A.width;
	int col = blockIdx.x*blockDim.x+threadIdx.x;
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	for(i=0;i<size;++i){
		Cvalue += A.elements[row*size+i]*B.elements[i*size+col];
	}
	C.elements[row*size+col] = Cvalue;
}
void test(const Matrix C);

int main(int argc, char* argv[]){
	
    int i;
	/* init matrices */
	Matrix h_A, h_B, h_C;
	cudaEvent_t start; 
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	/*******************/
    /** READING INPUT **/
    /*******************/
    int size = 0; //dimension of matrices
    scanf("%d", &size);
	int full_size = sizeof(int)*size*size;
	h_A.width = size;h_B.width = size;h_C.width = size;
    /* Allocate host memory */
	h_A.elements = (int*)malloc(full_size);
    h_B.elements = (int*)malloc(full_size);
    h_C.elements = (int*)malloc(full_size);
    
	for(i=0;i<size*size;++i){ scanf("%d", &h_A.elements[i]);}
    for(i=0;i<size*size;++i){ scanf("%d", &h_B.elements[i]);}
    /********************/
    /** FINISHED INPUT **/
    /********************/
    
	/*************************/
    /*  allocate device      */
    /*	memory for A,B,C     */
    /*************************/
    Matrix d_A, d_B, d_C;
	d_A.width = size;d_B.width = size;d_C.width = size;
	cudaMalloc(&d_A.elements, full_size);
    cudaMalloc(&d_B.elements, full_size);
    cudaMalloc(&d_C.elements, full_size);
	cudaEventRecord(start,0);
	/***********************************/
    /*      copy vectors A&B to device */
    /***********************************/
    cudaMemcpy(d_A.elements, h_A.elements, full_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.elements, h_B.elements, full_size, cudaMemcpyHostToDevice);
    
	/*********************/
    /*       call kernel */
    /*********************/
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(h_B.width/dimBlock.x, h_A.width/dimBlock.y);
	MatMul<<<dimGrid,dimBlock>>>(d_A, d_B, d_C);
	/**************************/
	/*       copy result back */
	/**************************/
    cudaMemcpy(h_C.elements, d_C.elements, full_size, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    fprintf(stderr,"Elapsed time = %f (s)\n",elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
	/*******************************************/
    /** Testing output, don't change anything! */
    /*******************************************/
    test(h_C);
    /* free device memory */
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
    /* free host memory */
	free(h_A.elements);
	free(h_B.elements);
	free(h_C.elements);
	
	return 0;
}
//function to test the input, don't change anything!
void test(const Matrix C){
    int i,j;
    //int size = C.width*C.width;
    for(i=0;i<C.width;++i)
    {
        for(j=0;j<C.width;++j) printf("%d  ", C.elements[i*C.width+j]);
        printf("\n");
    }
}