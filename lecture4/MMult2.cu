#include <stdio.h>
#include <cuda.h>

/* Matrices are stored in row-major order: */
/* M(row, col) = (M.width*row +col); */
typedef struct{
    /* suppose we use only square matrices */
    int width;
    float *elements;
} Matrix;

/* Thread block size */
#define TILE_WIDTH 2

/***********************/
/*  TODO, write KERNEL */
/***********************/
__global__ void MatMul(const float* Md, const float* Nd, float* Pd, const int Width)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    // Identify the row and column of the Pd element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;
    //Loop over the Md and Nd tiles required to compute the Pd element
    for (int m = 0; m < Width/TILE_WIDTH; m++) {
    // Coolaborative loading of Md and Nd tiles into shared memory
    Mds[ty][tx] = Md[Row*Width + (m*TILE_WIDTH + tx)];
    Nds[ty][tx] = Nd[Col + (m*TILE_WIDTH + ty)*Width];
    __syncthreads();
    for (int k = 0; k < TILE_WIDTH; k++) Pvalue += Mds[ty][k] * Nds[k][tx];
    __syncthreads();
}
    Pd[Row*Width+Col] = Pvalue;
}
/**/
void test(const Matrix C);
/**/
int main(int argc, char* argv[])
{	
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
    int full_size = sizeof(float)*size*size;
    h_A.width = size;
    h_B.width = size;
    h_C.width = size;
    /* Allocate host memory */
    h_A.elements = (float*)malloc(full_size);
    h_B.elements = (float*)malloc(full_size);
    h_C.elements = (float*)malloc(full_size);
    /**/
    for(i=0;i<size*size;++i){ scanf("%f", &h_A.elements[i]);}
    for(i=0;i<size*size;++i){ scanf("%f", &h_B.elements[i]);}
    /********************/
    /** FINISHED INPUT **/
    /********************/
    /*************************/
    /*  allocate device      */
    /*	memory for A,B,C     */
    /*************************/
    Matrix d_A, d_B, d_C;
    d_A.width = size;
    d_B.width = size;
    d_C.width = size;
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
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(h_B.width/dimBlock.x, h_A.width/dimBlock.y);
    MatMul<<<dimGrid,dimBlock>>>(d_A.elements, d_B.elements, d_C.elements,d_A.width);
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
void test(const Matrix C)
{
  int i,j;
  //int size = C.width*C.width;
  for(i=0;i<C.width;++i)
  {
    for(j=0;j<C.width;++j) printf("%4.1f  ", C.elements[i*C.width+j]);
    printf("\n");
  }
}
