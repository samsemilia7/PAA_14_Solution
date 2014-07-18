#include <stdio.h>
#include <cuda.h>
#define BLOCK_SIZE 2
/***********************/
__global__ void SpMV_ELL(int num_rows,float* data, int* col_index, int num_elem, float* x, float* y)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  /**************************************************************/
  /*  YOUR TODO STARTS HERE    				*/
  /*  Perform the multiplication between matrix M and vector x  */
  /*  The result is store in vector y	 	      	     	*/
  /**************************************************************/
  if(row < num_rows)
  {
    float dot = 0.0;
    for(int i=0;i<num_elem;i++)
      dot += data[row+i*num_rows] * x[col_index[row+i*num_rows]];
    y[row] = dot;
  }
  /**************************************************************/
  /*  YOUR TODO ENDS HERE					*/
  /**************************************************************/
}
/**/
void checkCUDAError(const char *msg);
/**/
int main(int argc, char* argv[])
{   
    int i;
    /**/
    int num_rows, num_elem;
    float* h_x, *h_data, *h_y;
    int *h_col_index;
    /**/
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    /*******************/
    /** READING INPUT **/
    /*******************/
    scanf("%d",&num_rows);
    scanf("%d",&num_elem);
    /* allocation for matrix M and vector x*/
    /* data and col_index contains num_rows x num_elem elements */
    h_data = (float*) malloc(sizeof(float)*num_rows*num_elem);
    h_col_index = (int*) malloc(sizeof(int)*num_rows*num_elem);
    /* vector x contains num_rows elements */
    h_x = (float*) malloc(sizeof(float)*num_rows);
    /* the result vector contains num_rows elements */
    h_y = (float*) malloc(sizeof(float)*num_rows);
    /* reading matrix and vector from stdin*/
    for(i=0;i<num_rows*num_elem;++i){ scanf("%f", &h_data[i]);}
    for(i=0;i<num_rows*num_elem;++i){ scanf("%d", &h_col_index[i]);}
    for(i=0;i<num_rows;++i){ scanf("%f", &h_x[i]);}
    /********************/
    /** FINISHED INPUT **/
    /********************/
    /*************************/
    /*  allocate device      */
    /*  memory for A,B,C     */
    /*************************/
    float *d_data, *d_x, *d_y;
    int *d_col_index;
    /**/
    cudaMalloc(&d_data,sizeof(float)*num_rows*num_elem);
    cudaMalloc(&d_col_index,sizeof(int)*num_rows*num_elem);
    cudaMalloc(&d_x,sizeof(float)*num_rows);
    cudaMalloc(&d_y,sizeof(float)*num_rows);
    cudaEventRecord(start,0);
    /***********************************/
    /*      copy input data to device  */
    /***********************************/
    cudaMemcpy(d_data,h_data,sizeof(float)*num_rows*num_elem,cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_index,h_col_index,sizeof(float)*num_rows*num_elem,cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,h_x,sizeof(float)*num_rows,cudaMemcpyHostToDevice);
    /* Calculate the number of blocks */
    int num_block = (num_rows + BLOCK_SIZE - 1)/BLOCK_SIZE;
    /******************************************************/
    /*       call kernel 				  */
    /*       n_block blocks, BLOCK_SIZE threads per block */
    /******************************************************/
    SpMV_ELL<<<num_block,BLOCK_SIZE>>>(num_rows,d_data,d_col_index,num_elem,d_x,d_y);
    checkCUDAError("Kernel Invoking");
    /**************************/
    /*       copy result back */
    /**************************/
    cudaMemcpy(h_y,d_y,sizeof(float)*num_rows,cudaMemcpyDeviceToHost);
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
    printf("The result vector:\n");
    for(int i=0;i<num_rows;++i)printf("%4.1f  ",h_y[i]);
    /* free device memory */
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_data);
    cudaFree(d_col_index);
    /* free host memory */
    free(h_x);
    free(h_y);
    free(h_data);
    free(h_col_index);
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
