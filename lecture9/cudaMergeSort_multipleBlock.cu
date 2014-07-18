/*
  MULTIPLE-BLOCK VERSION
  Assume that all sub-blocks in each half are located
  You are given 2 functions to calculate the rank of a key: get_rank_inclusive and get_rank_exclusive
  The input data is read from the stdin, it contains 5 lines:
  + Line 1: half_size, the number of elements in each half (half_size = n/2)
  + Line 2: input data (half_size*2 elements)
  + Line 3: sb_num, the number of sub-blocks in each half
  + Line 4: the array sb_len_left (the length of each sub-block in the left half)
  + Line 5: the array sb_len_right (the length of each sub-block in the right half)
*/

#include <stdio.h>
#include <cuda.h>
/**/
#define SAMPLE_INTERVAL 4 /* pick a sample every 4 elements */
/**/
/* 
   Inline device function, to compute a rank of a "key" in an array "arr" 
   of length "len" (including this key)
*/
static inline __device__ int get_rank_inclusive(int key, int* arr, int len);
/* 
   Inline device function, to compute a rank of a "key" in an array "arr" 
   of length "len" (excluding this key)
*/
static inline __device__ int get_rank_exclusive(int key, int* arr, int len);
/**/
__global__ void pairwise_merge(int* input, int half_size, int* sb_len_left, int* sb_len_right, int sb_num, int* output)
{
  int i, other_rank, output_rank;
  /**/
  int* left_half = input;
  int* right_half = input + half_size; 
  int* cur_output = output;
  /**/
  for(i=0;i<blockIdx.x;++i) 
  {
    left_half += sb_len_left[i];
    right_half += sb_len_right[i];
    cur_output = cur_output + sb_len_left[i] + sb_len_right[i];
  }
  if(threadIdx.x < sb_len_left[blockIdx.x])
  {
    int key = left_half[threadIdx.x];
    /* use function get_rank_exclusive() to calculate the rank 
    of key in the left_right*/
    other_rank = get_rank_exclusive(key,right_half,sb_len_right[blockIdx.x]);
    /* calculate the output rank of key */
    output_rank = threadIdx.x + other_rank;
    /* assign key to the correspoding position in the output array*/
    cur_output[output_rank] = key;
  }
  /**/
  /********************************************/
  if(threadIdx.x < sb_len_right[blockIdx.x])
  {
    int key = right_half[threadIdx.x];
    /* use function get_rank_inclusive() to calculate the rank 
    of key in the left_half*/
    other_rank = get_rank_inclusive(key,left_half,sb_len_left[blockIdx.x]);
    /* calculate the output rank of key */
    output_rank = threadIdx.x + other_rank;
    /* assign key to the correspoding position in the output array*/
    cur_output[output_rank] = key;
  }
  /**/
}/* end of the kernel*/
/**/
void checkCUDAError(const char *msg);
/**/
int main(int argc, char* argv[])
{   
    int i;
    /**/
    int* h_input, *h_output;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    /*******************/
    /** READING INPUT **/
    /*******************/
    int half_size,size;
    int sb_num; //number of sub-block
    /* read the value of half_size from stdin*/
    scanf("%d", &half_size);
    size = half_size*2;
    /* Allocate host memory */
    h_input = (int*) malloc(sizeof(int)*size);
    h_output = (int*) malloc(sizeof(int)*size);
    /* read input from stdin */
    for(i=0;i<size;++i) scanf("%d", &h_input[i]);
    /* read the value of sb_num */
    scanf("%d", &sb_num);
    int *h_sb_len_left, *h_sb_len_right;
    h_sb_len_left = (int*) malloc(sizeof(int)*sb_num);
    h_sb_len_right = (int*) malloc(sizeof(int)*sb_num);
    for(i=0;i<sb_num;++i) scanf("%d", &h_sb_len_left[i]);
    for(i=0;i<sb_num;++i) scanf("%d", &h_sb_len_right[i]);
    /**/
    /****************************/
    /** FINISHED INPUT READING **/
    /****************************/
    /******************************/
    /*  allocate device memories  */
    /******************************/
    int* d_input, *d_output, *d_sb_len_left, *d_sb_len_right;
    cudaMalloc(&d_input,sizeof(int)*size);
    cudaMalloc(&d_output,sizeof(int)*size);
    cudaMalloc(&d_sb_len_left,sizeof(int)*sb_num);
    cudaMalloc(&d_sb_len_right,sizeof(int)*sb_num);
    cudaEventRecord(start,0); 
    /***********************************/
    /*      copy input data to device  */
    /***********************************/
    cudaMemcpy(d_input, h_input, size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sb_len_left, h_sb_len_left, sb_num*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sb_len_right, h_sb_len_right, sb_num*sizeof(int), cudaMemcpyHostToDevice);
    /* invoke the kernel, with sb_num block, SAMPLE_INTERVAL threads */
    pairwise_merge<<<sb_num,SAMPLE_INTERVAL>>>(d_input,half_size,d_sb_len_left,d_sb_len_right,sb_num,d_output);
    checkCUDAError("kernel invocation\n");
    /* copy the sorted results back to host */
    cudaMemcpy(h_output, d_output, sizeof(int)*size, cudaMemcpyDeviceToHost);
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
    printf("The sorted array is :\n");
    for(int i=0;i<size;++i) printf("%d  ",h_output[i]);
    printf("\n");
    /* free device memory */
    cudaFree(d_sb_len_left);
    cudaFree(d_sb_len_right);
    cudaFree(d_input);
    cudaFree(d_output);
    /* free host memory */
    free(h_input);
    free(h_output);
    free(h_sb_len_left);
    free(h_sb_len_right);
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
/* 
   Inline device function, to compute a rank of a "key" in an array "arr" 
   of length "len" (including this key)
   Naive implementation. 
   Binary search can be used to implement more efficient function
*/
static inline __device__ int get_rank_inclusive(int key, int* arr, int len)
{
  int rank=0;
  while((rank < len) && (arr[rank]<=key)) ++rank;
  return rank;
}
/* 
   Inline device function, to compute a rank of a "key" in an array "arr" 
   of length "len" (excluding this key)
   Naive implementation. 
   Binary search can be used to implement more efficient function
*/
static inline __device__ int get_rank_exclusive(int key, int* arr, int len)
{
  int rank=0;
  while((rank < len) && (arr[rank]<key)) ++rank;
  return rank;
}
