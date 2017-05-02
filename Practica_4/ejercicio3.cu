
 
#include <stdio.h>
#include <stdlib.h>


#define ARRAY_SIZE 256
#define N 10
#define NUM_BLOCKS  6
#define THREADS_PER_BLOCK 256


/* Kernel para sumar dos vectores en un sólo bloque de hilos */
__global__ void matrix_mult(int *d_A, int *d_B, int *d_C)
{

    __shared__ int temp[N];

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    temp[threadIdx.x] = d_A[index] * d_B[index];
    __syncthreads();
    /*
    d_c[idx] = a_i * b_i;
    if( 0 == threadIdx.x ) {
        intsum = 0;
        for( inti= 0; i< N; i++ )
            sum += temp[i];
        *d_C[index] = sum;
        }
    */

    printf("\nKERNEL>>>> A[i] :  %d   B[i]:  %d   C[i]:  %d  \n",  d_A[index], d_B[index], d_C[index]);

}



/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}

void changeMatrix(int * matrixB)
{
    int * auxB = matrixB;
    for (int row=0; row<N; row++)
    {
        for (int col=0; col<N; col++)
        {
                matrixB[(row*N)+col]=(row*N)+col;
                //printf(" \n ROW: %d   COL: %d   INDEX: %d  ",   row, col, h_A[row*N+col]);
                //printf("\n");
        }
    }
}


/* Main routine */
int main(int argc, char *argv[])
{
        int *h_A, *h_B, *h_C; /* Arreglos del CPU */
    int *d_A, *d_B, *d_C;/* Arreglos del GPU */

    int i;
    size_t sz = N * N * sizeof(int);

    /*
     * Reservar memoria en el cpu
     */
    h_A = (int *) malloc(sz);
    h_B = (int *) malloc(sz);
    h_C = (int *) malloc(sz);

    /*
     * Parte 1A:Reservar memoria en el GPU
     */
     cudaMalloc((void**) &d_A, sz);
     cudaMalloc((void**) &d_B, sz);
     cudaMalloc((void**) &d_C, sz);


    for (int row=0; row<N; row++)
    {
        for (int col=0; col<N; col++)
        {
                h_A[(row*N)+col]=(row*N)+col;
                h_B[(row*N)+col]=(row*N)+col;
                //printf(" \n ROW: %d   COL: %d   INDEX: %d  ",   row, col, h_A[row*N+col]);
                //printf("\n");
        }
    }




    cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sz, cudaMemcpyHostToDevice);


    matrix_mult<<< N, N >>>(d_A, d_B, d_C ); 


    cudaThreadSynchronize();
    checkCUDAError("kernel invocation");
    cudaMemcpy(h_C, d_C, sz, cudaMemcpyDeviceToHost);

    checkCUDAError("memcpy");

    /* print out the result 
    printf("Results: ");
    for (i = 0; i < ARRAY_SIZE; i++) {
      printf("%d, ", h_c[i]);
    }
    printf("\n\n");
*/
    int * cpu_C;
    int sum;
    for (int row=0; row<N; row++)
    {
        for (int col=0; col<N; col++)
        {
            sum = 0.f;
            for (int n=0; n<N; n++){
                sum += h_A[row*N+n]*h_B[n*N+col];
            }
            cpu_C[row*N+col] = sum;
            printf("\nsum= %d ", sum);

        }
    }


/*

      for (int ROW=0; ROW < N; ROW++)
    {
        for (int COL=0; COL < N; COL++)
        {
            err += cpu_C[ROW * N + COL] - h_C[ROW * N + COL];

        }
    }
*/

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}



