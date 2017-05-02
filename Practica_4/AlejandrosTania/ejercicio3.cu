#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>

#define TAM_BLOQUE 16
#define TAM_MAt 4096

__global__ void matrix_mult(int *d_A, int *d_B, int *d_C, int N ) 
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    //printf("---------PRUEBAS----------- \nblockIdx.x: %d     blockDim.x: %d     threadIdx.x: %d \nblockIdx.y: %d     blockDim.y: %d     threadIdx.y: %d \n       col: %d  row: %d  \n",blockIdx.x, blockDim.x, threadIdx.x ,blockIdx.y, blockDim.y, threadIdx.y, col, row  );

    __shared__ int temp[4096];

    temp[col] = d_A[col] * d_B[row];
// Thread 0 sums the pairwiseproducts

    __syncthreads();

    if( 0 == threadIdx.x ) 
    {
        int sum = 0;
        for( int i= 0; i< N; i++ )
        {
            sum += temp[i];
            *d_C = sum;
        }
    }
}


int main(int argc, char *argv[]) {
    clock_t begin=clock();  // Para medir cuánto tarda

    int *h_A, *h_B, *h_C; /* Arreglos del CPU */
    int *d_A, *d_B, *d_C; /* Arreglos del GPU */
    int N = TAM_MAt;
    char *verbose;
    if(argc > 2) verbose = argv[2];
    else verbose = NULL;

    size_t sz = N * N * sizeof(int);

    h_A = (int*) malloc(sz);
    h_B = (int*) malloc(sz);
    h_C = (int*) malloc(sz);

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



    /* Copiar los vectores del CPU al GPU */
    cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sz, cudaMemcpyHostToDevice);

    dim3 dimBlock(TAM_BLOQUE, TAM_BLOQUE);
    int idx = (N-1)/TAM_BLOQUE + 1;
    dim3 dimGrid(idx, idx);
    matrix_mult<<<dimGrid,dimBlock>>>(d_A, d_B, d_C, N);


    /* Esperar a que todos los threads acaben y checar por errores */
    cudaThreadSynchronize();

    /* Copiar el resultado de nuevo al CPU */
    cudaMemcpy(h_C, d_C, sz, cudaMemcpyDeviceToHost);

    clock_t end=clock();  // Checa el tiempo inmediatamente después de terminar
    double diffticks=end-begin;
    double diffms=(diffticks*10)/CLOCKS_PER_SEC;
    
    printf("Tiempo usado: %f mSeg\n\n", diffms);


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

