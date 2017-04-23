/*
 * 
 * Programa de Introducción a los conceptos de CUDA
 * 
 *
 * 
 * 
 */
 
#include <stdio.h>
#include <stdlib.h>

/* Kernel para sumar dos vectores en un sólo bloque de hilos */
__global__ void vect_add(int *d_a, int *d_b, int *d_c)
{
    int idx = threadIdx.x;
    int a_i = d_a[idx];
    int b_i = d_b[idx];
    d_c[idx] = a_i * b_i;

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

/* Versión de múltiples bloques de la suma de vectores */
__global__ void vect_add_multiblock(int *d_a, int *d_b, int *d_c)
{
    /* Part 2C: Implementación del kernel pero esta vez permitiendo múltiples bloques de hilos. */

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int a_i = d_a[idx];
    int b_i = d_b[idx];
    d_c[idx] = a_i * b_i;
    
}



/* Numero de elementos en el vector */
#define ARRAY_SIZE 256

/*
 * Número de bloques e hilos
 * Su producto siempre debe ser el tamaño del vector (arreglo).
 */
#define NUM_BLOCKS  6
#define THREADS_PER_BLOCK 256

/* Main routine */
int main(int argc, char *argv[])
{
    int *h_a, *h_b, *h_c; /* Arreglos del CPU */
    int *d_a, *d_b, *d_c;/* Arreglos del GPU */

    int i;
    size_t sz = ARRAY_SIZE * sizeof(int);

    /*
     * Reservar memoria en el cpu
     */
    h_a = (int *) malloc(sz);
    h_b = (int *) malloc(sz);
    h_c = (int *) malloc(sz);

    /*
     * Parte 1A:Reservar memoria en el GPU
     */
     cudaMalloc((void**) &d_a, sz);
     cudaMalloc((void**) &d_b, sz);
     cudaMalloc((void**) &d_c, sz);

//Si fuera necesario sacar el resultado en otro vector..
     //cudaMalloc((void**) &d_out, ARRAY_BYTES);



    /* inicialización */
    for (i = 0; i < ARRAY_SIZE; i++) {
        h_a[i] = i;  //a = 1 ,2 3, 4 ...
        h_b[i] = i;
        h_c[i] = 0;    //Todos los elementos de C son 0
    }

    /* Parte 1B: Copiar los vectores del CPU al GPU */
    //cudaMemcpy( );

    cudaMemcpy(d_a, h_a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, sz, cudaMemcpyHostToDevice);


    /* run the kernel on the GPU */
    /* Parte 2A: Configurar y llamar los kernels */
    /* dim3 dimGrid( ); */
    /* dim3 dimBlock( ); */
    vect_add<<< 1, ARRAY_SIZE >>>(d_a, d_b, d_c ); 


    /* Esperar a que todos los threads acaben y checar por errores */
    cudaThreadSynchronize();
    checkCUDAError("kernel invocation");

    /* Part 1C: copiar el resultado de nuevo al CPU */
    cudaMemcpy(h_c, d_c, sz, cudaMemcpyDeviceToHost);

    checkCUDAError("memcpy");

    /* print out the result */
    printf("Results: ");
    for (i = 0; i < ARRAY_SIZE; i++) {
      printf("%d, ", h_c[i]);
    }
    printf("\n\n");



    /* Ejecutando ejrcicios 2B */

    vect_add_multiblock<<< NUM_BLOCKS, ARRAY_SIZE/NUM_BLOCKS >>>(d_a, d_b, d_c ); 


    cudaMemcpy(h_c, d_c, sz, cudaMemcpyDeviceToHost);



    printf("Results Multiblock: ");
    for (i = 0; i < ARRAY_SIZE; i++) {
      printf("%d, ", h_c[i]);
    }
    printf("\n\n");
    /* Parte 1D: Liberar los arreglos */



    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}



