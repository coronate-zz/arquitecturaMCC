
 
#include <stdio.h>
#include <stdlib.h>

#define N 10



int main(int argc, char *argv[])
{

    int *h_a, *h_b, *h_c; /* Arreglos del CPU */
    int *d_a, *d_b, *d_c;/* Arreglos del GPU */

    int i;
    size_t sz = N * sizeof(int);

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


    for (int row=0; row<N; row++){
        for (int col=0; col<N; col++){
                h_A[(row*N)+col]=(row*N)+col;
                h_B[(row*N)+col]=(row*N)+col;
                printf(" \n ROW: %d   COL: %d   INDEX: %d  ",   row, col, h_A[row*N+col]);
                printf("\n");
        }
    }

}