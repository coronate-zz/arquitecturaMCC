/* 
 * File:   mandel.c
 * Author: davidr
 *
 * Created on May 22, 2013, 9:42 AM
 */

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

# define NPOINTS 2000
# define MAXITER 2000

struct complex{
  double real;
  double imag;
};


void checkCUDAError(const char*);

__global__ void mandel_numpoints(int *d_np){

  double ztemp;
  int iter;
  struct complex z, c;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  c.real = -2.0+2.5*(double)(i)/(double)(NPOINTS)+1.0e-7;
  c.imag = 1.125*(double)(j)/(double)(NPOINTS)+1.0e-7;
  z=c;
  d_np[idx + NPOINTS*idy] = 0;
  for (iter=0; iter<MAXITER; iter++){
    ztemp=(z.real*z.real)-(z.imag*z.imag)+c.real;
    z.imag=z.real*z.imag*2+c.imag; 
    z.real=ztemp; 
    if ((z.real*z.real+z.imag*z.imag)>4.0e0) {
      d_np[idx + NPOINTS*idy] = 1;
      break;
    }
  }
}


/*
 * 
 */
int main(int argc, char** argv) {
  int numoutside = 0;
  double area, error;//, ztemp;
  //struct complex z, c;
  time_t t1,t2;

  int *h_np; /* Array to save numpoints in host */
  int *d_np;/* Array to save numpoints in device */

  size_t sz = NPOINTS * NPOINTS * sizeof(int);

  h_np = (int *) malloc(sz);

  cudaMalloc((void**) &d_np, sz);

  for(int i = 0; i < NPOINTS*NPOINTS; i++){
      h_np[i] = 0;
  }

  cudaMemcpy(d_np, h_np, sz, cudaMemcpyHostToDevice);

  dim3 dimGrid(100,100);
  dim3 dimBlock(20,20);

  t1 = time(NULL);
  mandel_numpoints<<<dimGrid,dimBlock>>>(d_np);

  cudaThreadSynchronize();
  checkCUDAError("kernel invocation");

  cudaMemcpy(h_np,d_np,sz,cudaMemcpyDeviceToHost);
  checkCUDAError("memcpy");
  t2 = time(NULL);

  for(int i=0; i < NPOINTS*NPOINTS; i++){
      if(h_np[i] > 0){
              numoutside++;
      }
  }

  area=2.0*2.5*1.125*(double)(NPOINTS*NPOINTS-numoutside)/(double)(NPOINTS*NPOINTS);
  error=area/(double)NPOINTS;

  printf("Area of Mandlebrot set = %12.8f +/- %12.8f\n",area,error);
  printf("Tiempo de ejecución: %f segundos \n",difftime(t2,t1));

  cudaFree(d_np);

  free(h_np);

  return 0;
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
  