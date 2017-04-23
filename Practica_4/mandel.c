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

/*
 * 
 */
int main(int argc, char** argv) {
  int i, j, iter, numoutside = 0;
  double area, error, ztemp;
  struct complex z, c;
  int temp;
  time_t t1,t2;

/*
 *   
 *
 *     Ciclos exteriores sobre npuntos inicializan z=c
 *
 *     Los ciclos internos contienen el calculo de z=z*z+c, y el test |z|>2
 */
  t1 = time(NULL);
  for (i=0; i<NPOINTS; i++) {
    for (j=0; j<NPOINTS; j++) {
      c.real = -2.0+2.5*(double)(i)/(double)(NPOINTS)+1.0e-7;
      c.imag = 1.125*(double)(j)/(double)(NPOINTS)+1.0e-7;
      z=c;
      for (iter=0; iter<MAXITER; iter++){
	ztemp=(z.real*z.real)-(z.imag*z.imag)+c.real;
	z.imag=z.real*z.imag*2+c.imag; 
	z.real=ztemp; 
	if ((z.real*z.real+z.imag*z.imag)>4.0e0) {
	  numoutside++; 
	  break;
	}
      }
    }
  }
  t2 = time(NULL);

/*
 *  Cálculo del área y del error
 */

      area=2.0*2.5*1.125*(double)(NPOINTS*NPOINTS-numoutside)/(double)(NPOINTS*NPOINTS);
      error=area/(double)NPOINTS;

      printf("Area of Mandlebrot set = %12.8f +/- %12.8f\n",area,error);
      printf("Tiempo de ejecución: %f segundos \n",difftime(t2,t1));

    return 0;
}

