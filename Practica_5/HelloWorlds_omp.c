#include <stdio.h>
#include <omp.h>
#define NUM_THREADS 2

int main() {

	#pragma omp parallel num_threads(6)
	{
		int i;
		printf("Hello World\n");
		for(i=0;i<6;i++)
			printf("Iter:%d\n",i);
	}
		printf("GoodBye World\n");

}