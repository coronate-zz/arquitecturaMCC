default:
	gcc	-g 	main.c 	cache.c -lm
	./a.out -us 8192 -bs 16 -a 1 -wa -wb trazas/spice10.trace

	#gdb a.out

ex1:
	gcc -g main.c cache.c -lm
	./a.out -us 8192 -bs 16 -a 1 -wa -wb trazas/spice10.trace
	#gdb a.out 
	#r -us 8192 -bs 16 -a 1 -wa -wb trazas/cc.trace


