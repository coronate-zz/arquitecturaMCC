default:
	gcc	-g 	main.c 	cache.c
	./a.out 
	gdb a.out

ex1:
	gcc -g main.c cache.c 
	./a.out -us 8192 -bs 16 -a 1 -wa -wb trazas/cc.trace
	gdb a.out 
	#r -us 8192 -bs 16 -a 1 -wa -wb trazas/cc.trace


