default:
	gcc	-g 	main.c 	cache.c -lm
	#./a.out -us 8192 -bs 16 -a 1 -wa -wb trazas/spice.trace
	./a.out -ds 8192 -is 8192 -bs 64 -a 2 -wa -wt trazas/spice.trace
	
	#gdb a.out

ex1:
	gcc -g main.c cache.c -lm
	./a.out -us 8192 -bs 16 -a 1 -wa -wb trazas/spice10.trace
	#gdb a.out 
	#r -us 8192 -bs 16 -a 1 -wa -wb trazas/cc.trace


default2:
	gcc	-g 	main.c 	cache.c -lm
	./a.out -us 8192 -bs 16 -a 4 -wa -wb trazas/spice100.trace
	
	#gdb a.out