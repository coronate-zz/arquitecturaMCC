; Arquitectura de computadoras
; Programa de demostración. 
.data
i:	.word32 0
j:	.word32 0
uno: .word32 1
A:  .space 12
.text
	

	add.d f0,f2, uno(r0);


	daddi r3, R0, 0;
	daddi r5,R0, 10;
	daddi r15, R0, 0;




WHIL:	slt R6, R2, R5
	beqz R6, ENDW
	
	mul.d f2, f4, f6
	sw R3, j(r0)

	daddi r2,r2,1
	sw	r2,i(r0)

	sW r3, A(r15);
	daddi r15, r15, 4;

	j WHIL
ENDW:	nop







