; Arquitectura de computadoras
; Programa de demostración. 
.data
i:	.word32 0
j:	.word32 0
.text
	daddi R2,R0,0;
	daddi r3, R0, 0;
	daddi r5,R0,10 ;
WHIL:	slt R6, R2, R5
	beqz R6, ENDW
	daddi r3, r3, 5
	sw R3, j(r0)
	daddi r2,r2,1
	sw	r2,i(r0)
	j WHIL
ENDW:	nop
	halt
