
.data
i:	.word32 0
j:	.word32 0
A:  .space 12

.text
	daddi r3, R0, 2 ;Multiplicador
	daddi r2, R0, i ;i
	daddi r5, R0, 10 ;J

	daddi r15, R0, 1; Guardar potencia anterior
	daddi r10, r0, 0 ;Pointer sobre A

	daddi r5,R0, 10;


WHIL:	slt R6, R2, R5
	beqz R6, ENDW
	dmul r11, r15, r3; Guarda en A(r10) R15*2
	sW r11, A(r10);
	daddi r10, r10, 4;  Mover el apuntador 


	daddi r2,r2,1
	sw	r2,i(r0)




	j WHIL
ENDW:	nop







