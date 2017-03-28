.data
i:  .word32 0 ;power
off:  .word32 4 ;offset
p: .word32 0 ;pointer
A:  .space 40

.text
  daddi R4, R0, 1 ; register that saves result
  ;daddi R3, R0, 2 ;Numero dos 
  ld R2, i(R0) ; load i
  ld R1, p(R0) ; load pointer
  daddi R5, R0, 10 ; load end loop

WHIL: slt R6, R2, R5
  beqz R6, ENDW
  sll R4, R4, 1
  sw R4, A(R1) ;Save on the r1 space of A
  daddi R1, R1, 4; move pointer a 4byte (32bit) offset
  daddi R2,R2,1 ; move counter
  j WHIL
ENDW: nop
  halt