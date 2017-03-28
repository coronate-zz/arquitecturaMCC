.text
daddi R1,R0,32

LOOP: lw r10,0(r1); Leer un elemento de un vector
    daddi r10,r10,4 ; Sumar 4 al elemento
    sw r10,0(r1); Escribir el nuevo valor
    daddi r1,r1,-4 ; Actualizar la var. índice
    bne r1,r0,LOOP ; Fin de vector?
