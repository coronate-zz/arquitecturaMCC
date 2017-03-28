.text
daddi R1,R0,32

LOOP: lw r10,0(r1); Leer elemento vector
    daddi r10,r10,4 ; Sumar 4 al elemento
    sw r10,0(r1); Escribir nuevo valor

    lb r11,-4(r1); 2a copia
    daddi r11,r11,4 ;
    sw r11,-4(r1); 

    lb r12,-8(r1); 3a copia
    daddi r12,r12,4 ; 
    sw r12,-8(r1);
     
    lb r13,-12(r1); 4a copia
    daddi r13,r13,4 ;
    sw r13,-12(r1); 

    daddi r1,r1,-16 ; Actualizar índice
    bne r1,r0,LOOP ; Fin de vector?
