GLOBAL:

.stack 100h
.data
my_variable db 'hello$'
.code          
;INITIALIZE DATA SEGMENT.
  mov  ax,@data
  mov  ds,ax

  call my_procedure

;FINISH.  
  mov  ax,4c00h
  int  21h           

proc my_procedure
  mov  dx,offset my_variable
  mov  ah,9
  int  21h
my_label:
  ret
endp