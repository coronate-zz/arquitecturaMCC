
gcc main.c cache.c
./a.out -h
./a.out trazas/cc.trace 
#Debemos escribir todas las configuraciones que nos pide la tabla 1 del PDF (pagina 7)
#Esta es la instruccion de la primera fila de la tabla 1
#
# ./a.out   -ds 8     -bs 16        -a 1            -wb            -wa         trazas/cc.trace

#        CacheSize   BlockSize   Associativity    Write-Back   WriteAllocate
#
./a.out -us 8192 -bs 16 -a 1 -wa -wb trazas/cc.trace