
/*
 * main.c
 */


#include <stdlib.h>
#include <stdio.h>
#include "cache.h" 
#include "main.h"
#include <string.h>

static FILE *traceFile;


int main(argc, argv)
  int argc;
  char **argv;
{
  parse_args(argc, argv);
  /*Cambia los Valores predeterminados del cache y lee el archivo en trazas/cc.trace 
  * Este archivo es el tracefile que será leido por el play_tarce*/
  
  printf("\nInicializando Cache...\n\n" );
  init_cache(); //Funcion creada por nosotros

  play_trace(traceFile);
  /*Es una función que lee linea por linea las instrucciones dentro del tarce file
  * dependiendo del tipo de instruccion que indique cada linea (&access_type, &addr)
  * ejecutaremos alguna función del cache. 
  */
  print_stats(); //Función creada por nosotros ¿¿dentro del Cache??
}






/************************************************************/
void parse_args(argc, argv) // Configure the cache using total cache size, block size, Block size...
  int argc;
  char **argv; //char** is a pointer to a char*. Making it a pointer to a pointer to a char
  /*
  * Argv and Argc: Son atributos del main y hacen 
  * referencia a los comanodos ingresados 
  * en la terminal argc indica el numero de elemntos
  * y argv es un array con los elementos ingresados
  */
{
  int arg_index, i, value;

  if (argc < 2) {
    printf("usage:  sim <options> <trace file>\n");
    exit(-1);
  }

  /* parse the command line arguments */
  for (i = 0; i < argc; i++)
    if (!strcmp(argv[i], "-h")) {  //cuando ejecutamos gcc main.c cache.c -h Se muestra la siguiente lista
      printf("\t-h:  \t\tthis message\n\n");
      printf("\t-bs <bs>: \tset cache block size to <bs>\n");
      printf("\t-us <us>: \tset unified cache size to <us>\n");
      printf("\t-is <is>: \tset instruction cache size to <is>\n");
      printf("\t-ds <ds>: \tset data cache size to <ds>\n");
      printf("\t-a <a>: \tset cache associativity to <a>\n");
      printf("\t-wb: \t\tset write policy to write back\n");
      printf("\t-wt: \t\tset write policy to write through\n"); 
      printf("\t-wa: \t\tset allocation policy to write allocate\n");
      printf("\t-nw: \t\tset allocation policy to no write allocate\n");
      exit(0);
    }
    /* Notemos que en todos los casos se utiliza la función set_cache_param 
    *(dentro de cache.c) que que toma el tipo de parametro que queremos
    * alterar y el valor que tomará */
    
  arg_index = 1;
  while (arg_index != argc - 1) {

    /* Dependiendo de la manera en la que ejecutemos a.out podemos cambiar caules son los
    * valores de nuestro cache:
    * EJEMPLO:  ./a.out -bs 500 -us 10 -is 30 -ds 50000  con este comando estamos creando un cache
    * con un tamaño de bloque de 500, unified size de 10, instruction cache size de 30 y una capacidad
    * de 50,000. Es necesario conocer las unidades que maneja cada flag  */

    if (!strcmp(argv[arg_index], "-bs")) {  //cuando ejecutamos ./a.out -bs X cambiamos el valor de BLOCK SIZE a x
      value = atoi(argv[arg_index+1]);
      set_cache_param(CACHE_PARAM_BLOCK_SIZE, value);
      arg_index += 2;
      continue;
    }

    if (!strcmp(argv[arg_index], "-us")) { //cuando ejecutamos ./a.out -us X cambiamos el valor de PARAM USIZE a X
      value = atoi(argv[arg_index+1]);
      set_cache_param(CACHE_PARAM_USIZE, value);
      arg_index += 2;
      continue;
    }

    if (!strcmp(argv[arg_index], "-is")) { //cuando ejecutamos ./a.out -is X cambiamos el valor de PARAM ISIZE a X
      value = atoi(argv[arg_index+1]);
      set_cache_param(CACHE_PARAM_ISIZE, value);
      arg_index += 2;
      continue;
    }

    if (!strcmp(argv[arg_index], "-ds")) {
      value = atoi(argv[arg_index+1]);
      set_cache_param(CACHE_PARAM_DSIZE, value);
      arg_index += 2;
      continue;
    }

    if (!strcmp(argv[arg_index], "-a")) {
      value = atoi(argv[arg_index+1]);
      set_cache_param(CACHE_PARAM_ASSOC, value);
      arg_index += 2;
      continue;
    }

    if (!strcmp(argv[arg_index], "-wb")) {
      set_cache_param(CACHE_PARAM_WRITEBACK, value);
      arg_index += 1;
      continue;
    }

    if (!strcmp(argv[arg_index], "-wt")) {
      set_cache_param(CACHE_PARAM_WRITETHROUGH, value);
      arg_index += 1;
      continue;
    }

    if (!strcmp(argv[arg_index], "-wa")) {
      set_cache_param(CACHE_PARAM_WRITEALLOC, value);
      arg_index += 1;
      continue;
    }

    if (!strcmp(argv[arg_index], "-nw")) {
      set_cache_param(CACHE_PARAM_NOWRITEALLOC, value);
      arg_index += 1;
      continue;
    }

    printf("error:  unrecognized flag %s\n", argv[arg_index]);
    exit(-1);

  }

  dump_settings(); //print cache configuration 


  traceFile = fopen(argv[arg_index], "r");
  /*Cuando ejecutamos ./a.out trazas/cc.trace el archivo en trazas será leido. 
  *Este debe ser el ultimo en la linea de comandos*/

  return;
}
/************************************************************/

/************************************************************/
void play_trace(inFile)
  FILE *inFile;
  /*Lo que hace play tarce es leer a lo largo del tarce file linea por linea.
  * Para cada linea idenitificara cual es el componente tipo de instruccion, 
  * cual es el adress y cuando termina el trace file*/
{
  unsigned addr, data, access_type;
  int num_inst;

  num_inst = 0;
  while(read_trace_element(inFile, &access_type, &addr)) { 
  /*read_trace_element: Lee del file y llena access_type con 
  * el primer numer del trace {0,1,2}, addr lo llena con el adress
  * de la instrucción. Finalmente read_trace_element es una funcion
  * que regresa 0 si llegamos al final del trace o 1 si aún hay mas
  * datos en el file.
  *
  * --Yo pensaría que dentro del switch, cuando se encuentre un TRACE_LOAD
  * se utilice el cache para operar con base a dicha función
  * cuando se encuentre TARCE_DATA_STORE  entonces ejecutamos
  * la segunda función del cache y asi... --
  */

    switch (access_type) {
    case TRACE_DATA_LOAD:    //==0
    case TRACE_DATA_STORE:   //==1
    case TRACE_INST_LOAD:    //==2
      perform_access(addr, access_type); //Simulate a dingle memory reference
      break;

    default:
      printf("skipping access, unknown type(%d)\n", access_type);
    }

    num_inst++;
    if (!(num_inst % PRINT_INTERVAL)) //PRINT_INTERVAL=const 100,000
      printf("processed %d references\n", num_inst);
  }

  flush(); //Función creada por nosotros, borra los datos del cache
}
/************************************************************/

/************************************************************/
int read_trace_element(inFile, access_type, addr)
  FILE *inFile;
  unsigned *access_type, *addr;
{
  int result;
  char c;

  result = fscanf(inFile, "%u %x%c", access_type, addr, &c); 
  /* Escaneamos nuestras instrucciones 2 408ed4 el 2 será asignado a access_type,
  * 408ed4 será asignado a la dirección y "\n" sera asignado a c*/

  while (c != '\n') 
   {
    result = fscanf(inFile, "%c", &c);
    if (result == EOF) 
      break;
    }
  if (result != EOF)
    return(1);
  else
    return(0);
}
/************************************************************/
