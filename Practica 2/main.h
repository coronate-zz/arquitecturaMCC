
/*
 * main.h
 */

/*Estos son solo algunos valores constantes que corresponden al primer numero
* de cada instrucción de los trace files
*/
#define TRACE_DATA_LOAD 0
#define TRACE_DATA_STORE 1
#define TRACE_INST_LOAD 2

#define PRINT_INTERVAL 100000

void parse_args();
void play_trace();
int  read_trace_element();

