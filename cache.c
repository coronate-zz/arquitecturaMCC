/*
 * cache.c
 */


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "cache.h"
#include "main.h"

/* cache configuration parameters */
static int cache_split = 0;
static int cache_usize = DEFAULT_CACHE_SIZE;
static int cache_isize = DEFAULT_CACHE_SIZE; 
static int cache_dsize = DEFAULT_CACHE_SIZE;
static int cache_block_size = DEFAULT_CACHE_BLOCK_SIZE;
static int words_per_block = DEFAULT_CACHE_BLOCK_SIZE / WORD_SIZE;
static int cache_assoc = DEFAULT_CACHE_ASSOC;
static int cache_writeback = DEFAULT_CACHE_WRITEBACK;
static int cache_writealloc = DEFAULT_CACHE_WRITEALLOC;

/* cache model data structures */
static Pcache icache;
static Pcache dcache;
static cache c1;
static cache c2;
static cache_stat cache_stat_inst;
static cache_stat cache_stat_data;





void imprimirBinario(int numero)
{

  for(int i=31 ;i>=0; i-- )
  {
    if(numero&(1<<i))
    {
      printf("1");

    }
    else
    {
      printf("0");
    }
  }
  //printf("\n");
}

/************************************************************/
void set_cache_param(param, value)
  int param;
  int value;
{

  switch (param) {
  case CACHE_PARAM_BLOCK_SIZE:
    cache_block_size = value;
    words_per_block = value / WORD_SIZE;
    break;
  case CACHE_PARAM_USIZE: //Unified 
    cache_split = FALSE;
    cache_usize = value;
    break;
  case CACHE_PARAM_ISIZE: //Instruction size
    cache_split = TRUE;
    cache_isize = value;
    break;
  case CACHE_PARAM_DSIZE: //Data Cache Size
    cache_split = TRUE;
    cache_dsize = value;
    break;
  case CACHE_PARAM_ASSOC: //Associativity
    cache_assoc = value;
    break;
  case CACHE_PARAM_WRITEBACK: 
    cache_writeback = TRUE;
    break;
  case CACHE_PARAM_WRITETHROUGH:
    cache_writeback = FALSE;
    break;
  case CACHE_PARAM_WRITEALLOC: 
    cache_writealloc = TRUE;
    break;
  case CACHE_PARAM_NOWRITEALLOC:
    cache_writealloc = FALSE;
    break;
  default:
    printf("error set_cache_param: bad parameter value\n");
    exit(-1);
  }

}

/************************************************************/
void init_contents(Pcache_line *head, Pcache_line *tail, int *set_contents, int n_sets){
  int i;
  for(i = 0; i < n_sets; i++){
    set_contents[i] = 0;
    head[i] = (Pcache_line) NULL;
    tail[i] = (Pcache_line) NULL;
  } 
}


//Initialize unified size cache
void init_cache_helper(cache *c, int size){
    unsigned bits_set, bits_offset;
    c->size = size;
    c->associativity = cache_assoc;
    c->n_sets = size/(cache_block_size * c->associativity);
    bits_offset = LOG2(cache_block_size);
    bits_set = LOG2(c->n_sets);
    c->index_mask = (1<< (bits_set+bits_offset)) - 1;
    c->index_mask_offset = bits_offset;

    //debemos calcular el 
    c->LRU_head = (Pcache_line*)malloc(sizeof(Pcache_line)*c->n_sets);
    c->LRU_tail = (Pcache_line*)malloc(sizeof(Pcache_line)*c->n_sets);
    c->set_contents = (int*)malloc(sizeof(int)*c->n_sets);
    c->contents = 0;
    init_contents(c->LRU_head, c->LRU_tail, c->set_contents,c->n_sets);
}





/************************************************************/
void init_cache() 
{

  init_cache_helper(&c1, cache_usize);
  printf("\nNUMERO SETS : \n%d\n", c1.n_sets );
  imprimirCache(c1);

  //char str[25];
  //getBin(16777215 , str);
  //printf("\nSTR:  %s\n", str);

}
/************************************************************/
void imprimirCache(cache miCache)
{

  printf("*** Datos en Cache ***" );
  printf("\n\t**Size:         %u\n",  miCache.size);
  printf("\t**Associativity : %u\n",  miCache.associativity);
  printf("\t**Num Sets :      %u\n",  miCache.n_sets);


  printf("Index Mask :  %d  || ",   miCache.index_mask);  
  imprimirBinario(miCache.index_mask); 

  printf("\nOffset Mask : %d      || ",  miCache.index_mask_offset);   
  imprimirBinario(miCache.index_mask_offset); 
  printf("\tContents: %u\n\n\n",  miCache.contents); 

}


/************************************************************/
void perform_access(addr, access_type)
  unsigned addr, access_type;
{

  //Imprimiendo las direcciones
  printf("\n\n------Realizando acceso a  Cache-----" );
  printf("\nAdress Hexadecimal:   %x  ",  addr);

  //
  int offset_size=LOG2(cache_block_size);

  int  ones_mask=0xffffff;
  int  numero = (int)ones_mask;
  printf("\nMascara de unos:      ");
  imprimirBinario(numero);


  int intAddr =(int)addr;
  printf("\nAdress Decimal:       %d  ",  intAddr);
  printf("\nAdress Binario:       ");
  imprimirBinario(intAddr);

  



  /* handle an access to the cache */

}
/************************************************************/

/************************************************************/
void flush()
{

  /* flush the cache */

}
/************************************************************/

/************************************************************/
void delete(head, tail, item)
  Pcache_line *head, *tail;
  Pcache_line item; //Esto esta raro no es un apuntador y sin embargo se usa item->LRU_prev
{
  if (item->LRU_prev) { 
    item->LRU_prev->LRU_next = item->LRU_next;
  } else {
    /* item at head */
    *head = item->LRU_next;
  }

  if (item->LRU_next) {
    item->LRU_next->LRU_prev = item->LRU_prev;
  } else {
    /* item at tail */
    *tail = item->LRU_prev;
  }
}
/************************************************************/

/************************************************************/
/* inserts at the head of the list */
void insert(head, tail, item)
  Pcache_line *head, *tail;
  Pcache_line item;
{
  item->LRU_next = *head;
  item->LRU_prev = (Pcache_line)NULL;

  if (item->LRU_next)
    item->LRU_next->LRU_prev = item;
  else
    *tail = item;

  *head = item;
}
/************************************************************/

/************************************************************/
void dump_settings()
{
  printf("*** CACHE SETTINGS ***\n");
  if (cache_split) {
    printf("  Split I- D-cache\n");
    printf("  I-cache size: \t%d\n", cache_isize);
    printf("  D-cache size: \t%d\n", cache_dsize);
  } else {
    printf("  Unified I- D-cache\n");
    printf("  Size: \t%d\n", cache_usize);
  }
  printf("  Associativity: \t%d\n", cache_assoc);
  printf("  Block size: \t%d\n", cache_block_size);
  printf("  Write policy: \t%s\n", 
	 cache_writeback ? "WRITE BACK" : "WRITE THROUGH");
  printf("  Allocation policy: \t%s\n",
	 cache_writealloc ? "WRITE ALLOCATE" : "WRITE NO ALLOCATE");
}
/************************************************************/

/************************************************************/
void print_stats()
{
  printf("\n*** CACHE STATISTICS ***\n");

  printf(" INSTRUCTIONS\n");
  printf("  accesses:  %d\n", cache_stat_inst.accesses);
  printf("  misses:    %d\n", cache_stat_inst.misses);

  if (!cache_stat_inst.accesses)
    printf("  miss rate: 0 (0)\n"); 
  else
    printf("  miss rate: %2.4f (hit rate %2.4f)\n", 
	 (float)cache_stat_inst.misses / (float)cache_stat_inst.accesses,
	 1.0 - (float)cache_stat_inst.misses / (float)cache_stat_inst.accesses);
  printf("  replace:   %d\n", cache_stat_inst.replacements);




  printf(" DATA\n");
  printf("  accesses:  %d\n", cache_stat_data.accesses);
  printf("  misses:    %d\n", cache_stat_data.misses);
  if (!cache_stat_data.accesses)
    printf("  miss rate: 0 (0)\n"); 
  else
    printf("  miss rate: %2.4f (hit rate %2.4f)\n", 
	 (float)cache_stat_data.misses / (float)cache_stat_data.accesses,
	 1.0 - (float)cache_stat_data.misses / (float)cache_stat_data.accesses);
  printf("  replace:   %d\n", cache_stat_data.replacements);




  printf(" TRAFFIC (in words)\n");
  printf("  demand fetch:  %d\n", cache_stat_inst.demand_fetches + 
	 cache_stat_data.demand_fetches);
  printf("  copies back:   %d\n", cache_stat_inst.copies_back +
	 cache_stat_data.copies_back);
}
/************************************************************/
