/*
 * cache.c
 */


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include "cache.h"
#include "main.h"

/* cache configuration parameters */
static int cache_split = 0;
int numeroTraze =0;
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

void dataLoadHit(Pcache_line compare, int addrIndex)
{
  printf("\nData LaodHit");
   delete(&c1.LRU_head[addrIndex], &c1.LRU_tail[addrIndex], compare);
   insert(&c1.LRU_head[addrIndex], &c1.LRU_tail[addrIndex], compare);

}

void dataStoreHit(Pcache_line compare, int addrIndex)
{
  if( cache_writeback== TRUE )
  {
    printf("\nData Store Hit /  WB");
   compare->dirty=1;
   delete(&c1.LRU_head[addrIndex], &c1.LRU_tail[addrIndex], compare);
   insert(&c1.LRU_head[addrIndex], &c1.LRU_tail[addrIndex], compare);
 }
 else
 {
    printf("\nData Store Hit /  WT");
   delete(&c1.LRU_head[addrIndex], &c1.LRU_tail[addrIndex], compare);
   insert(&c1.LRU_head[addrIndex], &c1.LRU_tail[addrIndex], compare);
 }



}
void dataLoadMiss(int addrIndex, int addrTag,  Pcache_line item)
{
  if(cache_writeback==FALSE)
  {
     printf("\nData Load Miss WT" );

    if(c1.set_contents[addrIndex] < c1.associativity)//Aun hay espacio
    {

        item->tag=addrTag;
        insert(&c1.LRU_head[addrIndex], &c1.LRU_tail[addrIndex], item);
        c1.set_contents[addrIndex]++;
        //printf("\n\t\t***Cabeza: %d  Next: %d   \n" , c1.LRU_head[addrIndex]->tag, c1.LRU_head[addrIndex]->LRU_next->tag);
        //printf("\n++++CONTENTS:  %d\n", c1.set_contents[addrIndex]);
        //printf("\nASSOC:  %d\n", c1.associativity );
    }
    else //Es necesario Borrar
    {
        printf("\n\t\t***Borrando Datos..." );
       // printf("\n\t\tCache antes de Cambios:");
       //imprimirCacheCompleto(&c1);
        item->tag=addrTag;
        insert(&c1.LRU_head[addrIndex], &c1.LRU_tail[addrIndex], item);
       // printf("\n\t\tCache despues de Cambios:");
       //imprimirCacheCompleto(&c1);
    }
  }
  else
  {
       printf("\nData Load Miss WB " );
    if(c1.set_contents[addrIndex] < c1.associativity)//Aun hay espacio
    {
        printf("\nAun hay espacio" );
        item->tag=addrTag;
        insert(&c1.LRU_head[addrIndex], &c1.LRU_tail[addrIndex], item);
        c1.set_contents[addrIndex]++;
    }
    else //Es necesario Borrar
    {
        printf("\n\t\t***Borrando Datos..." );

        item->tag=addrTag;
        if(c1.LRU_tail[addrIndex]->dirty==1)
        {
            //EN este caso tenemso el bit drty, accedemos a memoria
          printf("\n\t\t***Eliminando dato Dirty" );

        }
        else
        {
            //EN este caso tenemso el bit drty, accedemos a memoria
          printf("\n\t\t***Eliminando dato Limpio" );



        }
        delete(&c1.LRU_head[addrIndex], &c1.LRU_tail[addrIndex], c1.LRU_tail[addrIndex]);
        insert(&c1.LRU_head[addrIndex], &c1.LRU_tail[addrIndex], item);

    }

  }
                
}

void dataStoreMiss(int addrIndex, int addrTag, Pcache_line item)
{
  if(cache_writeback==FALSE && cache_writealloc==FALSE)
  {
    /*Dado que hay un miss y la politica es write through 
    * con write allocate entonce solo escribimos en memoria
    * no tenemos que modificar el cache.
    *           ESTADISTICAS.
    */
      printf("\nData Store Miss WT / NWA" );
      printf("\n\t    ~~~~~  No se Modificara cache   ~~~~   ");


  }
  else if(cache_writeback==FALSE && cache_writealloc==TRUE)
  {
      /* Cuando el dato no se encunetre en cache deberemos regresarlo 
      * despues de modificarlo en memoria.
    */
    printf("\nData Store Miss WT / WA" );
      if(c1.set_contents[addrIndex] < c1.associativity)//Aun hay espacio
    {
        printf("\n\t\t**Aun hay espacio" );
        item->tag=addrTag;
        insert(&c1.LRU_head[addrIndex], &c1.LRU_tail[addrIndex], item);
        c1.set_contents[addrIndex]++;
    }
    else //Es necesario Borrar
    {
        printf("\n\t\t**Borrando Datos..." );
        item->tag=addrTag;
        delete(&c1.LRU_head[addrIndex], &c1.LRU_tail[addrIndex], c1.LRU_tail[addrIndex]);
        insert(&c1.LRU_head[addrIndex], &c1.LRU_tail[addrIndex], item);
      }

  }
  else if(cache_writeback==TRUE && cache_writealloc==FALSE)
  {
    
  /*Como el dato solo se modifica en Memoria, no debemos cambiar
  * nada en cache.
  */
     printf("\nData Store Miss WB / NWA" );
      printf("\n\t    ~~~~~  No se Modificara cache   ~~~~   ");

  }
  else if(cache_writeback==TRUE && cache_writealloc==TRUE)
  {
    /* Modificamos el dato en memoria y luego lo regresamos a cache.
    *     -Si hay espacio en el set entonces ingresamos el dato.
    *     -Si el set esta lleno deberemos vaciar eliminar un dato.
                  +El dato puede estar limpio, en cuyo caso solo 
                    lo eliminamos
                  +Si el dato esta dirty entonces deberemos hacer otro
                   acceso a memoria y generar un caso particular para
                   esta simulación.
  */
    printf("\nData Store Miss WB / WA" );
    if(c1.set_contents[addrIndex] < c1.associativity)//Aun hay espacio
    {
        printf("\n\t\t***Aun hay espacio" );
        item->tag=addrTag;
        insert(&c1.LRU_head[addrIndex], &c1.LRU_tail[addrIndex], item);
        c1.set_contents[addrIndex]++;
    }
    else //Es necesario Borrar
    {
        printf("\n\t\t***Borrando Datos..." );

        item->tag=addrTag;
        if(c1.LRU_tail[addrIndex]->dirty==1)
        {
            //EN este caso tenemso el bit drty, accedemos a memoria
          printf("\n\t\t***Eliminando dato Dirty" );

        }
        else
        {
            //EN este caso tenemso el bit drty, accedemos a memoria
          printf("\n\t\t***Eliminando dato Limpio" );



        }
        delete(&c1.LRU_head[addrIndex], &c1.LRU_tail[addrIndex], c1.LRU_tail[addrIndex]);
        insert(&c1.LRU_head[addrIndex], &c1.LRU_tail[addrIndex], item);

    }


  
  }
  else
  {
    printf("\nError Politica desconocida" );
  }


}


unsigned mascara(int n1, int n2)
{
  /* Esta funcion caclula la mascara para el numero n1
  * y la desplaza n2 digitos a la izquierda:
  Ejemplo Sea n1=5, n2=3 la funcion genera 
          * (2^5-1)=0000011111
          * desplaza 3 = 0011111000 
  */
  return(((int)pow(2,n1)-1)<<n2);

}

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

void imprimirCacheCompleto(cache *c)
{
  printf("\n-----------------------------------------------------" );
  printf("\n\tImprimeindo Cache Completo:");
  Pcache_line item =malloc(sizeof(cache_line));  


  for( int cont=0; cont <= c1.n_sets; cont ++   )
  {
    if(c->LRU_head[cont]==NULL)
    {
      //no hay datos en ese index
    }
    else
    {
      item= c->LRU_head[cont];

      printf("\n\n\t Index: %d ",cont);
      printf("\n\t Tags: " );



      for(int i=1; i<=c->associativity; i++)
      {
        if(item->dirty==1)
        {
          printf("||  %d*   ", item->tag);

        }
        else
        { 
          printf("||   %d    ", item->tag);
        }

        if(item->LRU_next== NULL)
        {
          for(int j=0; j<c->associativity-i; j++)
          {
          printf("||   NULL    " );
          }
          break;
        }
        
       // printf("\n Adess de next:  %x" , item->LRU_next) );
        item=item->LRU_next; //No se puede hacer asignacion completa de objetos

      }
    }

    //end for cont
  }
    printf("\n-------------------------------------------------------" );

}
void imprimirDirecciones(unsigned addr )
{
  printf("\nAdress Hexadecimal:   %x  ",  addr);
  
  int intAddr =(int)addr;
  printf("\nAdress Decimal:       %d  ",  intAddr);
  printf("\nAdress Binario:       ");
  imprimirBinario(intAddr);

  int addrIndex, addrTag, addrOffset;
  addrTag=   (c1.tag_mask&addr)>>c1.tag_mask_offset;
  addrIndex= (c1.index_mask&addr)>>c1.index_mask_offset;

  printf("\n\nTagBits: %d     IndexBits: %d   OffsetBits: %d",c1.tag_mask_offset, c1.index_mask_offset, c1.offset_mask_offset );
  printf("\nImprimiendo Tag:    %d || ", addrTag);
  imprimirBinario(addrTag);


  printf("\nImprimiendo Index:  %d || ", addrIndex );
  imprimirBinario(addrIndex);


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
    bits_offset = ceil(LOG2(cache_block_size));
    bits_set =ceil(LOG2(c->n_sets));


    c->index_mask_offset = bits_offset;
    c->index_mask = mascara(bits_set, c->index_mask_offset);

    c->offset_mask = mascara(bits_offset,0);
    c->offset_mask_offset= bits_offset;
    c->tag_mask_offset = bits_offset+bits_set;
    c->tag_mask = mascara((32-c->tag_mask_offset), c->tag_mask_offset);

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
  imprimirCache(c1);

  //char str[25];
  //getBin(16777215 , str);
  //printf("\nSTR:  %s\n", str);

}
/************************************************************/
void imprimirCache(cache miCache)
{

  printf("\n\n*** Datos en Cache ***" );
  printf("\n\t**Size:         %u\n",  miCache.size);
  printf("\t**Associativity : %u\n",  miCache.associativity);
  printf("\t**Num Sets :      %u\n",  miCache.n_sets);
  printf("\t**Contents: %u",  miCache.contents); 

  printf("\nTag Mask : %d      || ",  miCache.tag_mask);   
  imprimirBinario(miCache.tag_mask); 

  printf("\nIndex Mask :  %d  || ",   miCache.index_mask);  
  imprimirBinario(miCache.index_mask); 

  printf("\nOffset Mask : %d      || ",  miCache.offset_mask);   
  imprimirBinario(miCache.offset_mask); 

}


/************************************************************/
void perform_access(addr, access_type)
  unsigned addr, access_type;
{
  numeroTraze++;
  printf("\nNumero de Instriccion: %d", numeroTraze );
  int addrIndex, addrTag, addrOffset;
  addrTag=   (c1.tag_mask&addr)>>c1.tag_mask_offset;
  addrIndex= (c1.index_mask&addr)>>c1.index_mask_offset;
  Pcache_line item;
  item = malloc(sizeof(cache_line));

  //Imprimiendo las direcciones
  printf("\n\n---------------Realizando acceso a  Cache------------\n" );
  imprimirDirecciones(addr);
   
  switch(access_type){
        case TRACE_INST_LOAD:
        printf("\n\nEjecutando Lectura Instruccion" );
            cache_stat_inst.accesses++;
            if(c1.LRU_head[addrIndex]==NULL)
            {  // Compulsory miss
                cache_stat_inst.misses++;
                c1.LRU_head[addrIndex]=malloc(sizeof(cache_line));  // Deberias validar que hay memoria!!
                
                item->tag=addrTag;
                c1.LRU_head[addrIndex]=item;
                c1.LRU_tail[addrIndex]=item;

                printf("\n\tPrimer dato ingresado");
                c1.LRU_head[addrIndex]->dirty=0;
                cache_stat_inst.demand_fetches+=4;
                c1.set_contents[addrIndex]++;
            }
            else // Hay infromacion, queremos ver si el dato se encuentra en cache
            {
                Pcache_line compare= malloc(sizeof(cache_line));  
                compare= c1.LRU_head[addrIndex]; //apuntador a cache_line
                bool flagEncontrado=FALSE;
                bool flagNext=TRUE ;
                int cont=1;
                while( cont<=c1.associativity && flagNext && !flagEncontrado)
                {
                    if(compare->tag==addrTag) //encontramos en cache la isntruccion
                    {
                        flagEncontrado=TRUE;
                    }
                    else //no encontramos el tag en una de las paginas
                    { 
                        if(compare->LRU_next==NULL)
                        {
                          printf("\n\t\t**No hay next" );
                          flagNext=FALSE;
                          break;

                        }
                        else
                        {
                          printf("\n\t\t**Buscando en next");
                          cont ++;
                          compare= compare->LRU_next;
                          
                        }
                    }
                }
                if(flagEncontrado) //el dato esta en memoria
                {
                  //HIT 
                    printf("\n\t :: El dato ESTA en memoria :: \n" );
                    delete(&c1.LRU_head[addrIndex], &c1.LRU_tail[addrIndex], compare);
                    insert(&c1.LRU_head[addrIndex], &c1.LRU_tail[addrIndex], compare);
                }
                else
                {
                    printf("\n\t:: El dato NO esta en memoria :: \n" );

                    if(c1.set_contents[addrIndex] < c1.associativity)//Aun hay espacio
                    {

                        item->tag=addrTag;
                        insert(&c1.LRU_head[addrIndex], &c1.LRU_tail[addrIndex], item);
                        c1.set_contents[addrIndex]++;

                    }
                    else //Es necesario Borrar
                    {
                        printf("\n\t\t***Borrando Datos..." );
                        item->tag=addrTag;
                        insert(&c1.LRU_head[addrIndex], &c1.LRU_tail[addrIndex], item);
                    }
                }


            }
            //imprimirCacheCompleto(&c1);

            break;

        case TRACE_DATA_LOAD:

            cache_stat_data.accesses++;
            printf("\n\nEjecutando Data Load" );
                       cache_stat_inst.accesses++;
            if(c1.LRU_head[addrIndex]==NULL)
            {  // Compulsory miss
                cache_stat_inst.misses++;
                c1.LRU_head[addrIndex]=malloc(sizeof(cache_line));  // Deberias validar que hay memoria!!
                
                item->tag=addrTag;
                c1.LRU_head[addrIndex]=item;
                c1.LRU_tail[addrIndex]=item;

                printf("\n\tPrimer dato ingresado");
                c1.LRU_head[addrIndex]->dirty=0;
                cache_stat_inst.demand_fetches+=4;
                c1.set_contents[addrIndex]++;
            }
            else // Hay infromacion, queremos ver si el dato se encuentra en cache
            {
                Pcache_line compare= malloc(sizeof(cache_line));  
                compare= c1.LRU_head[addrIndex]; //apuntador a cache_line
                bool flagEncontrado=FALSE;
                bool flagNext=TRUE ;
                int cont=1;
                while( cont<=c1.associativity && flagNext && !flagEncontrado)
                {
                  //printf("\nTAG: %d",compare->tag );
                  //printf("\nAddress: %d\n", addrTag );
                    if(compare->tag==addrTag) //encontramos en cache la isntruccion
                    {
                        flagEncontrado=TRUE;
                    }
                    else //no encontramos el tag en una de las paginas
                    { 
                        if(compare->LRU_next==NULL)
                        {
                          printf("\n\t\t**No hay next" );
                          flagNext=FALSE;
                          break;

                        }
                        else
                        {
                          printf("\n\t\t**Buscando en next");
                          cont ++;
                          compare= compare->LRU_next;
                          
                        }
                    }
                }
                if(flagEncontrado) //el dato esta en memoria
                {
                  //HIT 
                    printf("\n\t :: El dato ESTA en memoria :: \n" );
                    dataLoadHit(compare, addrIndex);
                }
                else
                {
                    printf("\n\t:: El dato NO esta en memoria :: \n" );
                    dataLoadMiss(addrIndex, addrTag, item);
                }



            }




            break;
        case TRACE_DATA_STORE:
            cache_stat_data.accesses++;
            printf("\n\nEjecutando Data Store" );
                       cache_stat_inst.accesses++;
            if(c1.LRU_head[addrIndex]==NULL)
            {  // Compulsory miss
                cache_stat_inst.misses++;
                c1.LRU_head[addrIndex]=malloc(sizeof(cache_line));  // Deberias validar que hay memoria!!
                
                item->tag=addrTag;
                c1.LRU_head[addrIndex]=item;
                c1.LRU_tail[addrIndex]=item;

                printf("\n\tPrimer dato ingresado");
                c1.LRU_head[addrIndex]->dirty=0;
                cache_stat_inst.demand_fetches+=4;
                c1.set_contents[addrIndex]++;
            }
            else // Hay infromacion, queremos ver si el dato se encuentra en cache
            {
                Pcache_line compare= malloc(sizeof(cache_line));  
                compare= c1.LRU_head[addrIndex]; //apuntador a cache_line
                bool flagEncontrado=FALSE;
                bool flagNext=TRUE ;
                int cont=1;
                while( cont<=c1.associativity && flagNext && !flagEncontrado)
                {
                  //printf("\nTAG: %d",compare->tag );
                  //printf("\nAddress: %d\n", addrTag );
                    if(compare->tag==addrTag) //encontramos en cache la isntruccion
                    {
                        flagEncontrado=TRUE;
                    }
                    else //no encontramos el tag en una de las paginas
                    { 
                        if(compare->LRU_next==NULL)
                        {
                          printf("\n\t\t**No hay next" );
                          flagNext=FALSE;
                          break;

                        }
                        else
                        {
                          printf("\n\t\t**Buscando en next");
                          cont ++;
                          compare= compare->LRU_next;
                          
                        }
                    }
                }
                if(flagEncontrado) //el dato esta en memoria
                {
                  //HIT 
                    printf("\n\t :: El dato ESTA en memoria :: \n" );
                    dataStoreHit(compare, addrIndex);

                    //Que estadisticas aumentan cuando encontramos datos?
                }
                else
                {
                    printf("\n\t:: El dato NO esta en memoria :: \n" );
                    dataStoreMiss(addrIndex, addrTag, item);
                }


            }

            break;
    }


  imprimirCacheCompleto(&c1);
  printf("\n\n\n");

  /* handle an access to the cache */

}
/************************************************************/
void perform_readData(unsigned addrIndex, unsigned addrTag )
{
  /* Los apuntadores funcionan de la siguiente manera:
  * - c1.LRU_head[addrIndex]= &cacheline_build;  
  *           LRU_head es un apuntador que apunta al apuntador de
  *           un cache_line, cuando escribimos LRU_head[index] esatamos
  *           seleccionando el contenido de un apuntador por lo que 
  *           LRU_head[index] solo será un apuntador a un cache_line.
  * - item= &cacheline_build;
  *           Es un apuntador de cache_line y por lo tanto podemos asignarle
  *           la dirección de un cache_line ya creado.
  * *head =&cacheline_build  || **head= cacheline_build;

  *
  */
  printf("Read Data\n");

  Pcache_line *head, item;
  cache_line  cacheline_build; 
  cacheline_build.tag =addrIndex;

  item= &cacheline_build;
  //**head=cacheline_build;

  insert(&c1.LRU_head[addrIndex], c1.LRU_tail[addrIndex], item );


  //c1.LRU_head[addrIndex]; //Falta crear un cline para poder acceder al tag ahora es null 
  //printf("\n HeadLine : %s",headLine);
  //*headLine->tag= addrTag;
  printf("\nitem tag: %d\n", item->tag );
  //printf("\nHead tag: %d\n", (**head).tag );

}


/************************************************************/
void flush()
{

  /* flush the cache */

}
/************************************************************/

/************************************************************/
void delete(head, tail, item)
  Pcache_line *head, *tail;
  Pcache_line item;
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
  item->LRU_next = *head; //*head=NULL cuando no hay elementos

  item->LRU_prev = NULL;

  if (item->LRU_next) //Si hay algun elemento en la linea
    item->LRU_next->LRU_prev = item;
  else
    *tail = item; 
    /*tail no se reubica y solo se asigna para el primer elemento
    * para mover el tail(jaja) debemos eliminar el ultimo elemento
    * (least bext) y el tail se asignara al prev del item:
    * tail = item->LRU_prev; 
    */


  *head = item;


}
/* ********************************************************** */

/************************************************************/
void dump_settings()
{
  printf("\n\n*** CACHE SETTINGS ***\n");
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
