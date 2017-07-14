#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 900000
#endif

/*******************************************************************************************/
//global variables to be used in qsort function
extern unsigned int inputBlockLength;
//store input block data
extern unsigned char inputBlockData[BLOCK_SIZE];
/*******************************************************************************************/

/*******************************************************************************************/
//BWT functions
void burrows_wheeler_transform(unsigned char *outputBlockData);
//for the qsort
int compare( const unsigned int *index1, const unsigned int *index2 );
//to store int in char array
void serialize_int(unsigned char *outputBlockData, unsigned int value, unsigned int *count);
/*******************************************************************************************/
