#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 900000
#endif

/***********************************************************************************************************/
void reverse_burrows_wheeler_transform(unsigned int inputBlockLength, unsigned char *inputBlockData, unsigned char *outputBlockData);
/***********************************************************************************************************/
