#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 900000
#endif

/***************************************************************************************************************/
//MTF
// linked list to store dictionary
typedef struct linked_list{
	unsigned char val;
	struct linked_list* next;
	struct linked_list* prev;
} linked_list;
/***************************************************************************************************************/

/***************************************************************************************************************/
//functions supporting reverse MTF
void reverse_move_to_front(unsigned int inputBlockLength, linked_list** head, linked_list** tail, linked_list* dictionaryLinkedList, unsigned char *inputFileData, unsigned char *outputDataIndex);
/***************************************************************************************************************/

/***************************************************************************************************************/
//helper functions
void initialize_linked_list(linked_list** head, linked_list** tail, linked_list* dictionaryLinkedList);
unsigned char search_index(linked_list* dictionary, unsigned char characterAtIndex);
linked_list* swap_index(linked_list* dictionary, unsigned char value);
/***************************************************************************************************************/

