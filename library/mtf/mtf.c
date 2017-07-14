#include "mtf.h"

/******************************************************************************************************/
//store dictionary into linked list, can't use arrays because of insert at 0 position
void initialize_linked_list(linked_list** head, linked_list** tail, linked_list* dictionaryLinkedList){
  *head = &dictionaryLinkedList[0];
  *tail = &dictionaryLinkedList[255];

  dictionaryLinkedList[0].val = 0;
  dictionaryLinkedList[0].next = &dictionaryLinkedList[1];
  dictionaryLinkedList[0].prev = NULL;

  dictionaryLinkedList[255].val = 255;
  dictionaryLinkedList[255].next = NULL;
  dictionaryLinkedList[255].prev = &dictionaryLinkedList[254];

  for(unsigned int i = 1; i < 255; i++){
    dictionaryLinkedList[i].val = i;
    dictionaryLinkedList[i].prev = &dictionaryLinkedList[i - 1];
    dictionaryLinkedList[i].next = &dictionaryLinkedList[i + 1];
  }
}
/******************************************************************************************************/

/******************************************************************************************************/
// search and return the index
unsigned char search_value(linked_list* dictionary, unsigned char characterAtIndex){
	unsigned int index = 0;
	while(dictionary->val != characterAtIndex){
		dictionary = dictionary->next;
		index++;
	}
	return index;
}
/******************************************************************************************************/

/******************************************************************************************************/
// swap dictionary and returns head
linked_list* swap_value(linked_list* dictionary, unsigned char indexToSwap){
	unsigned int index = 0;
	linked_list* head = dictionary;
	while(index != indexToSwap){
		dictionary = dictionary->next;
		index++;
	}

	if(dictionary != head){
		// take care of previous and next node to the current
		dictionary->prev->next = dictionary->next;
		if(dictionary->next != NULL){
			dictionary->next->prev = dictionary->prev;
		}

		// take care of current
		dictionary->prev = NULL;
		dictionary->next = head;
		head->prev = dictionary;
	}
	return dictionary;
}
/******************************************************************************************************/

/******************************************************************************************************/
//performs move to front using helper functions
void move_to_front(unsigned int inputBlockLength, linked_list** head, linked_list** tail, linked_list* dictionaryLinkedList, unsigned char *inputFileData, unsigned char *outputDataIndex){
  //store dictionary into linked list, can't use arrays because of insert at 0 position
  initialize_linked_list(head, tail, dictionaryLinkedList);
  //generate array list of indeces
  //for each value in input file, find the location of the value in the dictionary and 
  //put that value in the front of the linked list
  for(unsigned int i = 0; i < inputBlockLength; i++){
	  outputDataIndex[i] = search_value(*head, inputFileData[i]);
	  *head = swap_value(*head, outputDataIndex[i]);
  }
}
/******************************************************************************************************/