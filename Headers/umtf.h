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

/***************************************************************************************************************/
void reverse_move_to_front(unsigned int inputBlockLength, linked_list** head, linked_list** tail, linked_list* dictionaryLinkedList, unsigned char *inputFileData, unsigned char *outputDataIndex){
	//store dictionary into linked list, can't use arrays because of insert at 0 position
	initialize_linked_list(head, tail, dictionaryLinkedList);

	// generate array list of indeces
	for(unsigned int i = 0; i < inputBlockLength; i++){
		outputDataIndex[i] = search_index(*head, inputFileData[i]);
		*head = swap_index(*head, outputDataIndex[i]);
	}
}
/***************************************************************************************************************/

/***************************************************************************************************************/
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
/***************************************************************************************************************/

/***************************************************************************************************************/
// search and return the index
unsigned char search_index(linked_list* dictionary, unsigned char characterAtIndex){
	unsigned int index = 0;
	while(index != characterAtIndex){
		dictionary = dictionary->next;
		index++;
	}
	return dictionary->val;
}
/***************************************************************************************************************/

/***************************************************************************************************************/
// swap dictionary and returns head
linked_list* swap_index(linked_list* dictionary, unsigned char value){
	linked_list* head = dictionary;
	while(value != dictionary->val){
		dictionary = dictionary->next;
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
/***************************************************************************************************************/