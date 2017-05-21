/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//revere MTF Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BLOCK_SIZE 900000

// linked list to store dictionary
struct linked_list{
	unsigned char val;
	struct linked_list* next;
	struct linked_list* prev;
};
typedef struct linked_list dictionary_linked_list;

//functions supporting reverse MTF
void initialize_linked_list(dictionary_linked_list** head, dictionary_linked_list** tail, unsigned char *dictionary, dictionary_linked_list* DictionaryLinkedList, unsigned int uniqueChars);
unsigned char search_index(dictionary_linked_list* dictionary, unsigned char characterAtIndex);
dictionary_linked_list* swap_index(dictionary_linked_list* dictionary, unsigned char value);

int main(int argc, char **argv){
	//time measurement
	clock_t start, end;
	unsigned int cpu_time_used;
	//dictionary information
	unsigned char dictionary[256];
	unsigned int uniqueChars;
	//file information
	unsigned int inputBlockLength;
	unsigned char inputFileData[BLOCK_SIZE], outputDataIndex[BLOCK_SIZE];
	FILE *inputFile, *outFile;
	//structure to hold dictionary data
	dictionary_linked_list *head = NULL, *tail = NULL, DictionaryLinkedList[256];

	// check parameters
	if(argc != 3){
		printf("Incorrect input parameters.  Require 3\n");
		return -1;
	}

	// start time measure
	start = clock();

	//read input file, output file
	inputFile = fopen(argv[1], "rb");
	outFile = fopen(argv[2], "wb");

	while( (fread(&uniqueChars, sizeof(unsigned int), 1, inputFile)) ){
		//read the dictionary and block of data
  	fread(&dictionary, sizeof(unsigned char), uniqueChars, inputFile);
		inputBlockLength = fread(inputFileData, sizeof(unsigned char), BLOCK_SIZE, inputFile);

		//store dictionary into linked list, can't use arrays because of insert at 0 position
		initialize_linked_list(&head, &tail, dictionary, DictionaryLinkedList, uniqueChars);

		// generate array list of indeces
		for(unsigned int i = 0; i < inputBlockLength; i++){
			outputDataIndex[i] = search_index(head, inputFileData[i]);
			head = swap_index(head, outputDataIndex[i]);
		}

		//write output  data
		fwrite(outputDataIndex, sizeof(unsigned char), inputBlockLength, outFile);
	}

	//close files
	fclose(inputFile);
	fclose(outFile);
	
	// end time measure
	end = clock();
	cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("Time taken: %d:%d s\n", cpu_time_used / 1000, cpu_time_used % 1000);
  return 0;
}

//store dictionary into linked list, can't use arrays because of insert at 0 position
void initialize_linked_list(dictionary_linked_list** head, dictionary_linked_list** tail, unsigned char *dictionary, dictionary_linked_list* DictionaryLinkedList, unsigned int uniqueChars){
	*head = &DictionaryLinkedList[0];
	DictionaryLinkedList[0].val = dictionary[0];
	DictionaryLinkedList[0].prev = NULL;
	if(uniqueChars > 1){
		DictionaryLinkedList[0].next = &DictionaryLinkedList[1];
	}

	for(unsigned int i = 1; i < uniqueChars - 1; i++){
		DictionaryLinkedList[i].val = dictionary[i];
		DictionaryLinkedList[i].prev = &DictionaryLinkedList[i - 1];
		DictionaryLinkedList[i].next = &DictionaryLinkedList[i + 1];
	}

	*tail = &DictionaryLinkedList[uniqueChars - 1];
	DictionaryLinkedList[uniqueChars - 1].val = dictionary[uniqueChars - 1];
	DictionaryLinkedList[uniqueChars - 1].next = NULL;

	if(uniqueChars > 1){
	DictionaryLinkedList[uniqueChars - 1].prev = &DictionaryLinkedList[uniqueChars - 2];
	}
}

// search and return the index
unsigned char search_index(dictionary_linked_list* dictionary, unsigned char characterAtIndex){
	unsigned int index = 0;
	while(index != characterAtIndex){
		dictionary = dictionary->next;
		index++;
	}
	return dictionary->val;
}

// swap dictionary and returns head
dictionary_linked_list* swap_index(dictionary_linked_list* dictionary, unsigned char value){
	dictionary_linked_list* head = dictionary;
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
