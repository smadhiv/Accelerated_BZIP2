/*---------------------------------------------------------------------------------------------------------------------------------------------*/
//Sriram Madhivanan
//MTF Implementation
/*---------------------------------------------------------------------------------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BLOCK_SIZE 900000

//linked list to store dictionary
struct linked_list{
	unsigned char val;
	struct linked_list* next;
	struct linked_list* prev;
};
typedef struct linked_list dictionary_linked_list;

//functions supporting MTF
void initialize_dictionary(unsigned char *dictionary, unsigned char *inputFileData, unsigned int inputBlockLength, unsigned char *uniqueChars);
void initialize_linked_list(dictionary_linked_list** head, dictionary_linked_list** tail, dictionary_linked_list* DictionaryLinkedList, unsigned char uniqueChars);
unsigned char search_value(dictionary_linked_list* dictionary, unsigned char characterAtIndex);
dictionary_linked_list* swap_value(dictionary_linked_list* dictionary, unsigned char indexToSwap);

int main(int argc, char **argv){
	//to measure time
	clock_t start, end;
	unsigned int cpu_time_used;
	//dictionary
	unsigned char dictionary[256];
	//file information
	unsigned char uniqueChars;
	unsigned int inputBlockLength;
	unsigned char inputFileData[BLOCK_SIZE], outputDataIndex[BLOCK_SIZE];
	FILE *inputFile, *outFile;
	//structure to store each symbols
	dictionary_linked_list *head = NULL, *tail = NULL, DictionaryLinkedList[256];

	//check parameters
	if(argc != 3){
		printf("Incorrect input parameters.  Require 3\n");
		return -1;
	}

	//start time measure
	start = clock();

	//open input file, output file
	inputFile = fopen(argv[1], "rb");
	outFile = fopen(argv[2], "wb");
	
	//read one block at a time, process and write to output
	while( (inputBlockLength = fread(inputFileData, sizeof(unsigned char), BLOCK_SIZE, inputFile)) ){
		//reset dictionary and unique char values
		uniqueChars = 0;
		memset(dictionary, 255, 255);

		//initialize dictionary
		//generate dictionary of each symbols i.e set dictionary[n] = n, care when n = 255.
		//take a count of number of distinct characters in input data as uniqueChars
		//scope for optimization, work if removing the second 'if' statement below
		//dictionary should have a list of all unique elements in the input buffer
		//since max value possible in 255 and we stop the following loop at 254 we will always get the right dictionary
		initialize_dictionary(dictionary, inputFileData, inputBlockLength, &uniqueChars);

		//store dictionary into linked list, can't use arrays because of insert at 0 position
		initialize_linked_list(&head, &tail, DictionaryLinkedList, uniqueChars);

		//generate array list of indeces
		//for each value in input file, find the location of the value in the dictionary and 
		//put that value in the front of the linked list
		for(unsigned int i = 0; i < inputBlockLength; i++){
			outputDataIndex[i] = search_value(head, inputFileData[i]);
			head = swap_value(head, outputDataIndex[i]);
		}

		//write to output
		//1. number of unique characters,
		//2. array of each unique values and 
		//3. output data which is an array of indeces
		fwrite(&uniqueChars, sizeof(unsigned char), 1, outFile);
		fwrite(dictionary, sizeof(unsigned char), uniqueChars, outFile);
		fwrite(outputDataIndex, sizeof(unsigned char), inputBlockLength, outFile);
	}

	// end time measure
	end = clock();
	cpu_time_used = ((end - start)) * 1000 / CLOCKS_PER_SEC;
	printf("Time taken: %d:%d s\n", cpu_time_used / 1000, cpu_time_used % 1000);
	
	//close input file
	fclose(inputFile);	
	fclose(outFile);
  return 0;
}

//initialize dictionary
void initialize_dictionary(unsigned char *dictionary, unsigned char *inputFileData, unsigned int inputBlockLength, unsigned char *uniqueChars){
	//generate dictionary of each symbols i.e set dictionary[n] = n, care when n = 255.
	//take a count of number of distinct characters in input data as uniqueChars
	//scope for optimization, work if removing the second 'if' statement below
	for (unsigned int i = 0; i < inputBlockLength; i++){
		if(dictionary[inputFileData[i]] == 255){
			dictionary[inputFileData[i]] = inputFileData[i];
			uniqueChars++;
			if(inputFileData[i] == 255){
				dictionary[inputFileData[i]] = 0;
			}
		}
	}

	//dictionary should have a list of all unique elements in the input buffer
	//since max value possible in 255 and we stop the following loop at 254 we will always get the right dictionary
	for (unsigned int i = 0; i < 255; i++){
		if(dictionary[i] == 255){
			unsigned int j = i + 1;
			while(dictionary[j] == 255 && j < 255){
				j++;
			}
			if(j < 255){
				dictionary[i] = dictionary[j];
				dictionary[j] = 255;
			}
			else{
				break;
			}
		}
	}
}

//store dictionary into linked list, can't use arrays because of insert at 0 position
void initialize_linked_list(dictionary_linked_list** head, dictionary_linked_list** tail, dictionary_linked_list* DictionaryLinkedList, unsigned char uniqueChars){
	unsigned int listCount = 0;
	dictionary_linked_list *node;
	dictionary_linked_list *current = &DictionaryLinkedList[0];
	current->val = DictionaryLinkedList[0].val;
	current->prev = NULL;
	*head = current;

	for(unsigned int i = 1; i < uniqueChars; i++){
		node = &DictionaryLinkedList[++listCount];
		node->val = DictionaryLinkedList[i].val;
		current->next = node;
		node->prev = current;
		current = node;
	}
	current->next = NULL;
	*tail = current;
}

// search and return the index
unsigned char search_value(dictionary_linked_list* dictionary, unsigned char characterAtIndex){
	unsigned int index = 0;
	while(dictionary->val != characterAtIndex){
		dictionary = dictionary->next;
		index++;
	}
	return index;
}

// swap dictionary and returns head
dictionary_linked_list* swap_value(dictionary_linked_list* dictionary, unsigned char indexToSwap){
	unsigned int index = 0;
	dictionary_linked_list* head = dictionary;
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
